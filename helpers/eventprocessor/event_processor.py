import json
import logging
import os
import copy
import uuid

import numpy as np

from .utils import (
    encode_np_array,
    compute_softmax,
    compute_sigmoid,
    compute_argmax,
    compute_entropy,
    compute_margin_of_confidence,
    compute_ratio_of_confidence
)
from .performance_preprocessor import (
    preprocess_generic,
    preprocess_object_detection,
    preprocess_learning_to_rank
)

ADMIN_ORG_ID = os.environ.get('ADMIN_ORG_ID', None)

def process_event(json_event, organization_id):
    try:
        # Allows the admin org to upload events with another org_id.
        if organization_id != ADMIN_ORG_ID or 'organization_id' not in json_event:
            json_event['organization_id'] = organization_id

        # Turn prediction and groundtruth into single-element lists
        # for the convenience of accepting both single and multi-class.
        if 'prediction' in json_event:
            if not isinstance(json_event['prediction'], list):
                json_event['prediction'] = [json_event['prediction']]
            
            json_event['prediction'] = [p for p in json_event['prediction'] if p is not None]
        

        if 'groundtruth' in json_event:
            if not isinstance(json_event['groundtruth'], list):
                json_event['groundtruth'] = [json_event['groundtruth']]
            
            json_event['groundtruth'] = [g for g in json_event['groundtruth'] if g is not None]

        json_event['request_id'] = uuid.uuid4()

        # Decorate predictions with derived fields.
        for p in json_event.get('prediction', []):
            p.update(**process_prediction(p))

        # Generate prediction / groundtruth matches based on model_type.
        model_type = json_event.get('model_type', None)
        if model_type == 'OBJECT_DETECTION':
            processed_events = preprocess_object_detection(json_event)
        elif model_type == 'LEARNING_TO_RANK':
            processed_events = preprocess_learning_to_rank(json_event)
        else:
            processed_events = preprocess_generic(json_event)

        if 'embeddings' in json_event:
            if model_type == 'OBJECT_DETECTION':
                json_event['original_embeddings'] = encode_np_array(json_event['embeddings'])
            json_event['embeddings'] = encode_np_array(json_event['embeddings'], flatten=True, pool=True)

        json_event.pop('prediction', None)
        json_event.pop('groundtruth', None)

        return [json_event] + processed_events

    except Exception as e:
        # TODO: Send this to a log file for the user to see any ingestion errors.
        logging.error('Got an error for event ' + str(json_event))
        logging.exception(e)
        return []

def process_prediction(prediction):
    if 'logits' in prediction and prediction['logits']:
        if prediction['logits'] and len(prediction['logits']) == 1: # binary classifier
            positive_confidence = compute_sigmoid(prediction['logits']).tolist()
            prediction['confidences'] = [positive_confidence[0], 1 - positive_confidence[0]]
        else:
            prediction['confidences'] = compute_softmax(prediction['logits']).tolist()
        prediction['logits'] = encode_np_array(prediction['logits'], flatten=True)

    if 'confidences' in prediction and prediction['confidences']:
        confidence_vector = prediction['confidences']
        max_index = compute_argmax(confidence_vector)
        prediction['metrics'] = prediction.get('metrics', {})
        prediction['metrics']['entropy'] = compute_entropy(confidence_vector)
        prediction['metrics']['ratio_of_confidence'] = compute_ratio_of_confidence(confidence_vector)
        prediction['metrics']['margin_of_confidence'] = compute_margin_of_confidence(confidence_vector)
        prediction['confidence'] = confidence_vector[max_index]
        if 'class_names' in prediction:
            prediction['class_name'] = prediction['class_names'][max_index]

    if 'embeddings' in prediction and prediction['embeddings']:
        prediction['embeddings'] = encode_np_array(prediction['embeddings'], flatten=True)

    return prediction

def resolve_update(rows, update_event):

    if 'delete' in update_event: # this is a delete request
        return [], rows
    if len(rows) == 0: # empty data, do nothing
        return [], []
    if len(rows) == 1: # only a data row
        data_row = rows[0]
        annotation_rows = []
    else:
        annotation_rows = []
        for row in rows:
            if row.prediction is not None or row.groundtruth is not None:
                # annotation row
                annotation_rows.append(row)
            else:
                data_row = row

    if data_row.model_type == 'OBJECT_DETECTION':
        return [], [] # we don't do updates object detection models for now

    new_rows = []
    delete_rows = []
    if 'embeddings' in update_event:
        if update_event['embeddings'] is None:
            data_row.embeddings = None
        else:
            data_row.embeddings = encode_np_array(update_event['embeddings'], flatten=True, pool=True)
    if 'tags' in update_event:
        if update_event['tags'] is None:
            data_row.tags = None
        else:
            # Merge new tags with sqlalchemy tags
            data_row.tags = {
                **(data_row.tags),
                **(update_event['tags'])
            }
        if len(annotation_rows) > 0:
            # should only be one annotation row because we are in a classifier
            if update_event['tags'] is None:
                annotation_rows[0].tags = None
            else:
                # Merge new tags with sqlalchemy tags
                annotation_rows[0].tags = {
                    **(annotation_rows[0].tags),
                    **(update_event['tags'])
                }
    if 'groundtruth' in update_event or 'prediction' in update_event:
        if len(annotation_rows) == 0:
            new_row = copy.deepcopy(data_row.__dict__)
            new_row.pop('uuid')
            new_row.pop('embeddings')
            new_row.pop('_sa_instance_state')
            if 'groundtruth' in update_event:
                new_row['groundtruth'] = update_event['groundtruth']
            if 'prediction' in update_event:
                if update_event['prediction'] is None:
                    new_row['prediction'] = None
                else:
                    new_row['prediction'] =  process_prediction(update_event['prediction'])
            new_rows.append(new_row)
        else:
            # should only be one annotation row because we are in a classifier
            if 'groundtruth' in update_event:
                annotation_rows[0].groundtruth = update_event['groundtruth']
            if 'prediction' in update_event:
                if update_event['prediction'] is None:
                    annotation_rows[0].prediction = None
                else:
                    annotation_rows[0].prediction =  process_prediction(update_event['prediction'])

    if len(annotation_rows) > 0:
        if annotation_rows[0].groundtruth == None and annotation_rows[0].prediction == None:
            delete_rows.append(annotation_rows[0])

    return new_rows, delete_rows
