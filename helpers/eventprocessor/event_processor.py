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
        if 'prediction' in json_event and not isinstance(json_event['prediction'], list):
            json_event['prediction'] = [json_event['prediction']]

        if 'groundtruth' in json_event and not isinstance(json_event['groundtruth'], list):
            json_event['groundtruth'] = [json_event['groundtruth']]

        json_event['request_id'] = uuid.uuid4()

        # Decorate predictions with derived fields.
        for p in json_event.get('prediction', []):
            p = process_prediction(p)

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
    if 'logits' in prediction:
        if len(prediction['logits']) == 1: # binary classifier
            positive_confidence = compute_sigmoid(prediction['logits']).tolist()
            prediction['confidences'] = [positive_confidence[0], 1 - positive_confidence[0]]
        else:
            prediction['confidences'] = compute_softmax(prediction['logits']).tolist()
        prediction['logits'] = encode_np_array(prediction['logits'], flatten=True)

    if 'confidences' in prediction:
        confidence_vector = prediction['confidences']
        max_index = compute_argmax(confidence_vector)
        prediction['metrics'] = prediction.get('metrics', {})
        prediction['metrics']['entropy'] = compute_entropy(confidence_vector)
        prediction['metrics']['ratio_of_confidence'] = compute_ratio_of_confidence(confidence_vector)
        prediction['metrics']['margin_of_confidence'] = compute_margin_of_confidence(confidence_vector)
        prediction['confidence'] = confidence_vector[max_index]
        if 'class_names' in prediction:
            prediction['class_name'] = prediction['class_names'][max_index]

    if 'embeddings' in prediction:
        prediction['embeddings'] = encode_np_array(prediction['embeddings'], flatten=True)

    return prediction


def resolve_update(rows, update_event):

    for row in rows:
        # we only support update for classifier for now so there should be only 1 row per request_id
        if row['model_type'] == 'CLASSIFIER':
            if 'tags' in update_event:
                row['tags'] = update_event['tags']
            if 'groundtruth' in update_event:
                row['groundtruth'] = update_event['groundtruth']
            if 'prediction' in update_event:

            if 'embeddings' in update_event:
                row['embeddings'] = encode_np_array(update_event['embeddings'], flatten=True, pool=True)
                if ('prediction' in row and row['prediction'] is not None) or \
                   ('groundtruth' in row and row['groundtruth'] is not None):
                    row['groundtruth'] = update_event['groundtruth']
                    need_to_update_annotation = False
                else:
                    datapoint_row = row

            if

        updated_rows.append(row)

