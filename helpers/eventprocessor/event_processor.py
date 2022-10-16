import json
import logging
import os
import copy
import uuid

from .utils import (
    encode_np_array,
    compute_softmax,
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

        # Turn prediction and groundtruth into single-element lists if necessary.
        if 'prediction' in json_event and not isinstance(json_event['prediction'], list):
            json_event['prediction'] = [json_event['prediction']]

        if 'groundtruth' in json_event and not isinstance(json_event['groundtruth'], list):
            json_event['groundtruth'] = [json_event['groundtruth']]
        
        if not 'request_id' in json_event:
            json_event['request_id'] = uuid.uuid4()

        # Decorate predictions with derived fields.
        for p in json_event.get('prediction', []):
            if 'logits' in p:
                p['confidences'] = compute_softmax(p['logits']).tolist()
                p['logits'] = encode_np_array(p['logits'], flatten=True)

            if 'confidences' in p:
                box_confidences = p['box_confidences']
                max_index = compute_argmax(box_confidences)
                p['entropy'] = compute_entropy(box_confidences)
                p['ratio_of_confidence'] = compute_ratio_of_confidence(box_confidences)
                p['margin_of_confidence'] = compute_margin_of_confidence(box_confidences)
                p['confidence'] = box_confidences[max_index]
                if 'class_names' in p:
                    p['class_name'] = p['class_names'][max_index]

            if 'embeddings' in p:
                p['embeddings'] = encode_np_array(p['embeddings'], flatten=True)

        # Generate prediction / groundtruth matches based on model_type.
        model_type = json_event.get('model_type', None)
        if model_type == 'OBJECT_DETECTION':
            processed_events = preprocess_object_detection(json_event)
        elif model_type == 'LEARNING_TO_RANK':
            processed_events = preprocess_learning_to_rank(json_event)
        else:
            processed_events = preprocess_generic(json_event)

        if 'embeddings' in json_event:
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
