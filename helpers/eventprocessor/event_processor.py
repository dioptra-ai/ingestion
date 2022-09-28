import logging
import os
import uuid
import datetime

from .utils import (
    encode_np_array,
    in_place_walk_decode_embeddings
)
from .performance_preprocessor import (
    preprocess_object_detection,
    preprocess_question_answering,
    preprocess_automated_speech_recognition,
    preprocess_auto_completion,
    preprocess_semantic_similarity,
    preprocess_classifier
)

ADMIN_ORG_ID = os.environ.get('ADMIN_ORG_ID', None)

def process_event(json_event, organization_id):

    try:
        # Allows the admin org to upload events with another org_id.
        if organization_id != ADMIN_ORG_ID or 'organization_id' not in json_event:
            json_event['organization_id'] = organization_id

        json_event['processing_timestamp'] = datetime.datetime.utcnow().isoformat()

        in_place_walk_decode_embeddings(json_event)

        if 'embeddings' in json_event:
            embeddings = json_event.pop('embeddings')
            json_event['non_encoded_embeddings'] = embeddings
            json_event['embeddings'] = encode_np_array(embeddings, flatten=True, pool=True)

            # print(decode_np_array(json_event['embeddings']))

        processed_events = [json_event]

        if 'model_type' in json_event \
                and 'prediction' in json_event \
                and 'groundtruth' not in json_event:

            # unsupervised input

            if json_event['model_type'] == 'CLASSIFIER':
                processed_events = preprocess_classifier(json_event)
            elif json_event['model_type'] == 'OBJECT_DETECTION':
                processed_events = preprocess_object_detection(json_event)

        elif 'model_type' in json_event \
                and 'groundtruth' in json_event \
                and 'prediction' in json_event:

            # supervised input

            if json_event['model_type'] == 'ASR':
                processed_events = preprocess_automated_speech_recognition(json_event)

            elif json_event['model_type'] == 'OBJECT_DETECTION':
                processed_events = preprocess_object_detection(json_event)

            elif json_event['model_type'] == 'QUESTION_ANSWERING':
                processed_events = preprocess_question_answering(json_event)

            elif json_event['model_type'] == 'AUTO_COMPLETION':
                processed_events = preprocess_auto_completion(json_event)

            elif json_event['model_type'] == 'SEMANTIC_SIMILARITY':
                processed_events = preprocess_semantic_similarity(json_event)

            elif json_event['model_type'] == 'MULTIPLE_OBJECT_TRACKING':
                # TODO: Be smarter with MOT and do the Hungarian algorithm:
                # match previous detections to the same ground truth if their
                # iou is >= 0.5 even though another detection might be closer.
                processed_events = preprocess_object_detection(json_event)
            elif json_event['model_type'] == 'CLASSIFIER':
                processed_events = preprocess_classifier(json_event)
        else:

            if 'groundtruth' in json_event and 'prediction' in json_event \
                    and isinstance(json_event['groundtruth'], list) \
                    and isinstance(json_event['prediction'], list):

                processed_events = preprocess_object_detection(json_event)

            if 'groundtruth' in json_event and 'prediction' in json_event \
                    and isinstance(json_event['groundtruth'], dict) \
                    and isinstance(json_event['prediction'], list):

                processed_events = preprocess_question_answering(json_event)

        for my_event in processed_events:
            try:
                my_event['uuid'] = str(uuid.UUID(my_event['uuid'], version=4))
            except (ValueError, KeyError):
                my_event['uuid'] = str(uuid.uuid4())
            # TODO: Remove this when all the object detection logic is moved into the metrics engine.
            my_event.pop('non_encoded_embeddings', None)

        return processed_events

    except Exception as e:
        # TODO: Send this to a log file for the user to see any ingestion errors.
        logging.error('Got an error for event ' + str(json_event))
        logging.exception(e)
        return []
