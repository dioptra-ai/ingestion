from datetime import datetime
import json

def process(event):

    if '__time' in event:
        event['timestamp'] = event.pop('__time')

    # Support for UNIX timestamps in seconds and ms.
    if 'timestamp' in event and isinstance(event['timestamp'], int):
        try: # Try to parse as seconds.
            event['timestamp'] = datetime.utcfromtimestamp(event['timestamp']).strftime('%Y-%m-%d %H:%M:%S.%fZ')
        except ValueError: # Try to parse as ms.
            event['timestamp'] = datetime.utcfromtimestamp(event['timestamp'] / 1000).strftime('%Y-%m-%d %H:%M:%S.%fZ')

    # Backward-compatibility for flat fields.
    for key in ['features', 'tags', 'image_metadata', 'text_metadata', 'audio_metadata', 'video_metadata', 'groundtruth', 'prediction']:
        unflatten_dict(key, event)
        
    # Turn prediction and groundtruth into single-element lists
    # for the convenience of accepting both single and multi-class.
    if 'prediction' in event and not isinstance(event['prediction'], list):
        event['prediction'] = [event['prediction']]

    if 'predictions' not in event and 'prediction' in event:
        event['predictions'] = event['prediction']
    
    if 'predictions' in event:
        event['predictions'] = [p for p in event['predictions'] if p is not None]

    for i, prediction in enumerate(event.get('predictions', [])):
        # Backward-compatibility for string classes.
        if isinstance(prediction, str):
            event['predictions'][i] = {
                'class_name': prediction
            }
        prediction = event['predictions'][i]
        # Backward-compatibility for typed datapoints
        if not 'task_type' in prediction:
            if 'top' in prediction or 'left' in prediction or 'bottom' in prediction or 'right' in prediction:
                prediction['task_type'] = 'OBJECT_DETECTION'
            elif 'class_name' in prediction:
                prediction['task_type'] = 'CLASSIFICATION'
    
    if 'groundtruth' in event and not isinstance(event['groundtruth'], list):
        event['groundtruth'] = [event['groundtruth']]

    if 'groundtruths' not in event and 'groundtruth' in event:
        event['groundtruths'] = event['groundtruth']

    if 'groundtruths' in event:
        event['groundtruths'] = [g for g in event['groundtruths'] if g is not None]

    for i, groundtruth in enumerate(event.get('groundtruths', [])):
        # Backward-compatibility for string classes.
        if isinstance(groundtruth, str):
            event['groundtruths'][i] = {
                'class_name': groundtruth
            }
        groundtruth = event['groundtruths'][i]
        # Backward-compatibility for typed datapoints
        if not 'task_type' in groundtruth:
            if 'top' in groundtruth or 'left' in groundtruth or 'bottom' in groundtruth or 'right' in groundtruth:
                groundtruth['task_type'] = 'OBJECT_DETECTION'
            elif 'class_name' in groundtruth:
                groundtruth['task_type'] = 'CLASSIFICATION'

    # Backward-compatibility for top-level confidence.
    if 'confidence' in event and isinstance(event.get('prediction', None), dict) and not 'confidence' in event['prediction']:
        event['prediction']['confidence'] = event.pop('confidence')
    
    # Backward-compatibility for typed datapoints
    if 'image_metadata' in event:
        event['metadata'] = event['image_metadata']
        event['type'] = 'IMAGE'
    if 'video_metadata' in event:
        event['metadata'] = event['video_metadata']
        event['type'] = 'VIDEO'
    if 'audio_metadata' in event:
        event['metadata'] = event['audio_metadata']
        event['type'] = 'AUDIO'
    if 'text_metadata' in event:
        event['metadata'] = event['text_metadata']
        event['type'] = 'TEXT'

    return event

def unflatten_path(path, value):

    return {
        path[0]: value if len(path) == 1 else unflatten_path(path[1:], value)
    };

def unflatten_dict(prefix, my_dict):
    key_values = list(my_dict.items())
    for key, value in key_values:
        if key.startswith(f'{prefix}.'):

            if prefix not in my_dict:
                my_dict[prefix] = {}

            my_dict[prefix].update(unflatten_path(key.split('.')[1:], value))

            del my_dict[key]
