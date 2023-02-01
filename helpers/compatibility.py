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
    
    # Backward-compatibility for string classes.
    if 'prediction' in event and isinstance(event['prediction'], str):
        event['prediction'] = [{
            'class_name': event['prediction']
        }]
    if 'groundtruth' in event and isinstance(event['groundtruth'], str):
        event['groundtruth'] = [{
            'class_name': event['groundtruth']
        }]
    
    # Turn prediction and groundtruth into single-element lists
    # for the convenience of accepting both single and multi-class.
    if 'prediction' in event and not isinstance(event['prediction'], list):
        event['prediction'] = [event['prediction']]

    if 'groundtruth' in event and not isinstance(event['groundtruth'], list):
        event['groundtruth'] = [event['groundtruth']]

    # Backward-compatibility for top-level confidence.
    if 'confidence' in event and isinstance(event.get('prediction', None), dict) and not 'confidence' in event['prediction']:
        event['prediction']['confidence'] = event.pop('confidence')

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
