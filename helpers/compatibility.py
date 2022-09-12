def process(event):
    # TODO: rename the '__time' column to 'timestamp'
    if '__time' in event:
        event['timestamp'] = event.pop('__time')

    event.pop('committed', None)

    for key in ['features', 'tags', 'image_metadata', 'text_metadata', 'audio_metadata', 'video_metadata', 'groundtruth', 'prediction']:
        unflatten_dict(key, event)

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
