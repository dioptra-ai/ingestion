def process(event):
    event.pop('committed', None)
    event.pop('__time', None)  # Will be migrated to "timestamp"

    for key in ['features', 'tags', 'image_metadata', 'text_metadata', 'audio_metadata', 'video_metadata']:
        unflatten_dict(key, event)

    return event

def unflatten_path(path, value):
    if len(path) == 1:
        return {
            path[0]: value
        };
    else:
        return {
            path[0]: unflatten_path(path[1:], value)
        };

def unflatten_dict(prefix, my_dict):
    key_values = list(my_dict.items())
    for key, value in key_values:
        if key.startswith(f'{prefix}.'):
            my_dict[prefix] = unflatten_path(key.split('.')[1:], value)
            del my_dict[key]

    return my_dict
