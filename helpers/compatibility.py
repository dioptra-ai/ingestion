def process(event):
    event.pop('committed', None)

    return event