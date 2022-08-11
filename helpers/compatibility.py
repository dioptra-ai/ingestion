def process(event):
    event.pop('comitted', None)

    return event