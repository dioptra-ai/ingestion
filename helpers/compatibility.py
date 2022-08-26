def process(event):
    event.pop('committed', None)
    event.pop('__time', None) # Will be migrated to "timestamp"

    return event