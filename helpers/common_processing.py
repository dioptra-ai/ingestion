import datetime


def process(event, organization_id):
    event['organization_id'] = organization_id
    event['processing_timestamp'] = datetime.datetime.utcnow().isoformat()

    return event