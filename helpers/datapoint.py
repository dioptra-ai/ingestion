from schemas.pgsql import models

Datapoint = models.datapoint.Datapoint

def process_datapoint(record, pg_session):
    organization_id = record['organization_id']

    type = 'IMAGE' if 'image_metadata' in record else 'VIDEO' if 'video_metadata' in record else 'AUDIO' if 'audio_metadata' in record else 'TEXT' if 'text_metadata' in record else None

    metadata = record['image_metadata'] if 'image_metadata' in record else record['video_metadata'] if 'video_metadata' in record else record['audio_metadata'] if 'audio_metadata' in record else record['text_metadata'] if 'text_metadata' in record else None

    if 'id' in record:
        datapoint = pg_session.query(Datapoint).filter(Datapoint.id == record['id'])
        if metadata:
            datapoint.metadata = metadata
        if type:
            datapoint.type = type
    else:
        datapoint = Datapoint(
            organization_id=organization_id, 
            type=type, 
            metadata=metadata
        )
        pg_session.add(datapoint)
    
    return datapoint
