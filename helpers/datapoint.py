from sqlalchemy import inspect
from sqlalchemy.dialects.postgresql import insert

from schemas.pgsql import models

from .eventprocessor.utils import (
    encode_np_array
)

Tag = models.tag.Tag
Datapoint = models.datapoint.Datapoint
FeatureVector = models.feature_vector.FeatureVector

def process_datapoint(record, pg_session):
    organization_id = record['organization_id']

    type = 'IMAGE' if 'image_metadata' in record else 'VIDEO' if 'video_metadata' in record else 'AUDIO' if 'audio_metadata' in record else 'TEXT' if 'text_metadata' in record else None

    metadata = record['image_metadata'] if 'image_metadata' in record else record['video_metadata'] if 'video_metadata' in record else record['audio_metadata'] if 'audio_metadata' in record else record['text_metadata'] if 'text_metadata' in record else None

    if 'id' in record:
        datapoint = pg_session.query(Datapoint).filter(Datapoint.id == record['id']).first()
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

    if 'tags' in record:
        tags = record['tags']
        for tag in tags:
            if tags[tag] is None:
                pg_session.query(Tag).filter(Tag.datapoint == datapoint.id, Tag.name == tag).delete()
            else:
                pg_session.execute(
                    insert(Tag).values(
                        organization_id=organization_id,
                        datapoint=datapoint.id,
                        name=tag,
                        value=tags[tag]
                    ).on_conflict_do_update(
                        constraint='tags_datapoint_name_unique',
                        set_=dict(value=tags[tag])
                    )
                )
    
    if 'embeddings' in record:
        embeddings = record['embeddings']
        
        if inspect(datapoint).persistent:
            pg_session.query(FeatureVector).filter(FeatureVector.datapoint == datapoint.id and FeatureVector.name == 'embeddings').delete()
        
        pg_session.add(FeatureVector(
            organization_id=organization_id,
            datapoint=datapoint.id,
            name='embeddings',
            value=encode_np_array(embeddings, flatten=True)
        ))

    return datapoint
