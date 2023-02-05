import json
from sqlalchemy import text

from schemas.pgsql import models

from .eventprocessor.utils import (
    encode_np_array
)

Tag = models.tag.Tag
Datapoint = models.datapoint.Datapoint
FeatureVector = models.feature_vector.FeatureVector

def process_datapoint(record, pg_session):
    organization_id = record['organization_id']

    if 'id' in record:
        datapoint = pg_session.get(Datapoint, record['id'])

        if not datapoint:
            raise Exception(f"Datapoint {record['id']} not found")
        
        datapoint_id = datapoint.id

        update_values = {}
        if 'metadata' in record:
            update_values['metadata'] = json.dumps({
                **(datapoint.metadata_ or {}),
                **record['metadata']
            })
        if 'type' in record:
            update_values['type'] = record['type']

        if update_values:
            pg_session.execute(
                text(
                    f'UPDATE datapoints SET {", ".join([f"{key} = :{key}" for key in update_values])} WHERE id = :id'
                ),
                {
                    'id': datapoint_id, 
                    **update_values
                }
            )
    else:
        datapoint = pg_session.execute(
            text(
                'INSERT INTO datapoints (organization_id, type, metadata) VALUES (:organization_id, :type, :metadata) RETURNING *'
            ),
            {
                'organization_id': organization_id,
                'type': record.get('type'),
                'metadata': json.dumps(record.get('metadata'))
            }
        ).first()
        datapoint_id = datapoint.id

    if 'tags' in record:
        tags = record['tags']
        if tags is None:
            pg_session.query(Tag).filter(Tag.datapoint == datapoint_id).delete()
        else:
            for tag in tags:
                if tags[tag] is None:
                    pg_session.query(Tag).filter(Tag.datapoint == datapoint_id, Tag.name == tag).delete()
                else:
                    pg_session.execute(
                        text(
                            'INSERT INTO tags (organization_id, datapoint, name, value) VALUES (:organization_id, :datapoint, :name, :value) ON CONFLICT (datapoint, name) DO UPDATE SET value = :value'
                        ),
                        {
                            'organization_id': organization_id,
                            'datapoint': datapoint_id,
                            'name': tag,
                            'value': tags[tag]
                        }
                    )
    
    if 'embeddings' in record:
        embeddings = record['embeddings']
        if embeddings is None:
            pg_session.query(FeatureVector).filter(FeatureVector.datapoint == datapoint_id, FeatureVector.type == 'EMBEDDINGS').delete()
        else:
            pg_session.execute(
                text(
                    'DELETE FROM feature_vectors WHERE datapoint = :datapoint_id AND type = :type'
                ),
                {
                    'datapoint_id': datapoint_id,
                    'type': 'EMBEDDINGS'
                }
            )
            
            pg_session.add(FeatureVector(
                organization_id=organization_id,
                datapoint=datapoint_id,
                type='EMBEDDINGS',
                value=encode_np_array(embeddings, flatten=True)
            ))

    return datapoint_id
