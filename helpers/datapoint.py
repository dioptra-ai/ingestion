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

    if 'id' in record:
        datapoint = pg_session.get(Datapoint, record['id'])
        if not datapoint:
            raise Exception(f"Datapoint {record['id']} not found")
    else:
        datapoint = Datapoint(organization_id=organization_id)
        pg_session.add(datapoint)
        pg_session.flush()

    if 'metadata' in record:
        datapoint.metadata_ = {
            **(datapoint.metadata_ or {}),
            **record['metadata']
        }

    if 'type' in record:
        datapoint.type = record['type']

    if 'tags' in record:
        tags = record['tags']

        if tags is None:
            pg_session.query(Tag).filter(Tag.datapoint == datapoint.id).delete()
        else:
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
                            set_={'value': tags[tag]}
                        )
                    )

    if 'embeddings' in record:
        embeddings = record['embeddings']

        if embeddings is None:
            pg_session.query(FeatureVector).filter(
                FeatureVector.datapoint == datapoint.id, 
                FeatureVector.type == 'EMBEDDINGS',
                FeatureVector.model_name == None # TODO: take embeddings as a dict {value, model_name}
            ).delete()
        else:
            insert_statement = insert(FeatureVector).values(
                organization_id=organization_id,
                datapoint=datapoint.id,
                type='EMBEDDINGS',
                value=encode_np_array(embeddings, flatten=True),
                model_name=None
            )
            pg_session.execute(insert_statement.on_conflict_do_update(
                constraint='feature_vectors_datapoint_model_name_type_unique',
                set_={'value': insert_statement.excluded.value}
            ))

    return datapoint.id