import uuid
from sqlalchemy.dialects.postgresql import insert

from schemas.pgsql import models

from helpers.eventprocessor.utils import (
    encode_np_array
)

Tag = models.tag.Tag
Datapoint = models.datapoint.Datapoint
FeatureVector = models.feature_vector.FeatureVector

def process_datapoint_record(record, pg_session):
    organization_id = record['organization_id']

    if 'id' in record:
        datapoint = pg_session.get(Datapoint, record['id'])
        if not datapoint:
            raise Exception(f"Datapoint {record['id']} not found")
    else:
        datapoint = Datapoint(organization_id=organization_id)
        pg_session.add(datapoint)
        pg_session.flush()

    if '_preprocessor' in record:
        datapoint._preprocessor = record['_preprocessor']

    if 'metadata' in record:
        metadata = record['metadata']
        datapoint.metadata_ = {}

        if metadata is not None:
            for key in metadata:
                if metadata[key] is None:
                    datapoint.metadata_.pop(key, None)
                else:
                    datapoint.metadata_[key] = metadata[key]

    if 'type' in record:
        datapoint.type = record['type']

    if 'text' in record:
        datapoint.text = record['text']

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
                 # TODO: take embeddings as a dict {value, model_name}
                #  Today default model_name is ''
                # FeatureVector.model_name == model_name
            ).delete()
        else:
            insert_statement = insert(FeatureVector).values(
                organization_id=organization_id,
                datapoint=datapoint.id,
                type='EMBEDDINGS',
                encoded_value=encode_np_array(embeddings, flatten=True),
                # TODO: take embeddings as a dict {value, model_name}
                #  Today default model_name is ''
                # model_name=model_name
            )
            pg_session.execute(insert_statement.on_conflict_do_update(
                constraint='feature_vectors_datapoint_model_name_type_unique',
                set_={
                    'id': uuid.uuid4(),
                    'encoded_value': insert_statement.excluded.encoded_value
                }
            ))

    return datapoint
