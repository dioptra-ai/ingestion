import uuid
import numpy as np

from sqlalchemy.dialects.postgresql import insert

from schemas.pgsql import models

from helpers.eventprocessor.utils import (
    encode_np_array,
    decode_to_np_array,
    compute_argmax,
    compute_entropy,
    process_logits,
    resize_mask,
    encode_list
)

from helpers.metrics import segmentation_distribution

Prediction = models.prediction.Prediction
FeatureVector = models.feature_vector.FeatureVector

def process_prediction_records(records, datapoint, pg_session):
    predictions = []

    for p in records:
        if 'id' in p:
            prediction = pg_session.query(Prediction).filter(Prediction.id == p['id']).first()
            if not prediction:
                raise Exception(f"Prediction {p['id']} not found")
        else:
            prediction = Prediction(
                organization_id=datapoint.organization_id,
                datapoint=datapoint.id,
                task_type=p['task_type'],
                # This is needed otherwise pg_session.flush() will fail
                # trying to insert a prediction with a '' model_name when
                # '' already exists in the db and the model_name is provided in p.
                model_name=p.get('model_name', '')
            )

            pg_session.add(prediction)
            pg_session.flush()

        predictions.append(prediction)

        if '_preprocessor' in p:
            prediction._preprocessor = p['_preprocessor']

        if 'task_type' in p:
            prediction.task_type = p['task_type']

        if prediction.id is not None: # in mock sqlalchemy the id is None
            # Overriding predictions with the same datapoint id and the same model name.
            if prediction.task_type == 'CLASSIFICATION' or prediction.task_type == 'SEGMENTATION':
                pg_session.query(Prediction).filter(
                    Prediction.datapoint == datapoint.id,
                    Prediction.task_type == prediction.task_type,
                    Prediction.model_name == p.get('model_name', ''),
                    Prediction.id != prediction.id
                ).delete()

        if 'confidence' in p:
            prediction.confidence = p['confidence']
        if 'class_names' in p:
            if not isinstance(p['class_names'], list) and not p['class_names'] is None:
                raise Exception(f"class_names must be a list or null. Got {type(p['class_names'])}")
            prediction.class_names = p['class_names']
        if 'class_name' in p:
            prediction.class_name = p['class_name']
        if 'top' in p:
            prediction.top = p['top']
        if 'left' in p:
            prediction.left = p['left']
        if 'height' in p:
            prediction.height = p['height']
        if 'width' in p:
            prediction.width = p['width']
        if 'model_name' in p:
            prediction.model_name = p['model_name']

        if 'confidences' in p:
            confidence_vector = p['confidences']

            prediction.confidences = confidence_vector

            if confidence_vector is None:
                prediction.metrics = None
            else:
                max_index = compute_argmax(confidence_vector)
                prediction.confidence = confidence_vector[max_index]
                if p.get('class_names'):
                    prediction.class_name = p['class_names'][max_index]

                prediction.metrics = {**prediction.metrics} if prediction.metrics else {} # Changes the property reference otherwise sqlalchemy doesn't send an INSERT.
                prediction.metrics['entropy'] = compute_entropy(confidence_vector)

        if 'encoded_logits' in p:
            if isinstance(p['encoded_logits'], list):
                # mc dropout
                p['logits'] = [decode_to_np_array(encoded_logits).astype(np.float32).tolist() for encoded_logits in p['encoded_logits']]
            else:
                p['logits'] = decode_to_np_array(p['encoded_logits']).astype(np.float32).tolist()

        if 'logits' in p:
            logits = p['logits']

            if logits is None or np.array(logits).size == 0:
                pg_session.query(FeatureVector).filter(
                    FeatureVector.prediction == prediction.id,
                    FeatureVector.type == 'LOGITS',
                    FeatureVector.model_name == prediction.model_name
                ).delete()
            else:
                logits_results = process_logits(logits)
                prediction.metrics = {**prediction.metrics} if prediction.metrics else {} # Changes the property reference otherwise sqlalchemy doesn't send an INSERT.

                if 'entropy' in logits_results:
                    prediction.metrics['entropy'] = logits_results['entropy']
                if 'variance' in logits_results:
                    prediction.metrics['variance'] = logits_results['variance']
                if 'class_name' in logits_results:
                    prediction.class_name = logits_results['class_name']
                if 'confidence' in logits_results:
                    prediction.confidence = logits_results['confidence']
                if 'confidences' in logits_results:
                    prediction.confidences = logits_results['confidences']
                if 'segmentation_class_mask' in logits_results:
                    p['segmentation_class_mask'] = logits_results['segmentation_class_mask']
                if 'pixel_entropy' in logits_results:

                    insert_statement = insert(FeatureVector).values(
                        organization_id=datapoint.organization_id,
                        type='PXL_ENTROPY',
                        prediction=prediction.id,
                        encoded_value=encode_list(resize_mask(logits_results['pixel_entropy'])),
                        model_name=p.get('model_name', '')
                    )
                    pg_session.execute(insert_statement.on_conflict_do_update(
                        constraint='feature_vectors_prediction_model_name_type_unique',
                        set_={
                            'id': uuid.uuid4(),
                            'encoded_value': insert_statement.excluded.encoded_value
                        }
                    ))

                if 'pixel_variance' in logits_results:

                    insert_statement = insert(FeatureVector).values(
                        organization_id=datapoint.organization_id,
                        type='PXL_VARIANCE',
                        prediction=prediction.id,
                        encoded_value=encode_list(resize_mask(logits_results['pixel_variance'])),
                        model_name=p.get('model_name', '')
                    )
                    pg_session.execute(insert_statement.on_conflict_do_update(
                        constraint='feature_vectors_prediction_model_name_type_unique',
                        set_={
                            'id': uuid.uuid4(),
                            'encoded_value': insert_statement.excluded.encoded_value
                        }
                    ))

                insert_statement = insert(FeatureVector).values(
                    organization_id=datapoint.organization_id,
                    type='LOGITS',
                    prediction=prediction.id,
                    encoded_value=encode_np_array(logits),
                    model_name=prediction.model_name
                )
                pg_session.execute(insert_statement.on_conflict_do_update(
                    constraint='feature_vectors_prediction_model_name_type_unique',
                    set_={
                        'id': uuid.uuid4(),
                        'encoded_value': insert_statement.excluded.encoded_value
                    }
                ))

        if 'segmentation_class_mask' in p:
            segmentation_class_mask = p['segmentation_class_mask']
            if segmentation_class_mask and np.array(segmentation_class_mask).size > 0:
                prediction.encoded_segmentation_class_mask = encode_np_array(p['segmentation_class_mask'])
                prediction.encoded_resized_segmentation_class_mask = encode_list(resize_mask(p['segmentation_class_mask']))
                prediction.metrics = {**prediction.metrics} if prediction.metrics else {} # Changes the property reference otherwise sqlalchemy doesn't send an INSERT.
                prediction.metrics['distribution'] = segmentation_distribution(p['segmentation_class_mask'], prediction.class_names)

        if 'embeddings' in p:
            embeddings = p['embeddings']

            if not embeddings or np.array(embeddings).size == 0:
                pg_session.query(FeatureVector).filter(
                    FeatureVector.prediction == prediction.id,
                    FeatureVector.type == 'EMBEDDINGS'
                ).delete()
            else:
                if type(embeddings) is list:
                    embeddings = {
                        '': embeddings
                    }
                
                for layer_name in embeddings:
                    layer_embeddings = embeddings[layer_name]
                    embeddings_model_name = prediction.model_name + (f':{layer_name}' if layer_name else '')

                    if not layer_embeddings or np.array(layer_embeddings).size == 0:
                        pg_session.query(FeatureVector).filter(
                            FeatureVector.prediction == prediction.id,
                            FeatureVector.type == 'EMBEDDINGS',
                            FeatureVector.model_name == embeddings_model_name
                        ).delete()
                    else:
                        insert_statement = insert(FeatureVector).values(
                            organization_id=datapoint.organization_id,
                            type='EMBEDDINGS',
                            prediction=prediction.id,
                            encoded_value=encode_np_array(layer_embeddings),
                            model_name=prediction.model_name + (f':{layer_name}' if layer_name else '')
                        )
                        pg_session.execute(insert_statement.on_conflict_do_update(
                            constraint='feature_vectors_prediction_model_name_type_unique',
                            set_={
                                'id': uuid.uuid4(),
                                'encoded_value': insert_statement.excluded.encoded_value
                            }
                        ))
    
    return predictions
