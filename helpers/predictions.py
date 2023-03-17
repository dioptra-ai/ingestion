import uuid

from sqlalchemy.dialects.postgresql import insert

from schemas.pgsql import models

from .eventprocessor.utils import (
    encode_np_array,
    compute_argmax,
    compute_entropy,
    compute_margin_of_confidence,
    compute_ratio_of_confidence,
    process_logits
)

from .metrics import segmentation_distribution

Prediction = models.prediction.Prediction
FeatureVector = models.feature_vector.FeatureVector

def process_predictions(record, datapoint_id, pg_session):
    organization_id = record['organization_id']

    for p in record.get('predictions', []):
        if 'id' in p:
            prediction = pg_session.query(Prediction).filter(Prediction.id == p['id']).first()
            if not prediction:
                raise Exception(f"Prediction {p['id']} not found")
        else:
            prediction = Prediction(
                organization_id=organization_id,
                datapoint=datapoint_id,
                task_type=p['task_type'],
                # This is needed otherwise pg_session.flush() will fail 
                # trying to insert a prediction with a '' model_name when
                # '' already exists in the db and the model_name is provided in p.
                model_name=p.get('model_name')
            )
            pg_session.add(prediction)
            pg_session.flush()

        if 'task_type' in p:
            prediction.task_type = p['task_type']
        if 'confidences' in p:
            prediction.confidences = p['confidences']
        if 'confidence' in p:
            prediction.confidence = p['confidence']
        if 'class_names' in p:
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

        if 'logits' in p:
            logits = p['logits']

            if logits is None:
                pg_session.query(FeatureVector).filter(
                    FeatureVector.prediction == prediction.id,
                    FeatureVector.type == 'LOGITS',
                    FeatureVector.model_name == p.get('model_name', '')
                ).delete()
            else:
                p['confidences'], p['segmentation_class_mask'], prediction.encoded_segmentation_class_mask, entropy, variance = process_logits(logits)
                prediction.metrics = {**prediction.metrics} if prediction.metrics else {} # Changes the property reference otherwise sqlalchemy doesn't send an INSERT.
                prediction.metrics['entropy'] = entropy
                prediction.metrics['variance'] = variance
                insert_statement = insert(FeatureVector).values(
                    organization_id=organization_id,
                    type='LOGITS',
                    prediction=prediction.id,
                    encoded_value=encode_np_array(logits, flatten=True),
                    model_name=p.get('model_name', '')
                )
                pg_session.execute(insert_statement.on_conflict_do_update(
                    constraint='feature_vectors_prediction_model_name_type_unique',
                    set_={
                        'id': uuid.uuid4(),
                        'encoded_value': insert_statement.excluded.encoded_value
                    }
                ))

        if 'segmentation_class_mask' in p:
            prediction.segmentation_class_mask = p['segmentation_class_mask']
            prediction.encoded_segmentation_class_mask = encode_np_array(p['segmentation_class_mask'])
            prediction.metrics = {**prediction.metrics} if prediction.metrics else {} # Changes the property reference otherwise sqlalchemy doesn't send an INSERT.
            prediction.metrics['distribution'] = segmentation_distribution(prediction.segmentation_class_mask, prediction.class_names)

        if 'confidences' in p:
            confidence_vector = p['confidences']
            if confidence_vector is None:
                prediction.metrics = None
            else:
                max_index = compute_argmax(confidence_vector)
                prediction.confidence = confidence_vector[max_index]
                if 'class_names' in p:
                    prediction.class_name = p['class_names'][max_index]

                prediction.metrics = {**prediction.metrics} if prediction.metrics else {} # Changes the property reference otherwise sqlalchemy doesn't send an INSERT.
                prediction.metrics['entropy'] = compute_entropy(confidence_vector)
                prediction.metrics['ratio_of_confidence'] = compute_ratio_of_confidence(confidence_vector)
                prediction.metrics['margin_of_confidence'] = compute_margin_of_confidence(confidence_vector)

        if 'embeddings' in p:
            embeddings = p['embeddings']

            if embeddings is None:
                pg_session.query(FeatureVector).filter(
                    FeatureVector.prediction == prediction.id,
                    FeatureVector.type == 'EMBEDDINGS',
                    FeatureVector.model_name == p.get('model_name', '')
                ).delete()
            else:
                insert_statement = insert(FeatureVector).values(
                    organization_id=organization_id,
                    type='EMBEDDINGS',
                    prediction=prediction.id,
                    encoded_value=encode_np_array(embeddings, flatten=True),
                    model_name=p.get('model_name', '')
                )
                pg_session.execute(insert_statement.on_conflict_do_update(
                    constraint='feature_vectors_prediction_model_name_type_unique',
                    set_={
                        'id': uuid.uuid4(),
                        'encoded_value': insert_statement.excluded.encoded_value
                    }
                ))
