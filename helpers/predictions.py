import copy
from sqlalchemy import inspect, insert

from schemas.pgsql import models

from .eventprocessor.utils import (
    encode_np_array,
    compute_softmax,
    compute_sigmoid,
    compute_argmax,
    compute_entropy,
    compute_margin_of_confidence,
    compute_ratio_of_confidence
)

Prediction = models.prediction.Prediction
FeatureVector = models.feature_vector.FeatureVector

def process_predictions(record, datapoint_id, pg_session):
    organization_id = record['organization_id']

    for p in record.get('predictions', []):
        if 'id' in p:
            prediction = pg_session.query(Prediction).filter(Prediction.id == p['id']).first()
        else:
            prediction = Prediction(
                organization_id=organization_id, 
                datapoint=datapoint_id,
                task_type=p['task_type']
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
            if inspect(prediction).persistent:
                pg_session.query(FeatureVector).filter(FeatureVector.prediction == prediction.id, FeatureVector.type == 'LOGITS').delete()

            logits = p['logits']
            if logits is not None:
                if len(logits) == 1: # binary classifier
                    positive_confidence = compute_sigmoid(logits).tolist()
                    prediction.confidences = [positive_confidence[0], 1 - positive_confidence[0]]
                else:
                    prediction.confidences = compute_softmax(logits).tolist()

                pg_session.add(FeatureVector(
                    organization_id=organization_id,
                    type='LOGITS',
                    prediction=prediction.id,
                    value=encode_np_array(logits, flatten=True),
                    model_name=p.get('model_name', None)
                ))

        if 'confidences' in p:
            confidence_vector = p['confidences']
            if confidence_vector is None:
                prediction.metrics = None
            else:
                max_index = compute_argmax(confidence_vector)
                prediction.confidence = confidence_vector[max_index]
                if 'class_names' in p:
                    prediction.class_name = p['class_names'][max_index]
                
                prediction.metrics = copy.deepcopy(prediction.metrics) if prediction.metrics else {} # Changes the property reference otherwise sqlalchemy doesn't send an INSERT.
                prediction.metrics['entropy'] = compute_entropy(confidence_vector)
                prediction.metrics['ratio_of_confidence'] = compute_ratio_of_confidence(confidence_vector)
                prediction.metrics['margin_of_confidence'] = compute_margin_of_confidence(confidence_vector)

        if 'embeddings' in p:                
            if inspect(prediction).persistent:
                pg_session.query(FeatureVector).filter(FeatureVector.prediction == prediction.id, FeatureVector.type == 'EMBEDDINGS').delete()

            embeddings = p['embeddings']
            if embeddings is not None:
                pg_session.add(FeatureVector(
                    organization_id=organization_id,
                    type='EMBEDDINGS',
                    prediction=prediction.id,
                    value=encode_np_array(embeddings, flatten=True),
                    model_name=p.get('model_name', None)
                ))