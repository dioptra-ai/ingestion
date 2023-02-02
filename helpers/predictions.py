from sqlalchemy import inspect

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

def process_predictions(record, datapoint, pg_session):
    organization_id = record['organization_id']

    for p in record.get('predictions', []):
        if 'id' in p:
            prediction = pg_session.query(Prediction).filter(Prediction.id == p['id']).first()
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
        else:
            prediction = Prediction(
                organization_id=organization_id, 
                datapoint=datapoint.id,
                task_type=p['task_type'], 
                class_name=p.get('class_name', None),
                class_names=p.get('class_names', None),
                confidence=p.get('confidence', None),
                confidences=p.get('confidences', None),
                top=p.get('top', None),
                left=p.get('left', None),
                height=p.get('height', None),
                width=p.get('width', None),
                metrics=p.get('metrics', None),
                model_name=p.get('model_name', None)
            )
            pg_session.add(prediction)
        
        if 'logits' in p:
            if len(p['logits']) == 1: # binary classifier
                positive_confidence = compute_sigmoid(p['logits']).tolist()
                prediction.confidences = [positive_confidence[0], 1 - positive_confidence[0]]
            else:
                prediction.confidences = compute_softmax(p['logits']).tolist()

            if inspect(prediction).persistent:
                pg_session.query(FeatureVector).filter(FeatureVector.prediction == prediction.id and FeatureVector.name == 'logits').delete()

            pg_session.add(FeatureVector(
                organization_id=organization_id,
                name='logits',
                prediction=prediction.id,
                value=encode_np_array(p['logits'], flatten=True),
                model_name=p['model_name']
            ))

        if 'embeddings' in p:
            if inspect(prediction).persistent:
                pg_session.query(FeatureVector).filter(FeatureVector.prediction == prediction.id and FeatureVector.name == 'embeddings').delete()

            pg_session.add(FeatureVector(
                organization_id=organization_id,
                name='embeddings',
                prediction=prediction.id,
                value=encode_np_array(p['embeddings'], flatten=True),
                model_name=p['model_name']
            ))
        
        if 'confidences' in p:
            confidence_vector = p['confidences']
            max_index = compute_argmax(confidence_vector)
            prediction.confidence = confidence_vector[max_index]
            if 'class_names' in p:
                prediction.class_name = p['class_names'][max_index]
            
            prediction.metrics = prediction.get('metrics', {})
            prediction.metrics['entropy'] = compute_entropy(confidence_vector)
            prediction.metrics['ratio_of_confidence'] = compute_ratio_of_confidence(confidence_vector)
            prediction.metrics['margin_of_confidence'] = compute_margin_of_confidence(confidence_vector)
