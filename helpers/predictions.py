import uuid
from sqlalchemy.dialects.postgresql import insert

from schemas.pgsql import models

from .eventprocessor.utils import (
    encode_np_array,
    compute_softmax,
    compute_softmax2D,
    compute_sigmoid,
    compute_argmax,
    compute_entropy,
    compute_mean,
    compute_shape,
    compute_sum,
    compute_variance,
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
            if not prediction:
                raise Exception(f"Prediction {p['id']} not found")
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
            logits = p['logits']

            if logits is None:
                pg_session.query(FeatureVector).filter(
                    FeatureVector.prediction == prediction.id, 
                    FeatureVector.type == 'LOGITS',
                    FeatureVector.model_name == p.get('model_name', '')
                ).delete()
            else:
                if len(logits) == 1: # binary classifier
                    positive_confidence = compute_sigmoid(logits).tolist()
                    prediction.confidences = [positive_confidence[0], 1 - positive_confidence[0]]
                elif isinstance(logits[0], float): # multiple class classifier
                    prediction.confidences = compute_softmax(logits).tolist()
                elif len(compute_shape(logits)) == 3: #semantic segmentation
                    # dimension 0 is number of classes
                    # dimension 1 is height
                    # dimension 2 is width
                    probability_masks = [compute_softmax2D(logits[i]) for i in range(0, len(logits))]
                    probability_mean = compute_mean(probability_masks, axis=0)
                    prediction.segmentation_class_mask = compute_argmax(logits, axis=0)
                    # compute entropy
                    prediction.metrics = {}
                    prediction.metrics['entropy'] = compute_entropy(probability_mean) 
                    prediction.confidences = [0 for _ in range(0, len(logits))]
                    # for each class in the segmentation_class_mask, compute the confidence of the class based on the pixels in the mask
                    # assign the confidence of all classes not in that mask to 0
                    for i in range(0, len(logits)):
                        if i in prediction.segmentation_class_mask:
                            # find the pixels in the mask that equal i
                            # compute the mean of the probabilities of those pixels
                            # assign the confidence of that class to that mean
                            prediction.confidences[i] = compute_mean([probability_masks[i][j][k] for j in range(0, len(logits[0])) for k in range(0, len(logits[0][0])) if prediction.segmentation_class_mask[j][k] == i])
                else: # semantic segmentation with dropout
                    # dimension 0 is number of inferences
                    # dimension 1 is number of classes
                    # probability_masks is a list of probability masks for each inference
                    probability_masks = []
                    for i in range(0, len(logits)):
                        probability_i = [compute_softmax2D(logits[i][j]) for j in range(0, len(logits[0]))]
                        probability_masks.append(probability_i)
                    # mean_probs is the mean probabilities for each class for each inference over the image
                    mean_probs = compute_mean(probability_masks, axis = (3,4))                        
                    # variances is the variance of the probabilities for each class for each inference over the image
                    prediction.metrics['variances'] = compute_variance(mean_probs, axis = 0).tolist()
                    # probabilities is the average probability for each class over all inferences
                    # it is now 3 dimensional [num_classes, height, width]
                    probabilities = compute_mean(probability_masks, axis = 0)
                    prediction.segmentation_class_mask = compute_argmax(probabilities)
                    prediction.metrics['entropy'] = compute_entropy(compute_mean(probabilities, axis = 0))
                    prediction.confidences = [0 for _ in range(0, len(logits[0]))]
                    for i in range(0, len(logits[0])):
                        # find the pixels in the mask that equal i
                        # compute the mean of the probabilities of those pixels
                        # assign the confidence of that class to that mean
                        prediction.confidences[i] = compute_mean([probabilities[i][j][k] for j in range(0, len(logits[0][0])) for k in range(0, len(logits[0][0][0])) if prediction.segmentation_class_mask[j][k] == i])

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
