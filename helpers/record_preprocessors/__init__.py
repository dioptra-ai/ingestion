from ..eventprocessor.utils import encode_np_array, decode_to_np_array

from .patch_sample import PatchSamplePreprocessor

from schemas.pgsql import models

FeatureVector = models.feature_vector.FeatureVector

def get_preprocessor(config):
    preprocessor = config._preprocessor

    if preprocessor:
        type = preprocessor.get('type')
        if type is None:
            raise Exception(f"Preprocessor must have a type: {preprocessor}")

        if type == 'patch_sample':
            return PatchSamplePreprocessor(**preprocessor)
        else:
            raise Exception(f"Unknown preprocessor type: {type}")
    else:
        return None

def prediction_to_record(prediction, pg_session):
    record = prediction._asdict()

    # Logits
    logits = pg_session.query(FeatureVector).filter(
        FeatureVector.prediction == prediction.id,
        FeatureVector.type == 'LOGITS',
    ).first()
    if logits:
        record['logits'] = decode_to_np_array(logits.encoded_value)
    
    # Embeddings
    embeddings = pg_session.query(FeatureVector).filter(
        FeatureVector.prediction == prediction.id,
        FeatureVector.type == 'EMBEDDINGS',
    ).first()
    if embeddings:
        record['embeddings'] = decode_to_np_array(embeddings.encoded_value)

    return record

def groundtruth_to_record(groundtruth, pg_session):
    return groundtruth._asdict()

def preprocess_predictions(predictions, pg_session):
    prediction_records = []
    for prediction in predictions:
        preprocessor = get_preprocessor(prediction)

        print(f"Preprocessor: {preprocessor}")

        if preprocessor is not None:
            prediction_record = prediction_to_record(prediction, pg_session)

            print(f"Prediction record: {prediction_record}")

            prediction_records += preprocessor.process_prediction(prediction_record, pg_session)

            print(f"Prediction records: {prediction_records}")

    return prediction_records

def preprocess_groundtruths(groundtruths, pg_session):
    groundtruth_records = []
    for groundtruth in groundtruths:
        preprocessor = get_preprocessor(groundtruth)
        if preprocessor is not None:
            groundtruth_record = groundtruth_to_record(groundtruth, pg_session)
            groundtruth_records += preprocessor.process_groundtruth(groundtruth_record, pg_session)

    return groundtruth_records
