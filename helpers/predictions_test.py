import os
import json
from zipfile import ZipFile

import pytest

from mock_alchemy.mocking import UnifiedAlchemyMagicMock

from schemas.pgsql import models
Prediction = models.prediction.Prediction
FeatureVector = models.feature_vector.FeatureVector
from helpers.predictions import process_predictions

from helpers.eventprocessor.utils import (
    decode_list,
    decode_to_np_array,
    compute_shape
)

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'test_data')

def test_simple_process_predictions():
    session = UnifiedAlchemyMagicMock()

    datapoint_id = '345'
    record = {
        'organization_id': '123',
        'predictions': [{
            'task_type': 'CLASSIFICATION',
            'class_name': 'test',
            'model_name': 'test_model',
        }]
    }

    process_predictions(record, datapoint_id, session)

    results = session.query(Prediction).all()
    assert len(results) == 1

    prediction = results[0]
    assert prediction.datapoint == datapoint_id
    assert prediction.task_type == 'CLASSIFICATION'
    assert prediction.class_name == 'test'


def test_sem_seg_process_predictions():
    session = UnifiedAlchemyMagicMock()

    max_mask_size = 10
    os.environ['MAX_MASK_SIZE'] = str(max_mask_size)

    with ZipFile(os.path.join(TEST_DATA_DIR, 'semseg_pred_payload.json.zip')) as myzip:
        with myzip.open('semseg_pred_payload.json') as file:
            record = json.load(file)
    record['organization_id'] = '123'
    record['predictions'] = record['prediction']

    datapoint_id = '345'

    process_predictions(record, datapoint_id, session)

    results = session.query(Prediction).all()
    assert len(results) == 1
    prediction = results[0]
    assert prediction.datapoint == datapoint_id
    assert prediction.task_type == 'SEGMENTATION'

    assert len(compute_shape(decode_to_np_array(prediction.encoded_segmentation_class_mask))) == 2
    assert compute_shape(decode_list(prediction.encoded_resized_segmentation_class_mask)) == (max_mask_size, max_mask_size)

    assert prediction.metrics['entropy'] >= 0
    assert prediction.metrics['entropy'] <= 1
    assert len(prediction.confidences) == 19
