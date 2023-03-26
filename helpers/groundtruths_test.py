import os
import json
from zipfile import ZipFile

import pytest

from mock_alchemy.mocking import UnifiedAlchemyMagicMock

from schemas.pgsql import models
GroundTruth = models.groundtruth.GroundTruth
from helpers.groundtruths import process_groundtruths

from helpers.eventprocessor.utils import (
    decode_list,
    decode_to_np_array,
    compute_shape
)

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'test_data')

def test_simple_process_groundtruths():
    session = UnifiedAlchemyMagicMock()

    datapoint_id = '345'
    record = {
        'organization_id': '123',
        'groundtruths': [{
            'task_type': 'CLASSIFICATION',
            'class_name': 'test'
        }]
    }

    process_groundtruths(record, datapoint_id, session)

    results = session.query(GroundTruth).all()
    assert len(results) == 1

    groundtruth = results[0]
    assert groundtruth.datapoint == datapoint_id
    assert groundtruth.task_type == 'CLASSIFICATION'
    assert groundtruth.class_name == 'test'


def test_sem_seg_process_groundtruths():
    session = UnifiedAlchemyMagicMock()

    max_mask_size = 10
    os.environ['MAX_MASK_SIZE'] = str(max_mask_size)

    with ZipFile(os.path.join(TEST_DATA_DIR, 'semseg_gt_payload.json.zip')) as myzip:
        with myzip.open('semseg_gt_payload.json') as file:
            record = json.load(file)
    record['organization_id'] = '123'
    record['groundtruths'] = [record['groundtruth']]

    datapoint_id = '345'

    process_groundtruths(record, datapoint_id, session)

    results = session.query(GroundTruth).all()
    assert len(results) == 1

    groundtruth = results[0]
    assert groundtruth.datapoint == datapoint_id
    assert groundtruth.task_type == 'SEGMENTATION'
    assert len(compute_shape(decode_to_np_array(groundtruth.encoded_segmentation_class_mask))) == 2
    assert compute_shape(decode_list(groundtruth.encoded_resized_segmentation_class_mask)) == (max_mask_size, max_mask_size)
