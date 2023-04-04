import numpy as np

from schemas.pgsql import models

from helpers.eventprocessor.utils import (
    encode_np_array,
    resize_mask,
    encode_list,
    compute_shape,
    squeeze
)
from helpers.metrics import segmentation_distribution

GroundTruth = models.groundtruth.GroundTruth
FeatureVector = models.feature_vector.FeatureVector

def process_groundtruth_records(records, datapoint, pg_session):
    groundtruths = []

    for g in records:
        if 'id' in g:
            groundtruth = pg_session.query(GroundTruth).filter(GroundTruth.id == g['id']).first()
            if not groundtruth:
                raise Exception(f"Groundtruth {g['id']} not found")
        else:
            groundtruth = GroundTruth(
                organization_id=datapoint.organization_id,
                datapoint=datapoint.id,
                task_type=g['task_type']
            )
            pg_session.add(groundtruth)
            # Uncomment if groundtruth['id'] is needed to associate with other tables.
            # pg_session.flush()
        
        groundtruths.append(groundtruth)

        if '_preprocessor' in g:
            groundtruth._preprocessor = g['_preprocessor']

        if 'task_type' in g:
            groundtruth.task_type = g['task_type']

        if groundtruth.id is not None: # in mock sqlalchemy the id is None
            # Overriding groundtruth with the same datapoint id and the same task type.
            if groundtruth.task_type == 'CLASSIFICATION' or groundtruth.task_type == 'SEGMENTATION':
                pg_session.query(GroundTruth).filter(
                    GroundTruth.datapoint == datapoint.id,
                    GroundTruth.task_type == groundtruth.task_type,
                    GroundTruth.id != groundtruth.id
                ).delete()

        if 'class_name' in g:
            groundtruth.class_name = g['class_name']
        if 'class_names' in g:
            if not isinstance(g['class_names'], list) and not g['class_names'] is None:
                raise Exception(f"class_names must be a list or null. Got {type(g['class_names'])}")
            groundtruth.class_names = g['class_names']
        if 'top' in g:
            groundtruth.top = g['top']
        if 'left' in g:
            groundtruth.left = g['left']
        if 'height' in g:
            groundtruth.height = g['height']
        if 'width' in g:
            groundtruth.width = g['width']

        if 'segmentation_class_mask' in g:
            segmentation_class_mask = g['segmentation_class_mask']
            if segmentation_class_mask and np.array(segmentation_class_mask).size > 0:
                if len(compute_shape(g['segmentation_class_mask'])) == 3:
                    g['segmentation_class_mask'] = squeeze(g['segmentation_class_mask'])
                groundtruth.encoded_segmentation_class_mask = encode_np_array(g['segmentation_class_mask'])
                groundtruth.encoded_resized_segmentation_class_mask = encode_list(resize_mask(g['segmentation_class_mask']))
                groundtruth.metrics = {**groundtruth.metrics} if groundtruth.metrics else {} # Changes the property reference otherwise sqlalchemy doesn't send an INSERT.
                groundtruth.metrics['distribution'] = segmentation_distribution(g['segmentation_class_mask'], groundtruth.class_names)

    return groundtruths
