import numpy as np

from schemas.pgsql import models
from helpers.eventprocessor.utils import (
    encode_np_array,
    encode_list,
    resize_mask
)
from helpers.common import process_confidences

BBoxes = models.bbox.BBox

def process_bbox_records(bboxes, pg_session, prediction=None, groundtruth=None):
    for b in bboxes:
        if 'id' in b:
            bbox = pg_session.query(BBoxes).filter(BBoxes.id == b['id']).first()
        else:
            bbox = BBoxes(
                organization_id=prediction.organization_id if prediction else groundtruth.organization_id,
                prediction=prediction.id if prediction else None,
                groundtruth=groundtruth.id if groundtruth else None,
            )
            pg_session.add(bbox)
        
        if 'top' in b:
            bbox.top = b['top']
        if 'left' in b:
            bbox.left = b['left']
        if 'height' in b:
            bbox.height = b['height']
        if 'width' in b:
            bbox.width = b['width']

        if 'segmentation_mask' in b:
            segmentation_mask = b['segmentation_mask']
            if segmentation_mask and np.array(segmentation_mask).size > 0:
                bbox.encoded_segmentation_mask = encode_np_array(b['segmentation_mask'])
                bbox.encoded_resized_segmentation_mask = encode_list(resize_mask(b['segmentation_mask']))
                bbox.metrics = {**bbox.metrics} if bbox.metrics else {}
            else:
                bbox.encoded_segmentation_mask = None
                bbox.encoded_resized_segmentation_mask = None
                bbox.metrics = None

        if 'class_names' in b:
            if not isinstance(b['class_names'], list) and not b['class_names'] is None:
                raise Exception(f"class_names must be a list or null. Got {type(b['class_names'])}")
            bbox.class_names = b['class_names']

        if 'confidences' in b:
            bbox.confidences = b['confidences']
            processed_confidences = process_confidences(b['confidences'], bbox.class_names)
            bbox.confidence = processed_confidences['confidence']
            bbox.metrics = {
                **(bbox.metrics if bbox.metrics else {}), 
                **(processed_confidences['metrics'] if processed_confidences['metrics'] else {})
            }
            bbox.class_name = processed_confidences['class_name']

        if 'class_name' in b:
            bbox.class_name = b['class_name']

        if 'confidence' in b:
            bbox.confidence = b['confidence']