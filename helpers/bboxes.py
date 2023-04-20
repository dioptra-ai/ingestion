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
        
        # coco_polygon is a flat list of (x, y, x, y, ...) coordinates.
        if 'coco_polygon' in b:
            bbox.coco_polygon = b['coco_polygon']
            x_values = bbox.coco_polygon[::2]
            y_values = bbox.coco_polygon[1::2]
            bbox.top = b.get('top', min(y_values))
            bbox.left = b.get('left', min(x_values))
            bbox.height = b.get('height', max(y_values) - min(y_values))
            bbox.width = b.get('width', max(x_values) - min(x_values))
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
                segmentation_mask = np.array(segmentation_mask).astype(np.uint8)
                bbox.encoded_segmentation_mask = encode_np_array(b['segmentation_mask'])
                bbox.encoded_resized_segmentation_mask = encode_list(resize_mask(b['segmentation_mask']))
                # Top is the row index of the topmost nonzero element in the mask.
                # Left is the column index of the leftmost nonzero element in the mask.
                # Height is the number of nonzero rows in the mask.
                # Width is the number of nonzero columns in the mask.
                bbox.top = b.get('top', int(np.nonzero(segmentation_mask)[0].min()))
                bbox.left = b.get('left', int(np.nonzero(segmentation_mask)[1].min()))
                bbox.height = b.get('height', int(np.nonzero(segmentation_mask)[0].max() - bbox.top))
                bbox.width = b.get('width', int(np.nonzero(segmentation_mask)[1].max() - bbox.left))
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
        
        if 'objectness' in b:
            bbox.objectness = b['objectness']
