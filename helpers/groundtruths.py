from schemas.pgsql import models

from .eventprocessor.utils import encode_np_array, decode_to_np_array
from .metrics import segmentation_distribution

GroundTruth = models.groundtruth.GroundTruth
FeatureVector = models.feature_vector.FeatureVector

def process_groundtruths(record, datapoint_id, pg_session):
    organization_id = record['organization_id']

    for g in record.get('groundtruths', []):
        if 'id' in g:
            groundtruth = pg_session.query(GroundTruth).filter(GroundTruth.id == g['id']).first()
            if not groundtruth:
                raise Exception(f"Groundtruth {g['id']} not found")
        else:
            groundtruth = GroundTruth(
                organization_id=organization_id,
                datapoint=datapoint_id,
                task_type=g['task_type']
            )
            pg_session.add(groundtruth)
            # Uncomment if groundtruth['id'] is needed to associate with other tables.
            # pg_session.flush()

        if 'task_type' in g:
            groundtruth.task_type = g['task_type']
        if 'class_name' in g:
            groundtruth.class_name = g['class_name']
        if 'class_names' in g:
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
            groundtruth.segmentation_class_mask = g['segmentation_class_mask']
            groundtruth.encoded_segmentation_class_mask = encode_np_array(g['segmentation_class_mask'])
        elif 'encoded_segmentation_class_mask' in g:
            groundtruth.encoded_segmentation_class_mask = g['encoded_segmentation_class_mask']
            groundtruth.segmentation_class_mask = decode_to_np_array(g['encoded_segmentation_class_mask']).astype('uint16').tolist()

        if groundtruth.segmentation_class_mask is not None:
            groundtruth.metrics = {**groundtruth.metrics} if groundtruth.metrics else {} # Changes the property reference otherwise sqlalchemy doesn't send an INSERT.
            groundtruth.metrics['distribution'] = segmentation_distribution(groundtruth.segmentation_class_mask, groundtruth.class_names)

    return len(record.get('groundtruths', []))