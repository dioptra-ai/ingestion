from sqlalchemy import inspect

from schemas.pgsql import models

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
        if 'segmentation_class_mask' in g:
            groundtruth.segmentation_class_mask = g['segmentation_class_mask']
        if 'top' in g:
            groundtruth.top = g['top']
        if 'left' in g:
            groundtruth.left = g['left']
        if 'height' in g:
            groundtruth.height = g['height']
        if 'width' in g:
            groundtruth.width = g['width']
