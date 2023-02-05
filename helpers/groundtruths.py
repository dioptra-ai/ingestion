from sqlalchemy import inspect

from schemas.pgsql import models

GroundTruth = models.groundtruth.GroundTruth
FeatureVector = models.feature_vector.FeatureVector

def process_groundtruths(record, datapoint_id, pg_session):
    organization_id = record['organization_id']

    for g in record.get('groundtruths', []):
        if 'id' in g:
            groundtruth = pg_session.query(GroundTruth).filter(GroundTruth.id == g['id']).first()
            if 'task_type' in g:
                groundtruth.task_type = g['task_type']
            if 'class_name' in g:
                groundtruth.class_name = g['class_name']
            if 'top' in g:
                groundtruth.top = g['top']
            if 'left' in g:
                groundtruth.left = g['left']
            if 'height' in g:
                groundtruth.height = g['height']
            if 'width' in g:
                groundtruth.width = g['width']
        else:
            groundtruth = GroundTruth(
                organization_id=organization_id, 
                datapoint=datapoint_id,
                task_type=g['task_type'], 
                class_name=g.get('class_name', None),
                top=g.get('top', None),
                left=g.get('left', None),
                height=g.get('height', None),
                width=g.get('width', None)
            )
            pg_session.add(groundtruth)
