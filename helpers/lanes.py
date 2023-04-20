from schemas.pgsql import models

from helpers.common import process_confidences

Lanes = models.lane.Lane

def process_lane_records(lanes, pg_session, prediction=None, groundtruth=None):
    for l in lanes:
        if 'id' in l:
            lane = pg_session.query(Lanes).filter(Lanes.id == l['id']).first()
        else:
            lane = Lanes(
                organization_id=prediction.organization_id if prediction else groundtruth.organization_id,
                prediction=prediction.id if prediction else None,
                groundtruth=groundtruth.id if groundtruth else None,
            )
            pg_session.add(lane)
        
        # coco_polyline is a flat list of (x, y, x, y, ...) coordinates.
        if 'coco_polyline' in l:
            lane.coco_polyline = l['coco_polyline']

        if 'class_names' in l:
            if not isinstance(l['class_names'], list) and not l['class_names'] is None:
                raise Exception(f"class_names must be a list or null. Got {type(l['class_names'])}")
            lane.class_names = l['class_names']

        if 'confidences' in l:
            lane.confidences = l['confidences']
            processed_confidences = process_confidences(l['confidences'], lane.class_names)
            lane.confidence = processed_confidences['confidence']
            lane.metrics = {
                **(lane.metrics if lane.metrics else {}), 
                **(processed_confidences['metrics'] if processed_confidences['metrics'] else {})
            }
            lane.class_name = processed_confidences['class_name']

        if 'class_name' in l:
            lane.class_name = l['class_name']

        if 'confidence' in l:
            lane.confidence = l['confidence']
