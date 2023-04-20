from schemas.pgsql import models

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

        if 'class_name' in l:
            lane.class_name = l['class_name']

        if 'confidence' in l:
            lane.confidence = l['confidence']
