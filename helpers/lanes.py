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
        
        if 'confidence' in l:
            lane.confidence = l['confidence']

        # TODO: turn this into a table.
        if 'classifications' in l:
            if not l['classifications']:
                lane.classifications = None
            else:
                lane.classifications = l['classifications']
                for c in lane.classifications:
                    if 'confidences' in c and 'values' in c:
                        processed_confidences = process_confidences(c['confidences'], c['values'])
                        c['confidence'] = processed_confidences['confidence']
                        c['metrics'] = {
                            **(c['metrics'] if c.get('metrics') else {}), 
                            **(processed_confidences['metrics'] if processed_confidences['metrics'] else {})
                        }
                        c['value'] = processed_confidences['class_name']
