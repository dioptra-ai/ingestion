from schemas.pgsql import models

Completions = models.completion.Completion

def process_completion_records(completions, pg_session, prediction=None, groundtruth=None):
    for c in completions:
        if 'id' in c:
            completion = pg_session.query(Completions).filter(Completions.id == p['id']).first()
        else:
            completion = Completions(
                organization_id=prediction.organization_id if prediction else groundtruth.organization_id,
                prediction=prediction.id if prediction else None,
                groundtruth=groundtruth.id if groundtruth else None,
            )
            pg_session.add(completion)
        
        if 'confidence' in c:
            completion.confidence = c['confidence']
        
        if 'text' in c:
            completion.text = c['text']

        if 'metrics' in c:
            completion.metrics = {
                **(completion.metrics if completion.metrics else {}),
                **(c['metrics'] if c.get('metrics') else {})
            }
