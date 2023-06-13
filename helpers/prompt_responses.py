from schemas.pgsql import models

from helpers.common import process_confidences

Prompt_Responses = models.prompt_response.Prompt_Response

def process_prompt_response_records(prompt_responses, pg_session, prediction=None, groundtruth=None):
    for p in prompt_responses:
        if 'id' in p:
            prompt_response = pg_session.query(Prompt_Responses).filter(Prompt_Responses.id == p['id']).first()
        else:
            prompt_response = Prompt_Responses(
                organization_id=prediction.organization_id if prediction else groundtruth.organization_id,
                prediction=prediction.id if prediction else None,
                groundtruth=groundtruth.id if groundtruth else None,
            )
            pg_session.add(prompt_response)
        
        if 'confidence' in p:
            prompt_response.confidence = p['confidence']

        if 'prompt' in p:
            prompt_response.prompt = p['prompt']
        
        if 'context' in p:
            prompt_response.context = p['context']
            
        if 'response' in p:
            prompt_response.response = p['response']

        if 'metrics' in p:
            prompt_response.metrics = {
                **(prompt_response.metrics if prompt_response.metrics else {}),
                **(p['metrics'] if p.get('metrics') else {})
            }
