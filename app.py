import os
import itertools
import logging
from concurrent.futures import ThreadPoolExecutor
import time
import json

from schemas.pgsql import models, get_session
import sqlalchemy
from helpers import compatibility
from helpers.eventprocessor import event_processor
from functools import partial
import orjson
from smart_open import open as smart_open
from uuid import UUID
import boto3

Event = models.event.Event

event_inspector = sqlalchemy.inspect(Event)
valid_event_attrs = [c_attr.key for c_attr in event_inspector.mapper.column_attrs]

def is_valid_uuidv4(uuid_to_test):

    try:
        uuid_obj = UUID(uuid_to_test, version=4)
    except ValueError:
        return False
    return str(uuid_obj) == uuid_to_test

MAX_BATCH_SIZE = int(os.environ.get('MAX_BATCH_SIZE', '134217728'))
OVERRIDE_POSTGRES_ORG_ID = os.environ.get('OVERRIDE_POSTGRES_ORG_ID', None)

def update_events(events, organization_id):

    def update_event_group(rows, update_event, session):
        new_rows, delete_rows = event_processor.resolve_update(rows, update_event)
        for row in delete_rows:
            session.delete(row)
        for row in new_rows:
            session.add(Event(**row))

    print(f'Updating {len(events)} events...')

    tic = time.time()
    session = get_session()
    request_id_map = {event['request_id']: event for event in events}

    data = session.query(Event).filter(
            Event.request_id.in_(list(request_id_map.keys())),
            Event.organization_id == organization_id)\
        .order_by(Event.request_id).all()
    group = []
    current_request_id = ''
    for row in data:
        if current_request_id != row.request_id:
            update_event_group(group, request_id_map.get(current_request_id, {}), session)
            current_request_id = row.request_id
            group = []
        group.append(row)
    update_event_group(group, request_id_map.get(current_request_id, {}), session)
    session.commit()
    print(f'Updated {len(events)} events in {time.time() - tic} seconds')

def process_events(events, organization_id):
    logs = []
    if len(events) == 0:
        return logs

    tic = time.time()
    events_to_update = list(filter(lambda x: 'request_id' in x and is_valid_uuidv4(x['request_id']), events))

    if len(events_to_update) > 0:
        update_events(events_to_update, organization_id)

    logs.append(f'Updated {len(events_to_update)} events in {time.time() - tic} seconds')
    tic = time.time()

    events_to_create = list(filter(lambda x: 'request_id' not in x or not is_valid_uuidv4(x['request_id']), events))
    events_to_create = map(compatibility.process, events_to_create)
    events_to_create = [e for e in events_to_create if e is not None]
    events_to_create = list(map(
        partial(event_processor.process_event, organization_id=organization_id),
        events_to_create
    ))

    logs.append(f'Created {len(events_to_create)} events in {time.time() - tic} seconds')
    tic = time.time()

    events = list(itertools.chain(*events_to_create))

    session = get_session()
    session.add_all([Event(**{
        k: v for k, v in event.items() if k in valid_event_attrs
    }) for event in events])
    tic = time.time()
    session.commit()

    logs.append(f'Flushed {len(events)} events in {time.time() - tic} seconds')

    return logs

def dangerously_forward_to_myself(payload):

    print(f'Forwarding to myself: {payload}...')

    lambda_response = boto3.client('lambda').invoke(
        FunctionName=os.environ['AWS_LAMBDA_FUNCTION_NAME'],
        Payload=orjson.dumps(payload)
    )

    # Also available in lambda_response: 'StatusCode', 'ExecutedVersion', 'FunctionError', 'LogResult'
    response_json = json.loads(lambda_response['Payload'].read().decode('utf-8'))

    return response_json['logs']

def process_batch(url, organization_id, offset, limit):
    line_num = offset
    batched_events = []
    current_batch_size = 0

    # TODO: Add params for optional S3, GCP auth: https://github.com/RaRe-Technologies/smart_open#s3-credentials
    for dioptra_record_str in itertools.islice(smart_open(url), offset, limit):

        current_batch_size += len(dioptra_record_str)
        if current_batch_size >= 1.1 * MAX_BATCH_SIZE:
            raise Exception('Batch size exceeded - use the urls parameter')

        try:
            batched_events.append(orjson.loads(dioptra_record_str))
            line_num += 1
        except:
            print(f'Could not parse JSON record in {url}[{line_num}]')

    return process_events(batched_events, organization_id)
    

def process_batches(urls, organization_id):
    payloads = []
    for _, url in enumerate(urls):
        current_batch_size = 0
        offset_line = 0
        current_line = 0

        # TODO: Add params for optional S3, GCP auth: https://github.com/RaRe-Technologies/smart_open#s3-credentials
        for line in smart_open(url):
            current_batch_size += len(line)
            current_line += 1
            if current_batch_size >= MAX_BATCH_SIZE:
                payloads.append({
                    'url': url,
                    'organization_id': organization_id,
                    'offset': offset_line,
                    'limit': current_line
                })
                offset_line = current_line
                current_batch_size = 0
            
        if current_batch_size > 0:
            payloads.append({
                'url': url,
                'organization_id': organization_id,
                'offset': offset_line,
                'limit': current_line
            })

    with ThreadPoolExecutor() as executor:
        return list(executor.map(dangerously_forward_to_myself, payloads))

def handler(event, _):
    body = event

    if OVERRIDE_POSTGRES_ORG_ID is not None:
        print('WARNING: OVERRIDE_POSTGRES_ORG_ID is set, all events will be processed as if they were from organization ' + OVERRIDE_POSTGRES_ORG_ID)
        body['organization_id'] = OVERRIDE_POSTGRES_ORG_ID

    organization_id = body['organization_id']
    records = []
    logs = []

    if 'records' in body:
        records = body['records']
        print(f'Received {len(records)} records for organization {organization_id}')
        
        logs = process_events(records, organization_id)
    elif 'urls' in body:
        print(f"Received {len(body['urls'])} batch urls for organization {organization_id}")
        
        logs = process_batches(body['urls'], organization_id)
    elif 'url' in body:
        print(f"Received one batch url for organization {organization_id}")
        
        logs = process_batch(body['url'], organization_id, body.get('offset', 0), body.get('limit', float('inf')))
    else:
        raise Exception('No records or batch urls provided.')

    return {
        'statusCode': 200,
        'logs': logs
    }
