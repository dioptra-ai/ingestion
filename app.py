import os
import itertools
import logging
import time

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
    if len(events) == 0:
        return []

    tic = time.time()
    events_to_update = list(filter(lambda x: 'request_id' in x and is_valid_uuidv4(x['request_id']), events))

    if len(events_to_update) > 0:
        update_events(events_to_update, organization_id)

    events_to_create = list(filter(lambda x: 'request_id' not in x or not is_valid_uuidv4(x['request_id']), events))
    events_to_create = map(compatibility.process, events_to_create)
    events_to_create = [e for e in events_to_create if e is not None]
    events_to_create = list(map(
        partial(event_processor.process_event, organization_id=organization_id),
        events_to_create
    ))

    print(f'Processed {len(events_to_create)} events in {time.time() - tic} seconds')

    return list(itertools.chain(*events_to_create))

def flush_events(events):
    if len(events) == 0:
        return
    session = get_session()
    session.add_all([Event(**{
        k: v for k, v in event.items() if k in valid_event_attrs
    }) for event in events])
    tic = time.time()
    session.commit()
    print(f'Flushed {len(events)} events in {time.time() - tic} seconds')

def dangerously_forward_to_myself(payload):

    print(f'Forwarding to myself: {payload}...')

    boto3.client('lambda').invoke(
        FunctionName=os.environ['AWS_LAMBDA_FUNCTION_NAME'],
        InvocationType='Event',
        Payload=orjson.dumps(payload)
    )

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

    if len(batched_events):
        events = process_events(batched_events, organization_id)
        flush_events(events)
        batched_events = []

def process_batches(urls, organization_id):
    for _, url in enumerate(urls):
        current_batch_size = 0
        offset_line = 0
        current_line = 0

        # TODO: Add params for optional S3, GCP auth: https://github.com/RaRe-Technologies/smart_open#s3-credentials
        for line in smart_open(url):
            current_batch_size += len(line)
            current_line += 1
            if current_batch_size >= MAX_BATCH_SIZE:
                dangerously_forward_to_myself({
                    'url': url,
                    'organization_id': organization_id,
                    'offset': offset_line,
                    'limit': current_line
                })
                offset_line = current_line
                current_batch_size = 0
            
        if current_batch_size > 0:
            dangerously_forward_to_myself({
                'url': url,
                'organization_id': organization_id,
                'offset': offset_line,
                'limit': current_line
            })

def handler(event, _):
    body = event
    organization_id = body['organization_id']
    records = []

    if 'records' in body:
        records = body['records']
        print(f'Received {len(records)} records for organization {organization_id}')
        events = process_events(records, organization_id)
        flush_events(events)
    elif 'urls' in body:
        print(f"Received {len(body['urls'])} batch urls for organization {organization_id}")
        process_batches(body['urls'], organization_id)
    elif 'url' in body:
        print(f"Received one batch url for organization {organization_id}")
        process_batch(body['url'], organization_id, body.get('offset', 0), body.get('limit', float('inf')))
    else:
        raise Exception('No records or batch urls provided.')

    return {
        'statusCode': 200
    }
