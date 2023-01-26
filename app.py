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

Event = models.event.Event

event_inspector = sqlalchemy.inspect(Event)
valid_event_attrs = [c_attr.key for c_attr in event_inspector.mapper.column_attrs]

def is_valid_uuidv4(uuid_to_test):

    try:
        uuid_obj = UUID(uuid_to_test, version=4)
    except ValueError:
        return False
    return str(uuid_obj) == uuid_to_test

def process_events(events, organization_id):
    if len(events) == 0:
        return []
    tic = time.time()
    events_to_update = list(filter(lambda x: 'request_id' in x and is_valid_uuidv4(x['request_id']), events))
    print(f'{len(events_to_update)} records to be updated')
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

# 4GB k8s memory limit => up to 1GB footprint per batch
MAX_BATCH_SIZE = int(os.environ.get('MAX_BATCH_SIZE', '1073741824'))

def update_events(events, organization_id):
    if len(events) == 0:
        return

    def update_event_group(rows, update_event, session):
        new_rows, delete_rows = event_processor.resolve_update(rows, update_event)
        for row in delete_rows:
            session.delete(row)
        for row in new_rows:
            session.add(Event(**row))

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
    update_event_group(group, request_id_map[current_request_id], session)
    session.commit()
    print(f'Updated {len(events)} events in {time.time() - tic} seconds')

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

def process_batches(urls, organization_id):
    batched_events = []
    # TODO: Add params for optional S3, GCP auth: https://github.com/RaRe-Technologies/smart_open#s3-credentials

    try:
        for i, url in enumerate(urls):
            try:
                line_num = 0
                total_batch_size = 0
                for dioptra_record_str in smart_open(url):
                    try:
                        batched_events.append(orjson.loads(dioptra_record_str))
                        total_batch_size += len(dioptra_record_str)
                        line_num += 1
                    except:
                        print(f'Could not parse JSON record in {url}[{line_num}]')
                    else:
                        if total_batch_size >= MAX_BATCH_SIZE:
                            try:
                                processed_events = process_events(batched_events, organization_id)
                                flush_events(processed_events)
                            except Exception as e:
                                print(f'Failed to process or flush events: {e}. Moving on...')
                            finally:
                                total_batch_size = 0
                                batched_events = []

            except Exception as e:
                print(f'Failed to process {url}: {e}, moving on...')

        if len(batched_events):
            processed_events = process_events(batched_events, organization_id)
            flush_events(processed_events)
            batched_events = []

    except Exception as e:
        logging.exception(e)
        # TODO: Log this somewhere useful for the user to see ingestion failures.

def handler(event, context):
    body = orjson.loads(event['body'])
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
    else:
        raise Exception('No records or batch urls provided.')

    return {
        'statusCode': 200
    }
