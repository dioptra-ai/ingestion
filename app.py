import os
import itertools
from concurrent.futures import ThreadPoolExecutor
import time
import copy
import json
import traceback
import sys

from schemas.pgsql import models, get_session
import sqlalchemy
from helpers import compatibility
from helpers.eventprocessor import event_processor
from helpers.datapoint import process_datapoint
from helpers.predictions import process_predictions
from helpers.groundtruths import process_groundtruths
from functools import partial
import orjson
from smart_open import open as smart_open
from uuid import UUID
import boto3
from botocore.client import Config

Event = models.event.Event

event_inspector = sqlalchemy.inspect(Event)
valid_event_attrs = [c_attr.key for c_attr in event_inspector.mapper.column_attrs]
ADMIN_ORG_ID = os.environ.get('ADMIN_ORG_ID', None)

def is_valid_uuidv4(uuid_to_test):

    try:
        uuid_obj = UUID(uuid_to_test, version=4)
    except ValueError:
        return False
    return str(uuid_obj) == uuid_to_test

MAX_BATCH_SIZE_BYTES = int(os.environ.get('MAX_BATCH_SIZE_BYTES', 2048000000))

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
    tic = time.time()
    events_to_update = list(filter(lambda x: 'request_id' in x and is_valid_uuidv4(x['request_id']), events))

    if len(events_to_update) > 0:
        update_events(events_to_update, organization_id)

    print(f'Updated {len(events_to_update)} events in {time.time() - tic} seconds')
    tic = time.time()

    events_to_create = list(filter(lambda x: 'request_id' not in x or not is_valid_uuidv4(x['request_id']), events))
    events_to_create = map(compatibility.process_event, events_to_create)
    events_to_create = [e for e in events_to_create if e is not None]
    events_to_create = list(map(
        partial(event_processor.process_event, organization_id=organization_id),
        events_to_create
    ))

    print(f'Created {len(events_to_create)} events in {time.time() - tic} seconds')
    tic = time.time()

    events_to_create = list(itertools.chain(*events_to_create))

    session = get_session()
    session.add_all([Event(**{
        k: v for k, v in event.items() if k in valid_event_attrs
    }) for event in events_to_create])
    tic = time.time()
    session.commit()

    print(f'Flushed {len(events_to_create)} events in {time.time() - tic} seconds')

def process_records(records, organization_id):
    logs = []
    success_datapoints = 0
    success_predictions = 0
    success_groundtruths = 0
    failed_datrapoints = 0

    for record in records:
        try:
            record = compatibility.process_record(record)
            # Allows the admin org to upload events with another org_id.
            if organization_id != ADMIN_ORG_ID or 'organization_id' not in record:
                record['organization_id'] = organization_id

            pg_session = get_session()
            try:
                datapoint_id = process_datapoint(record, pg_session)
                num_predictions = process_predictions(record, datapoint_id, pg_session)
                num_groundtruths = process_groundtruths(record, datapoint_id, pg_session)
            except:
                pg_session.rollback()
                raise
            else:
                pg_session.commit()
                success_datapoints += 1
                success_predictions += num_predictions
                success_groundtruths += num_groundtruths
        except Exception as e:
            failed_datrapoints += 1
            record_str = orjson.dumps(record, option=orjson.OPT_SERIALIZE_NUMPY).decode('utf-8')
            logs += [f'ERROR: Could not process record: {record_str[:100] + "..." if len(record_str) > 100 else record_str}']
            if type(e).__name__ == 'IntegrityError':
                logs += [e.orig.diag.message_primary]
                logs += [e.orig.diag.message_detail]
            else:
                logs += [str(e)]
            print(traceback.format_exc())
            continue

    logs += [f'Successfully processed {success_datapoints} datapoints, {success_predictions} predictions, and {success_groundtruths} groundtruths.']
    if failed_datrapoints > 0:
        logs += [f'WARNING: Failed to process {failed_datrapoints} datapoints (see logs above).']

    return logs

def dangerously_forward_to_myself(payload):

    print(f'Forwarding to myself: {payload}...')
    lambda_response = boto3.client('lambda', config=Config(
        connect_timeout=900,
        read_timeout=900,
        retries={'max_attempts': 0}
    )).invoke(
        FunctionName=os.environ['AWS_LAMBDA_FUNCTION_NAME'],
        Payload=orjson.dumps(payload)
    )

    # Also available in lambda_response: 'StatusCode', 'ExecutedVersion', 'FunctionError', 'LogResult'
    if 'FunctionError' in lambda_response:
        return [lambda_response['Payload'].read().decode('utf-8')]
    else:
        response_json = json.loads(lambda_response['Payload'].read().decode('utf-8'))

        return response_json['logs']

def process_batch(url, organization_id, offset, limit):
    line_num = offset
    batched_events = []
    current_batch_size = 0
    logs = []

    # TODO: Add params for optional S3, GCP auth: https://github.com/RaRe-Technologies/smart_open#s3-credentials
    for dioptra_record_str in itertools.islice(smart_open(url), offset, limit):
        dioptra_record = orjson.loads(dioptra_record_str)
        current_batch_size += sys.getsizeof(dioptra_record)
        if current_batch_size >= 1.1 * MAX_BATCH_SIZE_BYTES:
            raise Exception('Batch size exceeded - use the urls parameter')
        try:
            batched_events.append(dioptra_record)
            line_num += 1
        except:
            logs += [f'Could not parse JSON record in {url}[{line_num}]']

    process_events(copy.deepcopy(batched_events), organization_id)

    logs += process_records(batched_events, organization_id)

    return logs

def process_batches(urls, organization_id):
    payloads = []
    for _, url in enumerate(urls):
        current_batch_size = 0
        offset_line = 0
        current_line = 0

        # TODO: Add params for optional S3, GCP auth: https://github.com/RaRe-Technologies/smart_open#s3-credentials
        # Alternative = fetch AWS creds from the organization settings in mongodb.
        for line in smart_open(url):
            current_batch_size += sys.getsizeof(orjson.loads(line))
            current_line += 1
            if current_batch_size >= 1.1 * MAX_BATCH_SIZE_BYTES:
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
        return [l for logs in executor.map(dangerously_forward_to_myself, payloads) for l in logs]

def handler(body, _):
    try:
        if OVERRIDE_POSTGRES_ORG_ID is not None:
            print('WARNING: OVERRIDE_POSTGRES_ORG_ID is set, all events will be processed as if they were from organization ' + OVERRIDE_POSTGRES_ORG_ID)
            body['organization_id'] = OVERRIDE_POSTGRES_ORG_ID

        organization_id = body['organization_id']
        records = []
        logs = []

        if 'records' in body:
            records = body['records']

            logs += [f'Processing {len(records)} records for organization {organization_id}']
            print(f'Processing {len(records)} records for organization {organization_id}...')

            process_events(copy.deepcopy(records), organization_id)
            logs += process_records(records, organization_id)
        elif 'urls' in body:
            logs += [f"Received {len(body['urls'])} urls for organization {organization_id}"]
            print(f"Received {len(body['urls'])} urls for organization {organization_id}...")
            
            logs += process_batches(body['urls'], organization_id)
        elif 'url' in body:
            logs += [f"Processing {body['url']} for organization {organization_id}"]
            print(f"Processing {body['url']} for organization {organization_id}...")
            
            logs += process_batch(body['url'], organization_id, body.get('offset', 0), body.get('limit', None))
        else:
            raise Exception('No records or batch urls provided.')

        return {
            'statusCode': 200,
            'logs': logs
        }
    except Exception as e:
        print(traceback.format_exc())

        return {
            'statusCode': 400,
            'logs': [str(e)]
        }
