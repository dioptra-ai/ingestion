import os
import itertools
from concurrent.futures import ThreadPoolExecutor
from lambda_multiprocessing import Pool
import multiprocessing
import time
import copy
import json
import traceback
import os, psutil
import gc

from schemas.pgsql import models, get_session
import sqlalchemy
from helpers import compatibility
from helpers.eventprocessor import event_processor
from helpers.record_preprocessors import preprocess_datapoint
from helpers.datapoint import process_datapoint_record
from helpers.predictions import process_prediction_records
from helpers.groundtruths import process_groundtruth_records
from functools import partial
import orjson
from smart_open import open as smart_open
from uuid import UUID
import boto3
from botocore.client import Config
from dioptra.lake.utils import ingestion

num_processes = multiprocessing.cpu_count()

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

def process_records(records, organization_id, parent_pg_session=None):
    logs = []
    success_datapoints = 0
    success_predictions = 0
    success_groundtruths = 0
    failed_datapoints = 0

    records = ingestion.process_records(records)

    for record in records:
        try:
            record = compatibility.process_record(record)
            # Allows the admin org to upload events with another org_id.
            if organization_id != ADMIN_ORG_ID or 'organization_id' not in record:
                record['organization_id'] = organization_id

            if parent_pg_session is not None:
                pg_session = parent_pg_session
                pg_session.begin_nested()
            else:
                pg_session = get_session()

            try:
                organization_id = record['organization_id']
                datapoint = process_datapoint_record(record, pg_session)
                predictions = process_prediction_records(record.get('predictions', []), datapoint, pg_session)
                groundtruths = process_groundtruth_records(record.get('groundtruths', []), datapoint, pg_session)

                logs += preprocess_datapoint(datapoint, pg_session)
            except:
                pg_session.rollback()
                raise
            else:
                pg_session.commit()
                success_datapoints += 1
                success_predictions += len(predictions)
                success_groundtruths += len(groundtruths)
        except KeyError as k:
            raise Exception(f'Property not found: {k}')
        except Exception as e:
            failed_datapoints += 1
            record_str = orjson.dumps(record, option=orjson.OPT_SERIALIZE_NUMPY).decode('utf-8')
            logs += [f'ERROR: Could not process record: {record_str[:256] + "..." if len(record_str) > 256 else record_str}']
            if type(e).__name__ == 'IntegrityError':
                logs += [e.orig.diag.message_primary]
                logs += [e.orig.diag.message_detail]
            else:
                logs += [str(e)]
            print(traceback.format_exc())
            continue

    logs += [f'Processed {success_datapoints} datapoints, {success_predictions} predictions, and {success_groundtruths} groundtruths.']
    print(f'Processed {success_datapoints} datapoints, {success_predictions} predictions, and {success_groundtruths} groundtruths.')

    if failed_datapoints > 0:
        logs += [f'WARNING: Failed to process {failed_datapoints} datapoints (see logs above).']
        print(f'WARNING: Failed to process {failed_datapoints} datapoints (see logs above).')

    return logs

def process_batch(url, organization_id, offset, limit):
    line_num = offset
    batched_events = []
    logs = []
    def process_and_flush_batch():
        nonlocal batched_events
        nonlocal logs
        if len(batched_events) > 0:
            print(f'Processing {len(batched_events)} records...')
            process_events(copy.deepcopy(batched_events), organization_id)
            logs += process_records(batched_events, organization_id)
            batched_events.clear()

    for dioptra_record_str in itertools.islice(smart_open(url), offset, limit):
        dioptra_record = orjson.loads(dioptra_record_str)
        memory_usage_pct = psutil.virtual_memory().percent

        print(f'Memory usage: {memory_usage_pct}%')

        # Fix this when we remove events deepcopy.
        if memory_usage_pct >= 0.9 * 50:
            process_and_flush_batch()
        try:
            batched_events.append(dioptra_record)
            line_num += 1
        except:
            logs += [f'Could not parse JSON record at {url}:{line_num}']
            print(f'Could not parse JSON record at {url}:{line_num}')

    process_and_flush_batch()

    return logs

# Process slices of the batch with multiprocessing.
def parallelize_batch(url, organization_id, offset, limit):
    tic = time.time()

    if limit is None:
        print('WARNING: unable parallelize batch without a "limit" parameter. Using a single vCPU.')
        logs = process_batch(url, organization_id, offset, limit)
    else:
        slice_size = (limit - offset) // num_processes
        slice_offsets = [offset + i * slice_size for i in range(num_processes)]
        slice_limits = [offset + (i + 1) * slice_size for i in range(num_processes)]
        slice_limits[-1] = limit

        print(f'Parallelizing batch over {num_processes} processes...')
        with Pool() as pool:
            logs = pool.starmap(process_batch, [(url, organization_id, offset, limit) for offset, limit in zip(slice_offsets, slice_limits)])
            logs = [log for sublist in logs for log in sublist]

    print(f'Processed batch in {time.time() - tic:.2f}s')
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
    response_body = lambda_response['Payload'].read().decode('utf-8')
    print(f'Forwarded batch response: {response_body}')
    if 'FunctionError' in lambda_response:
        error_body = f'{lambda_response["FunctionError"]}: {response_body}'
        print(f'ERROR: forwarded batch failed: {error_body}')
        logs = [error_body]
    else:
        response_json = json.loads(response_body)
        logs = response_json['logs']

    return {
        'statusCode': lambda_response['StatusCode'],
        'logs': logs
    }

def forward_batches(urls, organization_id):
    futures = []
    records = [] # Just here to hold in memory for estimation.
    with ThreadPoolExecutor() as executor:
        for _, url in enumerate(urls):
            offset_line = 0
            current_line = 0

            for line in smart_open(url):
                record = orjson.loads(line)
                current_line += 1
                memory_usage_pct = psutil.virtual_memory().percent

                print(f'Estimated memory usage: {memory_usage_pct}%')

                # TODO: Set to 0.9 * 100 when we remove events processing
                if memory_usage_pct >= 0.9 * 50:
                    previous_line = current_line - 1

                    if previous_line == offset_line:
                        raise Exception(f'Record {current_line} in {url} is too large to process')

                    print(f'Forwarding batch {offset_line}:{current_line}...')

                    futures.append(executor.submit(dangerously_forward_to_myself, {
                        'url': url,
                        'organization_id': organization_id,
                        'offset': offset_line,
                        'limit': previous_line
                    }))
                    records = []
                    offset_line = previous_line

                records.append(record)

            if offset_line < current_line:
                print(f'Forwarding batch {offset_line}:{current_line}...')
                futures.append(executor.submit(dangerously_forward_to_myself, {
                    'url': url,
                    'organization_id': organization_id,
                    'offset': offset_line,
                    'limit': current_line
                }))
                records = []

        return [future.result() for future in futures]

def handler(body, _):
    try:
        memory_available_before = psutil.virtual_memory().available
        gc.collect()
        memory_available_after = psutil.virtual_memory().available
        print(f'Garbage collection freed {memory_available_after - memory_available_before} bytes of memory. Memory usage: {psutil.virtual_memory().percent}%')

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

            results = forward_batches(body['urls'], organization_id)
            all_logs = [l for logs in map(lambda result: result['logs'], results) for l in logs]

            if any([result['statusCode'] == 200 for result in results]):
                logs += all_logs
            else:
                return {
                    'statusCode': 400,
                    'logs': all_logs
                }
        elif 'url' in body:
            logs += [f"Processing {body['url']} for organization {organization_id}"]
            print(f"Processing {body['url']} for organization {organization_id}...")

            logs += parallelize_batch(body['url'], organization_id, body.get('offset', 0), body.get('limit', None))
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
            'logs': [f'{type(e).__name__}: {e}']
        }