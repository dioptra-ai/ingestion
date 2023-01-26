import os
from concurrent.futures import ThreadPoolExecutor
import json
import datetime
import itertools
import logging
import time
import werkzeug

from lambda_multiprocessing import Pool

from flask import Flask, request, jsonify
from schemas.pgsql import models, get_session
import sqlalchemy
from helpers import compatibility
from helpers.eventprocessor import event_processor
from functools import partial
import orjson
from copy import deepcopy
from smart_open import open as smart_open

Event = models.event.Event

event_inspector = sqlalchemy.inspect(Event)
valid_event_attrs = [c_attr.key for c_attr in event_inspector.mapper.column_attrs]

app = Flask(__name__)

@app.errorhandler(Exception)
def handle_exception(e):
    logging.exception(e)

    if isinstance(e, werkzeug.exceptions.HTTPException):

        return jsonify({
            'errorType': e.name,
            'errorMessage': e.description
        }), e.code
    else:

        return jsonify({
            'errorType': 'Internal Server Error',
            'errorMessage': str(e)
        }), 500

process_pool = Pool()

def process_events(events, organization_id):
    if len(events) == 0:
        return []
    tic = time.time()
    events = process_pool.map(compatibility.process, events)
    events = [e for e in events if e is not None]
    events = process_pool.map(
        partial(event_processor.process_event, organization_id=organization_id),
        events
    )

    print(f'Processed {len(events)} events in {time.time() - tic} seconds')

    return list(itertools.chain(*events))

# 4GB k8s memory limit => up to 1GB footprint per batch
MAX_BATCH_SIZE = int(os.environ.get('MAX_BATCH_SIZE', '1073741824'))

def update_event_group(rows, update_event, session):
    new_rows, delete_rows = event_processor.resolve_update(rows, update_event)
    for row in delete_rows:
        session.delete(row)
    for row in new_rows:
        session.add(Event(**row))

def update_events(events, organization_id):
    if len(events) == 0:
        return
    session = get_session()
    request_id_map = {event['request_id']: event for event in events}

    try:
        stmt = session.query(Event)\
            .filter(
                Event.request_id.in_(list(request_id_map.keys())),
                Event.organization_id == organization_id)\
            .order_by(Event.request_id)
        data = stmt.all()
        group = []
        current_request_id = ''
        for row in data:
            if current_request_id != row.request_id:
                update_event_group(group, request_id_map.get(current_request_id, {}), session)
                current_request_id = row.request_id
                group = []
            group.append(row)
        update_event_group(group, request_id_map.get(current_request_id, {}), session)
        tic = time.time()
        session.commit()
        print(f'Updated {len(events)} events in {time.time() - tic} seconds')
    except TypeError as e:

        raise werkzeug.exceptions.BadRequest(str(e))

    except sqlalchemy.exc.ProgrammingError as e:

        raise werkzeug.exceptions.BadRequest(str(e).split(
            '\n')[0].replace('(psycopg2.errors.DatatypeMismatch)', '')
        )

def flush_events(events):
    if len(events) == 0:
        return
    session = get_session()
    try:
        # TODO: try to use on_conflict_do_update to enable upserts based on uuid.
        # https://docs.sqlalchemy.org/en/14/orm/persistence_techniques.html#orm-dml-returning-objects
        session.add_all([Event(**{
            k: v for k, v in event.items() if k in valid_event_attrs
        }) for event in events])
        tic = time.time()
        session.commit()
        print(f'Flushed {len(events)} events in {time.time() - tic} seconds')
    except TypeError as e:

        raise werkzeug.exceptions.BadRequest(str(e))

    except sqlalchemy.exc.ProgrammingError as e:

        raise werkzeug.exceptions.BadRequest(str(e).split(
            '\n')[0].replace('(psycopg2.errors.DatatypeMismatch)', '')
        )

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
                    except:
                        print(f'Could not parse JSON record in {url}[{line_num}]')
                    else:
                        if total_batch_size >= MAX_BATCH_SIZE:
                            try:
                                events_to_update = list(filter(lambda x: 'request_id' in x, batched_events))
                                events_to_create = list(filter(lambda x: 'request_id' not in x, batched_events))
                                print(f'{len(events_to_update)} records to be updated')
                                update_events(events_to_update, organization_id)
                                processed_events = process_events(events_to_create, organization_id)
                                flush_events(processed_events)
                            except Exception as e:
                                print(f'Failed to process or flush events: {e}. Moving on...')
                            finally:
                                total_batch_size = 0
                                batched_events = []
                    finally:
                        line_num += 1
                        total_batch_size += len(dioptra_record_str) * 8

                print(f'Processed {i + 1} of {len(urls)} batches')

            except Exception as e:
                print(f'Failed to process {url}: {e}, moving on...')

        if len(batched_events):
            events_to_update = list(filter(lambda x: 'request_id' in x, batched_events))
            events_to_create = list(filter(lambda x: 'request_id' not in x, batched_events))
            print(f'{len(events_to_update)} records to be updated')
            update_events(events_to_update, organization_id)
            processed_events = process_events(events_to_create, organization_id)
            flush_events(processed_events)
            batched_events = []
    except Exception as e:
        logging.exception(e)
        # TODO: Log this somewhere useful for the user to see ingestion failures.

thread_pool = ThreadPoolExecutor(max_workers=1)

@app.route('/ingest', methods = ['POST'])
def ingest():
    body = request.json
    organization_id = body['organization_id']
    records = []

    if 'records' in body:
        records = body['records']
        print(f'Received {len(records)} records for organization {organization_id}')
        events_to_update = list(filter(lambda x: 'request_id' in x, records))
        events_to_create = list(filter(lambda x: 'request_id' not in x, records))
        print(f'{len(events_to_update)} records to be updated')
        update_events(events_to_update, organization_id)
        events = process_events(events_to_create, organization_id)
        flush_events(events)
    elif 'urls' in body:
        print(f"Received {len(body['urls'])} batch urls for organization {organization_id}")
        thread_pool.submit(process_batches, body['urls'], organization_id)
    else:
        raise werkzeug.exceptions.BadRequest('No records or batch urls provided.')

    return {}, 200

def handler(event, context):

    print('event', event)

    return {
        'statusCode': 200,
        'body': json.dumps('Hello from Lambda!')
    }
