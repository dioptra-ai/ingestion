import os
from concurrent.futures import ThreadPoolExecutor
import datetime
import itertools
import logging
import time
import werkzeug
from multiprocessing import Pool
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

def resolve_update(rows, update_event, session):
    updated_rows = []
    new_rows = []
    need_to_update_annotation = 'groundtruth' in update_event
    datapoint_row = None
    for row in rows:
        if 'tags' in update_event:
            row['tags'] = update_event['tags']
        # we only support gt update for classifier for now
        if row['model_type'] == 'CLASSIFIER' and 'groundtruth' in update_event:
            if 'prediction' in row or 'groundtruth' in row:
                row['groundtruth'] = update_event['groundtruth']
                need_to_update_annotation = False
            else:
                datapoint_row = row

        updated_rows.append(row)

    if need_to_update_annotation:
        new_row = deepcopy(datapoint_row)
        new_row['groundtruth'] = update_event['groundtruth']
        new_row.pop('uuid')
        new_rows.append(new_row)

    print('updated_rows', flush=True)
    print(updated_rows, flush=True)
    print('new_rows', flush=True)
    print(new_rows, flush=True)

    session.bulk_update_mappings(updated_rows)
    session.bulk_insert_mappings(new_rows)

def update_events(events, organization_id):
    session = get_session()
    request_id_map = {event['request_id']: event for event in events}

    print(f'request_id_map', flush=True)
    print(request_id_map, flush=True)

    try:
        stmt = session.query(Event)\
            .filter(
                Event.request_id.in_(list(request_id_map.keys())),
                Event.organization_id == organization_id)\
            .order_by(Event.request_id)
        data = stmt.all()
        print(f'data', flush=True)
        print(data, flush=True)
        group = []
        current_request_id = ''
        for row in data:
            if current_request_id != row['request_id']:
                resolve_update(group, request_id_map[current_request_id], session)
                current_request_id = row['request_id']
                group = []
            group.append(row)
        resolve_update(group, request_id_map[current_request_id], session)
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
                                print(f'events_to_update {len(events_to_update)}', flush=True)
                                events_to_create = list(filter(lambda x: 'request_id' not in x, batched_events))
                                print(f'events_to_create {len(events_to_create)}', flush=True)
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
            processed_events = process_events(batched_events, organization_id)
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
        print(f'events_to_update {len(events_to_update)}', flush=True)
        events_to_create = list(filter(lambda x: 'request_id' not in x, records))
        print(f'events_to_create {len(events_to_create)}', flush=True)
        update_events(events_to_update, organization_id)
        events = process_events(records, organization_id)
        flush_events(events)
    elif 'urls' in body:
        print(f"Received {len(body['urls'])} batch urls for organization {organization_id}")
        thread_pool.submit(process_batches, body['urls'], organization_id)
    else:
        raise werkzeug.exceptions.BadRequest('No records or batch urls provided.')

    return {}, 200
