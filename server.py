import itertools
import logging
import os
from threading import Thread
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

def process_events(events, organization_id):
    with Pool(os.cpu_count()) as p:
        tic = time.time()
        events = p.map(compatibility.process, events)

        events = [e for e in events if e is not None]

        events = p.map(
            partial(event_processor.process_event, organization_id=organization_id),
            events
        )

        print(f'Processed {len(events)} events in {time.time() - tic} seconds')

        # event_processor.process_event returns a list of events for each parent event
        return list(itertools.chain(*events))

# 4GB k8s memory limit => up to 4MB per event
MAX_BATCH_SIZE = 1000

def flush_events(events):
    session = get_session()
    try:
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
            for dioptra_record_str in smart_open(url):
                try:
                    batched_events.append(orjson.loads(dioptra_record_str))
                except orjson.JSONDecodeError as e:
                    logging.warning(f'Failed to parse {dioptra_record_str}')
                    raise werkzeug.exceptions.BadRequest(f'Invalid JSON: {dioptra_record_str}')

                if len(batched_events) >= MAX_BATCH_SIZE:
                    processed_events = process_events(batched_events, organization_id)
                    flush_events(processed_events)
                    batched_events = []

            print(f'Processed {i + 1} of {len(urls)} batches')

        if len(batched_events):
            processed_events = process_events(batched_events, organization_id)
            flush_events(processed_events)
            batched_events = []
    except Exception as e:
        logging.error(f'Failed to process batches: {urls} for organization {organization_id}.')
        logging.exception(e)
        # TODO: Log this somewhere useful for the user to see ingestion failures.

@app.route('/ingest', methods = ['POST'])
def ingest():
    body = request.json
    organization_id = body['organization_id']
    records = []

    if 'records' in body:
        records = body['records']
        print(f'Received {len(records)} records for organization {organization_id}')
        events = process_events(records, organization_id)
        flush_events(events)
    elif 'urls' in body:
        print(f"Received {len(body['urls'])} batch urls for organization {organization_id}")
        daemon = Thread(target=process_batches, daemon=True, args=(body['urls'], organization_id))
        daemon.start()
    else:
        raise werkzeug.exceptions.BadRequest('No records or batch urls provided.')

    return {}, 200
