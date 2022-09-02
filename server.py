import logging
import os
import werkzeug
from multiprocessing import Pool
from flask import Flask, request
from schemas.pgsql import models, get_session
import sqlalchemy
from helpers import compatibility, common_processing
from helpers.eventprocessor import event_processor
from functools import partial
import orjson
from smart_open import open as smart_open

Event = models.event.Event

app = Flask(__name__)

@app.errorhandler(werkzeug.exceptions.BadRequest)
def handle_bad_request(e):
    logging.exception(e)
    return str(e), 400

@app.errorhandler(Exception)
def handle_exception(e):
    logging.exception(e)
    return 'Unexpected Error', 500

def process_events(events, organization_id):
    with Pool(os.cpu_count()) as p:
        events = p.map(compatibility.process, events)
        events = p.map(
            partial(common_processing.process,
                    organization_id=organization_id),
            events
        )

        return p.map(event_processor.process, events)

def flush_events(events):
        session = get_session()
        try:
            session.add_all([Event(**r) for r in events])
            session.commit()
            logging.info(f'Flushed {len(events)} events')
        except TypeError as e:

            raise werkzeug.exceptions.BadRequest(str(e))

        except sqlalchemy.exc.ProgrammingError as e:

            raise werkzeug.exceptions.BadRequest(str(e).split(
                '\n')[0].replace('(psycopg2.errors.DatatypeMismatch)', '')
            )

POSTGRES_MAX_BATCH_SIZE = 1000

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
        batched_events = []
        # TODO: Add params to the body for S3 auth

        for url in body['urls']:
            for dioptra_record_str in smart_open(url):
                processed_events = process_events(orjson.loads(dioptra_record_str))

                if len(batched_events) + len(process_events) >= POSTGRES_MAX_BATCH_SIZE:
                    flush_events(batched_events)
                    batched_events = []

                batched_events.extend(processed_events)

        if len(batched_events):
            flush_events(batched_events)
            batched_events = []
    else:
        raise werkzeug.exceptions.BadRequest('No records or batch urls provided.')
