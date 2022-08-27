import logging
import os
import werkzeug
from multiprocessing import Pool
from flask import Flask, request
from schemas.pgsql import models, get_session
import sqlalchemy
from helpers import compatibility, common_processing
from functools import partial

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

@app.route('/ingest', methods = ['POST'])
def ingest():
    body = request.json
    records = body['records']
    organization_id = body['organization_id']
    print(f'Received {len(records)} records for organization {organization_id}')

    with Pool(os.cpu_count()) as p:
        records = p.map(compatibility.process, records)
        records = p.map(
            partial(common_processing.process, organization_id=organization_id),
            records
        )

        session = get_session()
        try:
            session.add_all([Event(**r) for r in records])
            session.commit()
        except TypeError as e:

            raise werkzeug.exceptions.BadRequest(str(e))

        except sqlalchemy.exc.ProgrammingError as e:

            raise werkzeug.exceptions.BadRequest(str(e).split(
                '\n')[0].replace('(psycopg2.errors.DatatypeMismatch)', '')
            )

        return f'Ingested {len(records)} records'
