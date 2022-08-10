import logging
import werkzeug
from multiprocessing import Event
from flask import Flask, request
from schemas.pgsql import models, get_session
import sqlalchemy

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
    records = request.json['records']
    print(f'records: {records}')

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

    return f'Ingested {len(request.json["records"])} records'
