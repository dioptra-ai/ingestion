from asyncio import events
from multiprocessing import Event
from flask import Flask, request
from schemas.pgsql import models, get_session
import sqlalchemy

Event = models.event.Event

app = Flask(__name__)

@app.route('/ingest', methods = ['POST'])
def ingest():
    records = request.json['records']
    print(f'records: {records}')

    session = get_session()
    try:
        session.add_all([Event(**r) for r in records])
        session.commit()
    except TypeError as e:

        return str(e), 400
    except sqlalchemy.exc.ProgrammingError as e:

        return str(e).split('\n')[0].replace('(psycopg2.errors.DatatypeMismatch)', ''), 400

    return f'Ingested {len(request.json["records"])} records'
