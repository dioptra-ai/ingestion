from flask import Flask, request
from postgres import connection

app = Flask(__name__)

@app.route('/ingest', methods = ['POST'])
def ingest():
    records = request.json['records']
    print(f'records: {records}')

    cur = connection.cursor()
    for record in records:
        cur.execute(f"INSERT INTO events ({','.join(record.keys())}) VALUES ({','.join(['%s'] * len(record))})", list(record.values()))

    return f'Ingested {len(request.json["records"])} records'
