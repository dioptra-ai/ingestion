from flask import Flask, request

app = Flask(__name__)

@app.route('/ingest', methods = ['POST'])
def ingest():
    print('Ingestion service received a request')
    return f'Ingested {len(request.json["records"])} records'
