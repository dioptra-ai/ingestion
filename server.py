from flask import Flask

app = Flask(__name__)

@app.route('/ingest', methods = ['POST'])
def ingest():
    print('Ingestion service received a request')
    return 'ingested'
