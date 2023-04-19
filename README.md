# Ingestion

## How to ingest data in your local environment

1. Start all services from the main repo: https://github.com/dioptra-ai/dioptra
2. Create an API key in the UI
3. Send data to the ingestion service using the following command:

```bash
$ curl http://localhost:8080/2015-03-31/functions/function/invocations \
    -H 'Content-Type: application/json' \
    -H 'x-api-key: <API_KEY_FROM_LOCAL_ENV>' \
    -d '{"records": [...]}'
```
