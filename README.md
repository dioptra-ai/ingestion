# Ingestion

## Test data ingestion locally

1. Start all service with docker compose
1. Refer to this link to set env variables and send requests to the local service
   https://docs.aws.amazon.com/lambda/latest/dg/images-test.html#images-test-env
1. Send an ingestion request with the following:

   ```bash
   curl -XPOST "http://localhost:8082/2015-03-31/functions/function/invocations" \
   -H 'Content-Type: application/json; charset=utf-8' \
   -d @- << EOF
   {
      "organization_id": "63bf861b03cf1bef5aea0dc3",
      "records": [{
          ...
      }]
   }
   EOF
   ```
