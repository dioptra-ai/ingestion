FROM python:3.9-slim
EXPOSE 8082

WORKDIR /app/
RUN apt-get update && \
    apt-get install -y build-essential python3.9-dev libpq-dev
COPY ./requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt
COPY . .

ENTRYPOINT uwsgi --master --processes 4 --threads 50 --harakiri 300 --protocol http  --socket 0.0.0.0:8082  --module server:app
