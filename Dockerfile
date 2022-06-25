FROM python:3.9.7-slim
MAINTAINER "Jacques Arnoux <jacques@dioptra.ai>"
EXPOSE 8082

WORKDIR /app/
COPY ./requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt
COPY . .

ENTRYPOINT uwsgi --master  --workers 4  --threads 50 --harakiri 300 --protocol http  --socket 0.0.0.0:8082  --module server:app
