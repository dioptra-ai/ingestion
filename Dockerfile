FROM public.ecr.aws/lambda/python:3.9

RUN yum install -y gcc python3.9-devel postgresql-libs

COPY requirements.txt .
RUN pip3 install -r requirements.txt  --target "${LAMBDA_TASK_ROOT}"

COPY . ${LAMBDA_TASK_ROOT}

CMD [ "app.handler" ]
