FROM public.ecr.aws/lambda/python:3.9 as builder

RUN yum install -y gcc python3.9-devel postgresql-libs

COPY requirements.txt .
RUN pip3 install -r requirements.txt  --target "${LAMBDA_TASK_ROOT}"

COPY . ${LAMBDA_TASK_ROOT}
RUN rm -rf ${LAMBDA_TASK_ROOT}/test_data

CMD [ "app.handler" ]

FROM builder as testrunner

RUN pip3 install pytest pytest-cov pytest-mock mock-alchemy
COPY ./test_data ${LAMBDA_TASK_ROOT}/test_data

RUN cd ${LAMBDA_TASK_ROOT} && POSTGRES_USER='mock' pytest helpers
