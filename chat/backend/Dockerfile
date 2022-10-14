FROM python:3.8.13-slim-buster as builder

ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
ARG APP_NAME=persuasion4good

RUN mkdir /app \
    && groupadd -r persuasion4good \
    && useradd -r -s /bin/false -g persuasion4good persuasion4good \
    && chown -R persuasion4good:persuasion4good /app

FROM builder as model
WORKDIR /app
USER root

COPY ./app /app
RUN pip install --upgrade pip && pip install -r requirements.txt

FROM model as prod
USER persuasion4good
WORKDIR /app

CMD ["uvicorn", "proxy:app", "--host", "0.0.0.0", "--port", "8080"]