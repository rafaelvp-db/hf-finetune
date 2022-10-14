import json
import logging
import os
import requests
import traceback
import uuid
from twilio.twiml.messaging_response import MessagingResponse

from fastapi import (
    FastAPI, 
    Request
)

from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware

from fastapi.logger import logger as fastapi_logger

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

gunicorn_error_logger = logging.getLogger("gunicorn.error")
gunicorn_logger = logging.getLogger("gunicorn")
uvicorn_access_logger = logging.getLogger("uvicorn.access")
uvicorn_access_logger.handlers = gunicorn_error_logger.handlers
fastapi_logger.handlers = gunicorn_error_logger.handlers

# Define global variables
SERVICE_NAME = "proxy"
MAX_LENGTH = int(os.environ["MAX_LENGTH"])
SEP_TOKEN = int(os.environ["SEP_TOKEN"])
MAX_TURNS = int(os.environ["MAX_TURNS"])
DATABRICKS_TOKEN = os.environ["DATABRICKS_TOKEN"]
MODEL_ENDPOINT_URL = os.environ["MODEL_ENDPOINT_URL"]

# Initialize the FastAPI app
app = FastAPI(title=SERVICE_NAME)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


if __name__ != "__main__":
    fastapi_logger.setLevel(gunicorn_logger.level)
else:
    fastapi_logger.setLevel(logging.DEBUG)

def get_answer_payload(question: str, chat_history_ids = []):

    item = {
        "question": question,
        "chat_history_ids": chat_history_ids
    }

    num_turns = 0
    if len(item["chat_history_ids"]) > 0:
        item["chat_history_ids"] = item["chat_history_ids"][0]
        num_turns = item["chat_history_ids"].count(SEP_TOKEN)
    logger.info(f"Number of convo turns: {num_turns}")
    if num_turns >= MAX_TURNS:
        logger.info(f"Resetting chat history...")
        item["chat_history_ids"] = []

    # Make predictions and log
    headers = {
        "Authorization": f"Bearer {DATABRICKS_TOKEN}",
        "Content-type": "application/json"
    }

    request_payload = {"inputs": item}
    response = requests.post(
        url = MODEL_ENDPOINT_URL,
        headers = headers,
        data = json.dumps(request_payload)
    )
    logger.info(f'response: {response.json()}')
    json_response = json.loads(response.text)
    answer = json_response["predictions"]["answer"]
    chat_history_ids = json_response["predictions"]["chat_history_ids"]

    #If history is too long, reset it
    if len(chat_history_ids) >= MAX_LENGTH:
        logger.info(f"History is longer than MAX_LENGTH ({MAX_LENGTH}), resetting it")
        chat_history_ids = []
    
    model_output = {
        "answer": answer,
        "chat_history_ids": chat_history_ids
    }

    # Make response payload
    payload = jsonable_encoder(model_output)

    return payload


@app.post("/predict")
async def predict(
    request: Request,
    item: dict
):
    """Prediction endpoint.
    1. This should be a post request!
    2. Make sure to post the right data.
    item: dict. Example:
        {
            "question": "hi, how are you?",
            "chat_history_ids": []
        }
    """

    try:
        # Parse data
        fastapi_logger.info(f"Input: {str(item)}")
        logger.info(f"Headers: {request.headers}")

        # Define UUID for the request
        request_id = uuid.uuid4().hex

        # Log input data
        fastapi_logger.info(json.dumps({
            "service_name": SERVICE_NAME,
            "type": "InputData",
            "request_id": request_id,
            "data": item,
        }))

        payload = get_answer_payload(item)

        # Log output data
        fastapi_logger.info(json.dumps({
            "service_name": SERVICE_NAME,
            "type": "OutputData",
            "request_id": request_id,
            "data": payload
        }))
        
    except Exception as e:
        payload = {
            "answer": "I'm not sure I got that"
        }
        fastapi_logger.error(f"Error: {traceback.format_exc()}")

    headers = {
        "Access-Control-Allow-Private-Network": "true"
    }
    response = JSONResponse(
        headers=headers,
        content=payload
    )

    return response

@app.post("/bot")
async def bot(
    request: Request,
    item: dict
):

    resp = None
    try:
        logger.info(f"Bot call - item: {item}")
        #incoming_message = item["body"]
        incoming_message = request.get("Body", "").lower()
        payload = get_answer_payload(question = incoming_message)
        resp = MessagingResponse()
        msg = resp.message()
        msg.body(payload["answer"])
        logger.info(f"Response: {resp}")
    except Exception as e:
        fastapi_logger.error(f"Error: {traceback.format_exc()}")
        msg.body("I'm not sure I got that")
    finally:
        return str(resp)




