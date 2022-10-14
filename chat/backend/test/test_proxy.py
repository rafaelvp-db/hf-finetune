from fastapi.testclient import TestClient
from fastapi import Request
from ..app.proxy import app


client = TestClient(app)


def test_predict():
    json_input = {
        "question": "hi how are you today?",
        "chat_history_ids": []
    }

    response = client.post("/predict", json = json_input)
    assert response.status_code == 200
    assert len(response.json()["answer"]) > 0

def test_bot_message():
    response = client.post("/bot", json = {"": "hi how are you today?"})
    assert response.status_code == 200
    assert response.json()