# Databricks notebook source
!pip install --upgrade pip && pip install --upgrade transformers

# COMMAND ----------

from transformers.pytorch_utils import *
import mlflow
from mlflow.tracking import MlflowClient
import torch

client = MlflowClient()

!mkdir /tmp/model

client.download_artifacts(
  run_id = "4b2c833519dd481086f4e7b6d291d0e3",
  path = "checkpoint-4000",
  dst_path = "/tmp/model/"
)

# COMMAND ----------

TARGET_DIR = "/tmp/model/checkpoint-4000/artifacts/checkpoint-4000"
!ls {TARGET_DIR}

# COMMAND ----------

from transformers import AutoModelWithLMHead, AutoConfig, AutoTokenizer

config = AutoConfig.from_pretrained(TARGET_DIR)
model = AutoModelWithLMHead.from_pretrained(TARGET_DIR, config = config)
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")

model.to("cpu")

# COMMAND ----------

import torch
import numpy as np

def ask_question(
  question,
  chat_history_ids = [],
  max_length = 20,
  temperature = 3.0,
  repetition_penalty = 20.0
):
  
  new_user_input_ids = tokenizer.encode(
    str(question) + str(tokenizer.eos_token),
    return_tensors='pt'
  )

  chat_history_tensor = torch.from_numpy(np.array(chat_history_ids))
  chat_history_ids = torch.reshape(input = chat_history_tensor, shape = (-1,))

  if (len(chat_history_ids) > 0):
    bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=0)
  else:
    bot_input_ids = new_user_input_ids

  chat_history_ids = model.generate(
    bot_input_ids,
    eos_token_id = tokenizer.eos_token_id,
    max_length = max_length,
    pad_token_id = tokenizer.eos_token_id,  
    no_repeat_ngram_size = 3,
    do_sample  = True, 
    top_k = 10, 
    top_p = 0.9,
    repetition_penalty = repetition_penalty,
    temperature = temperature
  )

  answer = tokenizer.decode(
    chat_history_ids[:, bot_input_ids.shape[-1]:][0],
    skip_special_tokens = True
  )

  return answer, chat_history_ids

  
def predict(model_input):

  answer, chat_history_ids = ask_question(
    question = model_input["question"],
    chat_history_ids = model_input["chat_history_ids"]
  )

  result = {
    "answer": answer,
    "chat_history_ids": chat_history_ids[0].tolist()
  }

  return result

# COMMAND ----------

model_input = {
  "question": "hi",
  "chat_history_ids": []
}

answers = predict(model_input)
answers
