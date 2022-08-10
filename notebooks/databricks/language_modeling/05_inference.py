# Databricks notebook source
!pip install --upgrade pip && pip install --upgrade transformers

# COMMAND ----------

from transformers.pytorch_utils import *
from transformers import AutoTokenizer
import mlflow
import torch

model = mlflow.pytorch.load_model("runs:/a26ba30c318c424496f37876ef8b9450/model")
model.cuda(0)

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")

# COMMAND ----------

import torch
import numpy as np

def ask_question(
  question,
  chat_history_ids = [],
  max_length = 50,
  temperature = 80.0,
  repetition_penalty = 70.0
):
  
  new_user_input_ids = tokenizer.encode(
    str(question) + str(tokenizer.eos_token),
    return_tensors='pt'
  )

  chat_history_ids = torch.from_numpy(np.array(chat_history_ids))

  if (len(chat_history_ids) > 0):
    bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1).cuda(0)
  else:
    bot_input_ids = new_user_input_ids.cuda(0)

  chat_history_ids = model.generate(
    bot_input_ids,
    eos_token_id = tokenizer.eos_token_id,
    max_length=max_length,
    pad_token_id = tokenizer.eos_token_id,  
    no_repeat_ngram_size=3,
    do_sample=True, 
    top_k=100, 
    top_p=0.7,
    repetition_penalty = repetition_penalty,
    temperature=temperature
  ).cuda(0)

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
  "question": "hi.",
  "chat_history_ids": []
}

answers = predict(model_input)
answers
