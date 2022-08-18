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
  path = "checkpoint-8000",
  dst_path = "/tmp/model/"
)

# COMMAND ----------

TARGET_DIR = "/tmp/model/checkpoint-8000/artifacts/checkpoint-8000"
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

def generate(
  question,
  chat_history_ids = [],
  max_new_tokens = 20,
  temperature = 1.0,
  repetition_penalty = 10.0,
  do_sample = True,
  top_k = 5,
  top_p = 0.9,
  no_repeat_ngram_size = 3,
  length_penalty = 20.0
):
  
  new_user_input_ids = tokenizer.encode(
    str(question) + str(tokenizer.eos_token),
    return_tensors = 'pt'
  )

  chat_history_tensor = torch.tensor(chat_history_ids)
  if (len(chat_history_ids) > 0):
    bot_input_ids = torch.cat([chat_history_tensor, torch.flatten(new_user_input_ids)], dim = -1)
    bot_input_ids = torch.reshape(bot_input_ids, shape = (1, -1))
  else:
    bot_input_ids = new_user_input_ids
    
  chat_history_ids = model.generate(
    bot_input_ids,
    eos_token_id = tokenizer.eos_token_id,
    max_new_tokens = max_new_tokens,
    pad_token_id = tokenizer.eos_token_id,  
    no_repeat_ngram_size = no_repeat_ngram_size,
    do_sample  = do_sample, 
    top_k = top_k, 
    top_p = top_p,
    repetition_penalty = repetition_penalty,
    temperature = temperature,
    length_penalty = length_penalty,
    forced_eos_token_id = tokenizer.eos_token_id
  )
  
  """chat_history_ids = model.generate(
    bot_input_ids,
    eos_token_id = tokenizer.eos_token_id
  )"""

  answer = tokenizer.decode(
    chat_history_ids[:, bot_input_ids.shape[-1]:][0],
    skip_special_tokens = True
  )

  return answer, chat_history_ids

  
def predict(model_input):

  answer, chat_history_ids = generate(
    question = model_input["question"],
    chat_history_ids = model_input["chat_history_ids"]
  )

  result = {
    "answer": answer,
    "chat_history_ids": chat_history_ids[0].tolist()
  }

  return result

# COMMAND ----------

question = "hello"

model_input = {
  "question": question,
  "chat_history_ids": []
}

result = predict(model_input)
answer = result["answer"]
chat_history_ids = result["chat_history_ids"]

print("Question: ", model_input["question"])
print("Answer: ", answer)

# COMMAND ----------

question = "no, I haven't heard of them"

model_input = {
  "question": question,
  "chat_history_ids": chat_history_ids
}

result = predict(model_input)
answer = result["answer"]
chat_history_ids = result["chat_history_ids"]

print("Question: ", model_input["question"])
print("Answer: ", answer)

# COMMAND ----------

question = "sure, please"

model_input = {
  "question": question,
  "chat_history_ids": chat_history_ids
}

result = predict(model_input)
answer = result["answer"]
chat_history_ids = result["chat_history_ids"]

print("Question: ", model_input["question"])
print("Answer: ", answer)
