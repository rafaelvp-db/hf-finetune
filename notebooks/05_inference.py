# Databricks notebook source
!pip install --upgrade pip && pip install --upgrade transformers -q

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Loading our Fine Tuned Model from MLflow

# COMMAND ----------

from transformers.pytorch_utils import *
import mlflow
from mlflow.artifacts import download_artifacts
import torch

client = MlflowClient()

!mkdir /tmp/model

download_artifacts(
  run_id = "cc4d089017d7446baaf93268dcc87aef",
  artifact_path = "checkpoint-1500",
  dst_path = "/tmp/model/"
)

# COMMAND ----------

TARGET_DIR = "/tmp/model/checkpoint-1500/artifacts/checkpoint-1500"
!ls {TARGET_DIR}

# COMMAND ----------

from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

config = AutoConfig.from_pretrained(TARGET_DIR)
model = AutoModelForCausalLM.from_pretrained(TARGET_DIR, config = config)
tokenizer = AutoTokenizer.from_pretrained(TARGET_DIR)

model.to("cpu")

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Generating our Chatbot Messages

# COMMAND ----------

import torch
import numpy as np

def generate(
  question,
  max_new_tokens = 128,
  chat_history_ids = [],
  temperature = 1.0,
  repetition_penalty = 0.0,
  do_sample = False,
  top_k = 1,
  top_p = 0.75,
  no_repeat_ngram_size = 3,
  length_penalty = 100.0
):
  """
    This function is a wrapper on top of our fine tuned model.
    It preprocesses and tokenizes our inputs, and calls the generate function
    from the DialoGPT model we trained.
  """
  
  new_user_input_ids = tokenizer.encode(
    str(question) + str(tokenizer.eos_token),
    return_tensors = 'pt'
  )

  print(f"User input: {new_user_input_ids}")

  chat_history_tensor = torch.tensor(chat_history_ids)
  if (len(chat_history_ids) > 0):
    # Beginning of conversation
    bot_input_ids = torch.cat([chat_history_tensor, torch.flatten(new_user_input_ids)], dim = -1)
    bot_input_ids = torch.reshape(bot_input_ids, shape = (1, -1))
  else:
    # Continuing an existing conversation
    bot_input_ids = new_user_input_ids
    
  chat_history_ids = model.generate(
    bot_input_ids,
    max_length = 96,
    pad_token_id = tokenizer.eos_token_id,  
    no_repeat_ngram_size = 3,       
    do_sample = True, 
    top_k = 100, 
    top_p = 0.7,
    temperature = 0.8
  )
  
  print(chat_history_ids)

  # Using our tokenizer to decode the outputs from our model
  # (e.g. convert model outputs to text)
  
  answer = tokenizer.decode(
    chat_history_ids[:, bot_input_ids.shape[-1]:][0],
    skip_special_tokens = True
  )

  return answer.split("  ")[0], chat_history_ids

  
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

# MAGIC %md
# MAGIC 
# MAGIC ## Chatting with our Model

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
print("Answer: ", answer.split("  "))

# COMMAND ----------

question = "nothing planned, you?"

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

question = "great, sounds fun."

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


