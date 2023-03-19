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

!rm -rf /tmp/model && mkdir -p /tmp/model

download_artifacts(
  run_id = "4cf4fc4957644a56b1b81c3beccb0d9c",
  artifact_path = "checkpoint-26000",
  dst_path = "/tmp/model/"
)

# COMMAND ----------

TARGET_DIR = "/tmp/model/checkpoint-26000/artifacts/checkpoint-26000"
!ls {TARGET_DIR}

# COMMAND ----------

from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

config = AutoConfig.from_pretrained(TARGET_DIR)
model = AutoModelForCausalLM.from_pretrained(TARGET_DIR, config = config)
tokenizer = AutoTokenizer.from_pretrained(padding_side = "left", pretrained_model_name_or_path = "gpt2")

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
  num_beams = 10,
  chat_history_ids = None,
  max_new_tokens = 30, 
  no_repeat_ngram_size = 3, 
  do_sample = False, 
  top_k = 10, 
  top_p = 0.7,
  repetition_penalty = 0.0,
  temperature = 0.1
):
  """
    This function is a wrapper on top of our fine tuned model.
    It preprocesses and tokenizes our inputs, and calls the generate function
    from the DialoGPT model we trained.
  """
  
  new_user_input_ids = tokenizer.encode(
    str(tokenizer.eos_token) + str(question),
    return_tensors = 'pt'
  )

  print(f"User input: {new_user_input_ids}")

  chat_history_tensor = torch.tensor(chat_history_ids)
  if (len(chat_history_ids) > 0):
    # Continuing an existing conversation
    bot_input_ids = torch.cat([chat_history_tensor, torch.flatten(new_user_input_ids)], dim = -1)
    bot_input_ids = torch.reshape(bot_input_ids, shape = (1, -1))
  else:
    # Beginning of conversation
    bot_input_ids = new_user_input_ids

  print("Bot input IDs: " + str(bot_input_ids))

  chat_history_ids = model.generate(
    bot_input_ids,
    num_beams = num_beams,
    max_new_tokens = max_new_tokens,
    pad_token_id = tokenizer.eos_token_id,  
    no_repeat_ngram_size = no_repeat_ngram_size,       
    do_sample = do_sample,
    top_k = top_k, 
    top_p = top_p,
    temperature = temperature
  )
  
  print("History: " + str(chat_history_ids))

  # Using our tokenizer to decode the outputs from our model
  # (e.g. convert model outputs to text)
  
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

# MAGIC %md
# MAGIC 
# MAGIC ## Chatting with our Model

# COMMAND ----------

question = "hi, can you tell me more about save the children?"

model_input = {
  "question": question,
  "chat_history_ids": []
}

result = predict(model_input)
answer = result["answer"]
chat_history_ids = result["chat_history_ids"]

print("Question: ", model_input["question"])
print("Answer: ", answer.split("   "))

# COMMAND ----------

question = "great, I would like to help"

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

question = "ten dollars"

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

question = "awesome, where do I sign?"

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

question = "haha, how is that playing out?"

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

question = "all right"

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

question = "thanks, you too!"

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


