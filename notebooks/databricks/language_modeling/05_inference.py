# Databricks notebook source
!ls -all /tmp/ubuntu

# COMMAND ----------

from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("/tmp/ubuntu/model")
tokenizer = AutoTokenizer.from_pretrained("/tmp/ubuntu/tokenizer")

# COMMAND ----------

import torch
import numpy as np

def ask_question(
  question,
  chat_history_ids = [],
  max_length = 1000,
  temperature = 50.0,
  repetition_penalty = 50.0
):
  
  new_user_input_ids = tokenizer.encode(
    str(question) + str(tokenizer.eos_token),
    return_tensors='pt'
  )

  chat_history_ids = torch.from_numpy(np.array(chat_history_ids))

  if (len(chat_history_ids) > 0):
    bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1)
  else:
    bot_input_ids = new_user_input_ids

  chat_history_ids = model.generate(
    bot_input_ids,
    eos_token_id = tokenizer.eos_token_id,
    max_length=100,
    pad_token_id = tokenizer.eos_token_id,  
    no_repeat_ngram_size=3,
    do_sample=True, 
    top_k=100, 
    top_p=0.7,
    repetition_penalty = 50.0,
    temperature=0.5
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
  "question": "I have an issue with my monitor.",
  "chat_history_ids": []
}

answers = predict(model_input)
answers
