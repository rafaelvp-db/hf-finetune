# Databricks notebook source
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("/dbfs/tmp/ubuntu/")
model = AutoModelForCausalLM.from_pretrained("/dbfs/tmp/ubuntu/")

# COMMAND ----------

question = "hi I have an issue with my ubuntu desktop. can you help me?"
chat_history_ids = []

# COMMAND ----------

import torch
import numpy as np

def ask_question(question, chat_history_ids) -> (str, []):

  new_user_input_ids = tokenizer.encode(
    str(question) + str(tokenizer.eos_token),
    return_tensors='pt'
  )

  chat_history_ids = torch.from_numpy(np.array(chat_history_ids))
  bot_input_ids = []

  if (len(chat_history_ids) > 0):
    bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1)
  else:
    bot_input_ids = new_user_input_ids

  chat_history_ids = model.generate(
    bot_input_ids, max_length=200,
    pad_token_id=tokenizer.eos_token_id,  
    no_repeat_ngram_size=3,       
    do_sample=True, 
    top_k=100, 
    top_p=0.7,
    temperature=0.5,
    repetition_penalty=1.0
  )

  answer = tokenizer.decode(
    chat_history_ids[:, bot_input_ids.shape[-1]:][0],
    skip_special_tokens=True
  )
  
  return answer, chat_history_ids

# COMMAND ----------

answer, chat_history_ids = ask_question(question, chat_history_ids)
print(answer)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Gradio

# COMMAND ----------

def predict(question):
    print(question)
    answer, chat_history_ids = ask_question(question, [])
    return [answer, tokenizer.decode(chat_history_ids)]

# COMMAND ----------

!pip install gradio

# COMMAND ----------

import gradio as gr

# COMMAND ----------

gr.Interface(
  fn = predict,
  inputs=["text", "state"],
  outputs=["chatbot", "state"],
  flagging_dir="/dbfs/tmp/ubuntu/flagging/"
).launch(share=True)

# COMMAND ----------


