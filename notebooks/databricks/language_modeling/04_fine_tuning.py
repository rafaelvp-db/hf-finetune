# Databricks notebook source
!pip install --upgrade pip && pip install --upgrade transformers && pip install --upgrade wheel && pip install pyspark==3.3.0 huggingface_hub==0.7.0 evaluate==0.1.2
!pip install --upgrade datasets

# COMMAND ----------

!rm -rf /dbfs/tmp/ubuntu/trainer/

# COMMAND ----------

import logging
logger = spark._jvm.org.apache.log4j
logging.getLogger("py4j.java_gateway").setLevel(logging.ERROR)

# COMMAND ----------

from transformers import (
  AutoTokenizer,
  AutoModelForCausalLM,
  DataCollatorWithPadding,
  DataCollatorForLanguageModeling,
  AutoTokenizer,
  Trainer,
  TrainingArguments
)

import datasets
import torch
  
dataset = datasets.load_from_disk("/tmp/ubuntu/dataset")
dataset

# COMMAND ----------

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")

# COMMAND ----------

print(f"Model max length: {tokenizer.model_max_length}")
print(f"Model max length single sentence: {tokenizer.max_len_single_sentence}")
print(f"Model max_model_input_sizes: {tokenizer.max_model_input_sizes}")

# COMMAND ----------

def tokenize(batch, tokenizer):
  tokenizer.pad_token = tokenizer.eos_token
  batch["context"] = batch["context"].replace("<EOS>", tokenizer.eos_token)
  
  input_ids = tokenizer(
    batch["context"],
    return_tensors = "pt",
    return_attention_mask = False,
    padding = True,
    truncation = True
  )["input_ids"][0]
  
  labels = tokenizer(
      batch["label"],
      return_tensors = "pt",
      return_attention_mask = False,
      padding = "max_length",
      max_length = 1024,
      truncation = True
    )["input_ids"][0]
  
  result = {
    "input_ids": input_ids,
    "labels": labels
  }
  
  return result

# COMMAND ----------

dataset = dataset.map(lambda x: tokenize(x, tokenizer), remove_columns = ["label", "context"])

# COMMAND ----------

print("Sample (encoded) input_ids: ", dataset["train"]["input_ids"][0][:10])
print("Sample (encoded) labels: ", dataset["train"]["labels"][0][:10])

# COMMAND ----------

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    #predictions = np.argmax(predictions, axis=1)
    perplexity = load("perplexity", module_type="metric")
    return perplexity.compute(predictions = predictions, model_id='gpt2')

# COMMAND ----------

model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
model

# COMMAND ----------

tokenizer.pad_token = tokenizer.eos_token
collator = DataCollatorWithPadding(tokenizer = tokenizer, padding = "max_length", max_length = 1024)

args = TrainingArguments(
  output_dir = "/dbfs/tmp/ubuntu/trainer/",
  overwrite_output_dir = True,
  per_device_train_batch_size = 4
)

trainer = Trainer(
  data_collator = collator,
  compute_metrics = compute_metrics,
  model = model,
  args = args,
  train_dataset = dataset["train"],
  eval_dataset = dataset["test"],
  tokenizer = tokenizer
)

# COMMAND ----------

trainer.train()

#TODO:

# 1. Trainer callback

# COMMAND ----------

from evaluate import load

def compute_metrics(eval_pred):
    perplexity = load("perplexity", module_type="metric")
    predictions, labels = eval_pred
    return perplexity.compute(predictions = predictions, model_id='gpt2')

# COMMAND ----------

trainer.model.save_pretrained("/tmp/ubuntu/model")

# COMMAND ----------

tokenizer.save_pretrained("/tmp/ubuntu/tokenizer")

# COMMAND ----------

import mlflow

with mlflow.start_run() as run:
  model_info = mlflow.pytorch.log_model(model, artifact_path = "model")

# COMMAND ----------


