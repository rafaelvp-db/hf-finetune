# Databricks notebook source
!pip install --upgrade pip && pip install --upgrade transformers && pip install --upgrade wheel && pip install pyspark==3.3.0 evaluate==0.1.2
!pip install --upgrade datasets

# COMMAND ----------

import logging
logger = spark._jvm.org.apache.log4j
logging.getLogger("py4j.java_gateway").setLevel(logging.ERROR)

import datasets
import torch

from transformers import (
  AutoTokenizer,
  AutoModelForCausalLM,
  DataCollatorWithPadding,
  Trainer,
  TrainingArguments
)

TARGET_DIR = "/dbfs/tmp/persuasion4good"
!rm -rf {TARGET_DIR}/trainer
!cp -r {TARGET_DIR}/dataset /tmp/dataset

dataset = datasets.load_from_disk("/tmp/dataset")

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Loading our HF Tokenizer and Backbone Model

# COMMAND ----------

#We will create a custom tokenization function leveraging microsoft/DialoGPT-Medium tokenizer

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
print(f"Model max length: {tokenizer.model_max_length}")
print(f"Model max length single sentence: {tokenizer.max_len_single_sentence}")
print(f"Model max_model_input_sizes: {tokenizer.max_model_input_sizes}")

def tokenize(batch, tokenizer, max_length = 1024):
  tokenizer.pad_token = tokenizer.eos_token
  batch["context"] = batch["context"].replace("<EOS>", tokenizer.eos_token)
  
  input_ids = tokenizer(
    batch["context"],
    return_tensors = "pt",
    return_attention_mask = False,
    padding = "max_length",
    truncation = True,
    max_length = max_length
  )["input_ids"][0]
  
  labels = tokenizer(
      batch["label"],
      return_tensors = "pt",
      return_attention_mask = False,
      padding = "max_length",
      max_length = max_length,
      truncation = True
    )["input_ids"][0]
  
  result = {
    "input_ids": input_ids,
    "labels": labels
  }
  
  return result

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Tokenizing our dataset

# COMMAND ----------

import numpy as np
import torch

#Let's try to have an idea about the sizes we are dealing with

embedded_lengths = []

for i in range(0, 100):
  embedded = tokenizer(dataset["train"].shuffle()[0]["context"], return_tensors = "pt", return_attention_mask = False)
  embedded_lengths.append(len(torch.flatten(embedded["input_ids"])))
  
  
mean_embed_size = np.mean(embedded_lengths)
std_embed_size = np.std(embedded_lengths)

print("Average embedding length: ", mean_embed_size)
print("Median embedding length: ", np.median(embedded_lengths))
print("Max embedding length: ", np.max(embedded_lengths))
print("Min embedding length: ", np.min(embedded_lengths))
print("STD for embedding length: ", std_embed_size)

# COMMAND ----------

max_length = int(mean_embed_size + (2 * std_embed_size))
print("Tokenizing with padding to max length: ", max_length)
dataset = dataset.map(lambda x: tokenize(x, tokenizer, max_length = max_length), remove_columns = ["label", "context"])

# COMMAND ----------

# DBTITLE 1,Checking our outputs
print("Sample (encoded) input_ids: ", dataset["train"]["input_ids"][0][:10])
print("Sample (encoded) labels: ", dataset["train"]["labels"][0][:10])

# COMMAND ----------

# DBTITLE 1,Sanity Checks
tokenizer.decode(dataset["train"]["input_ids"][0])
tokenizer.decode(dataset["train"]["labels"][0])

# COMMAND ----------

# DBTITLE 1,Computing Our Metrics
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    perplexity = load("perplexity", module_type="metric")
    perplexity_metric = perplexity.compute(predictions = predictions, model_id='gpt2')
    return {"perplexity": perplexity_metric}

# COMMAND ----------

# DBTITLE 1,Downloading our Backbone Model
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
model

# COMMAND ----------

# By default, Hugging Face has an MLflow callback. We need to setup some parameters.

import os

os.environ["MLFLOW_EXPERIMENT_NAME"] = "/Users/rafael.pierre@databricks.com/persuasion4good"
os.environ["MLFLOW_FLATTEN_PARAMS"] = "1"
os.environ["HF_MLFLOW_LOG_ARTIFACTS"] = "1"
os.environ["MLFLOW_NESTED_RUN"] = "1"

# COMMAND ----------

from transformers import EarlyStoppingCallback, IntervalStrategy
from  transformers.trainer_utils import SchedulerType

tokenizer.pad_token = tokenizer.eos_token
#collator = DataCollatorWithPadding(tokenizer = tokenizer, padding = "max_length", max_length = 1024)

args = TrainingArguments(
  output_dir = f"{TARGET_DIR}/trainer/",
  per_device_train_batch_size = 2,
  per_device_eval_batch_size = 1,
  eval_accumulation_steps = 1,
  learning_rate = 2e-5,
  weight_decay = 0.7,
  adam_epsilon = 1e-8,
  max_grad_norm = 1.0,
  num_train_epochs = 100.0,
  lr_scheduler_type = SchedulerType.CONSTANT_WITH_WARMUP,
  warmup_steps = 5,
  logging_steps = 100,
  save_steps = 1000,
  no_cuda = False,
  overwrite_output_dir = True,
  seed = 42,
  local_rank = -1,
  fp16 = True,
  fp16_opt_level = 'O1',
  metric_for_best_model = 'perplexity'
)

trainer = Trainer(
  #data_collator = collator,
  compute_metrics = compute_metrics,
  model = model.cuda(),
  args = args,
  train_dataset = dataset["train"],
  eval_dataset = dataset["test"],
  tokenizer = tokenizer
)

# COMMAND ----------

import mlflow
import torch

torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
trainer.train()

# COMMAND ----------

# TODO: keep everything inside the same run
model = trainer.model
model_info = mlflow.pytorch.log_model(model, artifact_path = "model")

# COMMAND ----------

trainer.save_model("/tmp/persuasion4good/model")
tokenizer.save_pretrained("/tmp/persuasion4good/tokenizer")

trainer.save_model("/dbfs/tmp/persuasion4good/model")
tokenizer.save_pretrained("/dbfs/tmp/persuasion4good/tokenizer")
