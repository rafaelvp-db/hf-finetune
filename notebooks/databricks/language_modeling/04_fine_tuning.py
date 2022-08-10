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

def tokenize(batch, tokenizer):
  tokenizer.pad_token = tokenizer.eos_token
  batch["context"] = batch["context"].replace("<EOS>", tokenizer.eos_token)
  
  input_ids = tokenizer(
    batch["context"],
    return_tensors = "pt",
    return_attention_mask = False,
    padding = "max_length",
    truncation = True,
    max_length = 512
  )["input_ids"][0]
  
  labels = tokenizer(
      batch["label"],
      return_tensors = "pt",
      return_attention_mask = False,
      padding = "max_length",
      max_length = 512,
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

dataset = dataset.map(lambda x: tokenize(x, tokenizer), remove_columns = ["label", "context"])

# COMMAND ----------

# DBTITLE 1,Checking our outputs
print("Sample (encoded) input_ids: ", dataset["train"]["input_ids"][0][:10])
print("Sample (encoded) labels: ", dataset["train"]["labels"][0][:10])

# COMMAND ----------

# DBTITLE 1,Computing Our Metrics
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    perplexity = load("perplexity", module_type="metric")
    return perplexity.compute(predictions = predictions, model_id='gpt2')

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

tokenizer.pad_token = tokenizer.eos_token
collator = DataCollatorWithPadding(tokenizer = tokenizer, padding = "max_length", max_length = 512)

args = TrainingArguments(
  output_dir = f"{TARGET_DIR}/trainer/",
  per_device_train_batch_size = 2,
  learning_rate = 5e-5,
  weight_decay = 0.0,
  adam_epsilon = 1e-8,
  max_grad_norm = 1.0,
  num_train_epochs = 3,
  warmup_steps = 0,
  logging_steps = 1000,
  save_steps = 3500,
  save_total_limit = None,
  no_cuda = False,
  overwrite_output_dir = True,
  seed = 42,
  local_rank = -1,
  fp16 = True,
  fp16_opt_level = 'O1'
)

trainer = Trainer(
  data_collator = collator,
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
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
model_info = None

with mlflow.start_run(run_name = "/Users/rafael.pierre@databricks.com/hf-persuasion4good", nested=True) as run:
  trainer.train()
  model = trainer.model
  model_info = mlflow.pytorch.log_model(model, artifact_path = "model")
  
print(f"Model info: {model_info}")

# COMMAND ----------

# DBTITLE 1,Calculate Metrics for the Testing Set
#from evaluate import load

#def compute_metrics(eval_pred):
    #perplexity = load("perplexity", module_type="metric")
    #predictions, labels = eval_pred
    #return perplexity.compute(predictions = predictions, model_id='gpt2')

# COMMAND ----------

trainer.save_model("/tmp/persuasion4good/model")
tokenizer.save_pretrained("/tmp/persuasion4good/tokenizer")

trainer.save_model("/dbfs/tmp/persuasion4good/model")
tokenizer.save_pretrained("/dbfs/tmp/persuasion4good/tokenizer")

# COMMAND ----------

with mlflow.start_run(run_name = "/Users/rafael.pierre@databricks.com/hf-persuasion4good", nested=True) as run:
  model = trainer.model
  model_info = mlflow.pytorch.log_model(model, artifact_path = "model")
