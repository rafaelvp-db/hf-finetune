# Databricks notebook source
!pip install --upgrade pip && pip install --upgrade transformers && pip install --upgrade evaluate wheel && pip install pyspark==3.3.0
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

dataset = datasets.load_from_disk(f"{TARGET_DIR}/dataset", keep_in_memory = True)
train_dataset = dataset["train"]
test_dataset = dataset["test"]

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Loading our HF Tokenizer and Backbone Model

# COMMAND ----------

#We will create a custom tokenization function leveraging microsoft/DialoGPT-Medium tokenizer

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small", return_special_tokens_mask = True)
print(f"Model max length: {tokenizer.model_max_length}")
print(f"Model max length single sentence: {tokenizer.max_len_single_sentence}")
print(f"Model max_model_input_sizes: {tokenizer.max_model_input_sizes}")

def tokenize(batch, tokenizer, feature, eos = True, max_length = 1024, pad_to_multiple_of = 8):
  tokenizer.pad_token = tokenizer.eos_token
  
  if eos:
    batch[feature] = batch[feature].replace("<EOS>", tokenizer.eos_token)
    
  remainder = pad_to_multiple_of - (max_length % pad_to_multiple_of)
  max_length_multiple = max_length + remainder
  
  input_ids = tokenizer(
    batch[feature],
    return_tensors = "pt",
    return_attention_mask = False,
    padding = "max_length",
    truncation = "longest_first",
    max_length = max_length_multiple,
    pad_to_multiple_of = pad_to_multiple_of
  )["input_ids"][0]
  
  result_key = "labels"
  if feature != "label":
    result_key = "input_ids"
    
  result = {
    f"{result_key}": input_ids
  }
  
  return result

# COMMAND ----------

# DBTITLE 1,Making sure we don't have too small utterances
dataset = dataset.filter(lambda example: len(example['label']) > 2 and len(example['context']) > 2)

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
percentile_95th = np.quantile(embedded_lengths, 0.85)
std_embed_size = np.std(embedded_lengths)

print("Average embedding length: ", mean_embed_size)
print("Median embedding length: ", np.median(embedded_lengths))
print("95th percentile - embedding length: ", percentile_95th)
print("Max embedding length: ", np.max(embedded_lengths))
print("Min embedding length: ", np.min(embedded_lengths))
print("STD for embedding length: ", std_embed_size)

# COMMAND ----------

max_input_length = int(percentile_95th)
print("Tokenizing with padding to ", max_input_length)
dataset = dataset.map(lambda x: tokenize(x, tokenizer, feature = "context", max_length = max_input_length), remove_columns = ["context"])
dataset = dataset.map(lambda x: tokenize(x, tokenizer, feature = "label", eos = False, max_length = max_input_length), remove_columns = ["label"])

# COMMAND ----------

# DBTITLE 1,Checking our outputs
print("Sample (encoded) input_ids: ", dataset["train"]["input_ids"][0][:10], "length: ", len(dataset["train"]["input_ids"][0]))
print("Sample (encoded) labels: ", dataset["train"]["labels"][0][:10], "length: ", len(dataset["train"]["labels"][0]))

# COMMAND ----------

# DBTITLE 1,Sanity Checks
print("Decoded embedding - context: ", tokenizer.decode(dataset["train"].shuffle()["input_ids"][0]), "\n")
print("Decoded embedding - labels: ", tokenizer.decode(dataset["train"].shuffle()["labels"][0]))

# COMMAND ----------

# DBTITLE 1,Downloading our Backbone Model
from transformers import AutoModelWithLMHead, AutoConfig #AutoModelForCausalLM
config = AutoConfig.from_pretrained("microsoft/DialoGPT-small")
model = AutoModelWithLMHead.from_pretrained(
  "microsoft/DialoGPT-small",
  config = config
)

# COMMAND ----------

# By default, Hugging Face has an MLflow callback which is alredy switched on. We just need to setup some parameters.

import os

os.environ["MLFLOW_EXPERIMENT_NAME"] = "/Users/rafael.pierre@databricks.com/persuasion4good"
os.environ["MLFLOW_FLATTEN_PARAMS"] = "1"
os.environ["HF_MLFLOW_LOG_ARTIFACTS"] = "1"
os.environ["MLFLOW_NESTED_RUN"] = "1"

# COMMAND ----------

def compute_metrics(eval_loss: float):
  
  loss = torch.exp(eval_loss).float()
  return loss

# COMMAND ----------

from transformers import EarlyStoppingCallback, IntervalStrategy, DataCollatorForLanguageModeling

tokenizer.pad_token = tokenizer.eos_token
collator = DataCollatorForLanguageModeling(
  tokenizer = tokenizer,
  mlm = False,
  pad_to_multiple_of = 8
)

args = TrainingArguments(
  output_dir = f"{TARGET_DIR}/trainer/",
  evaluation_strategy = IntervalStrategy.STEPS, # "steps"
  eval_steps = 50, # Evaluation and Save happens every 50 steps
  save_total_limit = 5, # Only last 5 models are saved. Older ones are deleted.
  per_device_train_batch_size = 4,
  per_device_eval_batch_size = 4,
  eval_accumulation_steps = 1,
  learning_rate = 5e-5,
  weight_decay = 0.0,
  adam_epsilon = 1e-8,
  warmup_steps = 0.0,
  max_grad_norm = 1.0,
  num_train_epochs = 10000.0,
  logging_steps = 1000,
  no_cuda = False,
  overwrite_output_dir = True,
  seed = 42,
  local_rank = -1,
  fp16 = False,
  metric_for_best_model = 'eval_loss',
  load_best_model_at_end = True,
  disable_tqdm = False,
  prediction_loss_only=True
)

trainer = Trainer(
  data_collator = collator,
  compute_metrics = compute_metrics,
  model = model.cuda(),
  args = args,
  train_dataset = dataset["train"],
  eval_dataset = dataset["test"],
  tokenizer = tokenizer,
  callbacks = [EarlyStoppingCallback(
    early_stopping_patience = 5,
    early_stopping_threshold = 0.0005
  )]
)

# COMMAND ----------

import mlflow
import torch

torch.cuda.empty_cache()

with mlflow.start_run(nested = True) as run:
  #os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
  trainer.train()
  metrics = trainer.evaluate()
  print(metrics)
  mlflow.log_metrics(metrics)
  model = trainer.model
  model_info = mlflow.pytorch.log_model(model, artifact_path = "model")

# COMMAND ----------