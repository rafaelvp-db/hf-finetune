# Databricks notebook source
!pip install --upgrade pip && pip install --upgrade transformers evaluate wheel datasets

# COMMAND ----------

import logging
logger = spark._jvm.org.apache.log4j
logging.getLogger("py4j.java_gateway").setLevel(logging.ERROR)

from dataset import PersuasionDataset

from transformers import (
  AutoTokenizer,
  AutoModelForCausalLM,
  DataCollatorWithPadding,
  Trainer,
  TrainingArguments
)

TARGET_DIR = "/dbfs/tmp/persuasion4good"
!rm -rf {TARGET_DIR}/trainer

# COMMAND ----------

from torch.utils.data import random_split

full_dataset = PersuasionDataset(
  database = "persuasiondb",
  table = "context_final",
  max_length_input = 512
)

train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Loading our HF Tokenizer and Backbone Model

# COMMAND ----------

# DBTITLE 1,Downloading our Backbone Model
from transformers import AutoModelForCausalLM, AutoConfig
config = AutoConfig.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained(
  "microsoft/DialoGPT-medium",
  config = config
)

# COMMAND ----------

# DBTITLE 1,DialoGPT Example
model

# COMMAND ----------

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# Let's chat for 5 lines
for step in range(5):
    # encode the new user input, add the eos_token and return a tensor in Pytorch
    new_user_input_ids = tokenizer.encode(input(">> User:") + tokenizer.eos_token, return_tensors='pt')

    # append the new user input tokens to the chat history
    bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids

    # generated a response while limiting the total chat history to 1000 tokens, 
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    # pretty print last ouput tokens from bot
    print("DialoGPT: {}".format(tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)))

# COMMAND ----------

# DBTITLE 1,Configuring MLflow Integration
# By default, Hugging Face has an MLflow callback which is alredy switched on. 
# We just need to setup some parameters.

import os

os.environ["MLFLOW_EXPERIMENT_NAME"] = "/Users/rafael.pierre@databricks.com/persuasion4good"
os.environ["MLFLOW_FLATTEN_PARAMS"] = "1"
os.environ["HF_MLFLOW_LOG_ARTIFACTS"] = "1"
os.environ["MLFLOW_NESTED_RUN"] = "1"
os.environ["WANDB_DISABLED"] = "1"

# COMMAND ----------

# DBTITLE 1,Freezing Layers
for param in model.transformer.parameters():
  param.requires_grad = False

for param in model.lm_head.parameters():
  param.requires_grad = True

# COMMAND ----------

from evaluate import load

def compute_metrics(eval_preds):
  metric = load("perplexity", module_type="metric")
  results = metric.compute(predictions=eval_preds, model_id='gpt2')
  return results

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
  eval_steps = 100, # Evaluation and Save happens every 50 steps
  save_total_limit = 5, # Only last 5 models are saved. Older ones are deleted.
  per_device_train_batch_size = 4,
  per_device_eval_batch_size = 2,
  eval_accumulation_steps = 10,
  learning_rate = 5e-5,
  weight_decay = 0.0,
  adam_epsilon = 1e-8,
  warmup_steps = 0.0,
  max_grad_norm = 1.0,
  num_train_epochs = 1.0,
  logging_steps = 10,
  no_cuda = False,
  overwrite_output_dir = True,
  seed = 42,
  local_rank = -1,
  fp16 = False,
  metric_for_best_model = 'eval_loss',
  greater_is_better = False,
  load_best_model_at_end = True,
  disable_tqdm = False,
  prediction_loss_only=True,
  report_to = ["mlflow"]
)

trainer = Trainer(
  compute_metrics = compute_metrics,
  model = model.cuda(),
  args = args,
  train_dataset = train_dataset,
  eval_dataset = test_dataset,
  #tokenizer = tokenizer,
  callbacks = [EarlyStoppingCallback(
    early_stopping_patience = 5,
    early_stopping_threshold = 0.0001
  )]
)

# COMMAND ----------

import mlflow
import torch

torch.cuda.empty_cache()

with mlflow.start_run(nested = True) as run:
  os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
  trainer.train()
  metrics = trainer.evaluate()
  print(metrics)
  mlflow.log_metrics(metrics)
  model = trainer.model
  model_info = mlflow.pytorch.log_model(model, artifact_path = "model")

# COMMAND ----------


