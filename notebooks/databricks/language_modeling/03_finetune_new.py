# Databricks notebook source
!pip install --upgrade pip && pip install --upgrade transformers && pip install --upgrade wheel && pip install pyspark==3.3.0 huggingface_hub==0.7.0 evaluate==0.1.2 

# COMMAND ----------

import logging
logger = spark._jvm.org.apache.log4j
logging.getLogger("py4j.java_gateway").setLevel(logging.ERROR)

# COMMAND ----------

import numpy as np
import pandas as pd
import pathlib
import pickle

from sklearn.model_selection import train_test_split

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm.notebook import tqdm, trange
from typing import List, Tuple, Dict

from pathlib import Path

from transformers import (
    MODEL_WITH_LM_HEAD_MAPPING,
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
)

# COMMAND ----------

from transformers import AutoTokenizer, PreTrainedTokenizer

class ConversationDataset(Dataset):
  def __init__(
    self,
    encodings,
    labels
  ):
    self.encodings = encodings
    self.labels = labels

  def __getitem__(self, idx):
    item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
    item['labels'] = torch.tensor(self.labels[idx])
    return item

  def __len__(self):
    return len(self.labels)

# COMMAND ----------

np.array()

# COMMAND ----------

import torch
tokenizer.pad_token = tokenizer.eos_token
tokenized = tokenizer([f"this is a test", f"another one"], return_tensors="pt", padding=True, truncation=True)
tokenized["input_ids"]
test_cat = [torch.cat((encoding, torch.tensor([tokenizer.eos_token_id])), dim=0) for encoding in tokenized["input_ids"]]
test_cat

# COMMAND ----------

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
tokenizer.pad_token = tokenizer.eos_token
my_list = ["this is a test", "another one"]
encodings = tokenizer(my_list, return_tensors="pt", padding=True, truncation=True)
torch.reshape(encodings["input_ids"], (1, -1))

# COMMAND ----------

# DBTITLE 1,Collate & Tokenize
from typing import Dict

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
tokenizer.pad_token = tokenizer.eos_token

def tokenize(row, tokenizer, eos = True) -> Dict:
  tokenized = tokenizer(row, return_tensors="pt", padding=True, truncation=True)
  eos = torch.reshape(torch.tensor([tokenizer.eos_token_id] * len(tokenized["input_ids"])), (-1, 1))
  tokenized["input_ids"] = torch.cat((tokenized["input_ids"], eos), dim=1)
  return tokenized

def get_encodings(df, tokenizer, label = "context"):
  
  encodings = []
  labels = []
  for _, row in df.iterrows():
    tokenized_inputs = tokenize(row.drop(label), tokenizer)
    tokenized_label = tokenize(row[label], tokenizer)
    encodings.append(tokenized_inputs)
    labels.append(tokenized_label)
  
  return encodings, labels

# COMMAND ----------

from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(output_dir="/dbfs/tmp/ubuntu/test_trainer")

# COMMAND ----------

from evaluate import load

def compute_metrics(eval_pred):
    perplexity = load("perplexity", module_type="metric")
    return perplexity.compute(predictions=predictions, model_id='gpt2')

# COMMAND ----------

from transformers import AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")

df = spark.sql("select * from ubuntu_contextualized").toPandas()
df_train, df_test = train_test_split(df)

training_encodings, training_labels = get_encodings(df_train, tokenizer = tokenizer)
testing_encodings, testing_labels = get_encodings(df_test, tokenizer = tokenizer)
train_df = ConversationDataset(encodings = training_encodings, labels = training_labels)
test_df = ConversationDataset(encodings = testing_encodings, labels = testing_labels)

model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")

trainer = Trainer(
    model = model,
    args=training_args,
    train_dataset=train_df,
    eval_dataset=test_df,
    compute_metrics=compute_metrics,
)

# COMMAND ----------

df.shape

# COMMAND ----------

len(training_encodings)

# COMMAND ----------

import torch

trainer.train()

# COMMAND ----------


