# Databricks notebook source
!pip install --upgrade pip && pip install --upgrade transformers && pip install --upgrade wheel && pip install pyspark==3.3.0 huggingface_hub==0.7.0 evaluate==0.1.2 datasets

# COMMAND ----------

import logging
logger = spark._jvm.org.apache.log4j
logging.getLogger("py4j.java_gateway").setLevel(logging.ERROR)

# COMMAND ----------

from transformers import (
  AutoTokenizer,
  AutoModelForCausalLM,
  DataCollatorForLanguageModeling,
  AutoTokenizer,
  Trainer,
  TrainingArguments
)

import datasets
import torch
  
dataset = datasets.load_from_disk("/tmp/ubuntu/hf_dataset/")
dataset

# COMMAND ----------

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
import torch

def prepare_dataset(batch, tokenizer):
  tokenizer.pad_token = tokenizer.eos_token
  result = {
    "labels": tokenizer(
      batch["context"],
      return_tensors = "pt",
      return_attention_mask = False,
      padding = True,
      truncation = True
    ),
  }
  return result
  
train_dataset_processed = dataset.map(
  lambda batch: prepare_dataset(batch, tokenizer),
  remove_columns = ["context"] + [f"context/{idx}" for idx in range(1,6)]
)

# COMMAND ----------

!rm -rf /dbfs/tmp/ubuntu/trainer/

# COMMAND ----------

model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
collator = DataCollatorForLanguageModeling(tokenizer = tokenizer, mlm = False)

args = TrainingArguments(
  output_dir = "/dbfs/tmp/ubuntu/trainer/",
  overwrite_output_dir = True,
  per_device_train_batch_size = 12
)

trainer = Trainer(
  model = model,
  args = args,
  train_dataset = dataset["train"],
  data_collator = collator,
)

# COMMAND ----------

trainer.train()

# COMMAND ----------

df_collated = df.copy()
df_collated["_collated_context"] = df.drop("context", axis=1).apply(lambda x: tokenizer.eos_token.join(x.astype(str)))


# COMMAND ----------

from torch.nn import functional as F

def collate(tensor_list) -> torch.TensorType:
  
  padded_list = []
  max_size = max([tensor.shape[1] for tensor in tensor_list])
  for tensor in tensor_list:
    diff = max_size - tensor.shape[1]
    padded_tensor = F.pad(tensor, (0, diff), value = tokenizer.eos_token_id)
    padded_list.append(padded_tensor)
  return padded_list
  

def tokenize(df, tokenizer, eos = True):
  tokenized_list = []
  df["_collated_context"] = df.apply(lambda x: tokenizer.eos_token.join(x))
    preprocessed_text = tokenizer(tokenizer.eos_token.join(row), return_attention_mask = False, return_tensors = "pt")["input_ids"]
    eos_token = torch.tensor([[tokenizer.eos_token_id]])
    tokenized = torch.cat((encoding_tensor, eos_token), dim=1)
    encoding_tensor["input_ids"]
    tokenized_list.append(tokenized)
  padded_tensor = collate(tokenized_list)
  return padded_tensor

# COMMAND ----------

tokenized_features = tokenized
tokenizer(list(df["context"].values), return_tensors = "pt", return_attention_mask = False, padding = True, truncation = True)

# COMMAND ----------

tensor_input

# COMMAND ----------

import torch

encoding_list = []
for _,row in df.drop("context", axis=1).iterrows():
  encoding_row = []
  tokenized = tokenizer(list(row.values.astype(str)), return_attention_mask = False)
  for inputs in tokenized["input_ids"]:
    encoding = inputs + [tokenizer.eos_token_id]
    encoding_row.extend(encoding)
  encoding_list.append(encoding_row)
  
torch.tensor(encoding_list)

# COMMAND ----------

list(df.values.astype(str))

# COMMAND ----------

tokenizer(list(df.values.astype(str)))

# COMMAND ----------

combined_df = df.apply(lambda row: '_'.join(row.values.astype(str)), axis=1)

def tokenize(df, columns, tokenizer):
  tensor_list = torch.tensor([])
  tokenizer.pad_token = tokenizer.eos_token
  for column in columns:
    tokenized = None
    tokenized = tokenizer(df["context/1"].values.tolist(), return_tensors="pt", padding=True, truncation=True)
    eos_token_column = torch.reshape(torch.tensor([tokenizer.eos_token_id]
    tokenized["input_ids"] = torch.cat((encoding, torch.tensor(tokenizer.eos_token_id)), dim=1) for encoding in tokenized["input_ids"]
    tensor_list.append(tokenized["input_ids"])
  return torch.tensor(tensor_list)

tokenize(df = df, columns = df.drop("context", axis=1).columns, tokenizer = tokenizer)

# COMMAND ----------



# COMMAND ----------

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
tokenizer.pad_token = tokenizer.eos_token
encodings = tokenizer(mylist, return_tensors="pt", padding=True, truncation=True, return_attention_mask = False)
torch.reshape(encodings["input_ids"], (1, -1))

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


