# Databricks notebook source
!pip install --upgrade pip && pip install --upgrade transformers && pip install --upgrade wheel && huggingface_hub==0.7.0 evaluate==0.1.2 pyarrow
!pip install --upgrade datasets

# COMMAND ----------

import logging
logger = spark._jvm.org.apache.log4j
logging.getLogger("py4j.java_gateway").setLevel(logging.ERROR)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Dict

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Reading our data

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC select * from persuasiondb.dialog_contextualized

# COMMAND ----------

# DBTITLE 1,Taking out columns that won't be used
from pyspark.sql import functions as F

query = f"select * from persuasiondb.dialog_contextualized"

df_final = spark.sql(query) \
  .drop(
    "conversation_id",
    "id",
    "turn",
    "agent"
  )

# COMMAND ----------

df_pandas = df_final.toPandas()
df_pandas

# COMMAND ----------

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")

def collate(row, eos_token = tokenizer.eos_token):
    collated = ""
    context = list(reversed(row))
    for i in range(0, len(row)):
        collated += f"{context[i].lower()}{eos_token}".lstrip()
    return collated
    
df_pandas["context"] = df_pandas.drop("label", axis=1).apply(lambda x: collate(x), axis=1)
df_pandas["label"] = df_pandas["label"].str.lower()
df_pandas = df_pandas.loc[:, ["label", "context"]]
df_pandas.context.values[0]

# COMMAND ----------

df = spark.createDataFrame(df_pandas)
spark.sql("drop table if exists persuasiondb.final_dataset")
df.write.saveAsTable("persuasiondb.final_dataset")

# COMMAND ----------

from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from transformers import TextDataset
from datasets.splits import NamedSplit
from sklearn.model_selection import train_test_split
import pyarrow as pa

#Clean existing data
TARGET_DIR = "/tmp/persuasion4good"
!rm -rf {TARGET_DIR}

#Create training and testing set
df_train, df_test = train_test_split(df_pandas, test_size=0.15)

dataset = DatasetDict({
  "train": Dataset.from_pandas(df_train, split = "train"),
  "test": Dataset.from_pandas(df_test, split = "test")
})

dataset

# COMMAND ----------

def tokenize(batch, tokenizer, feature, eos = True, max_length = 512, pad_to_multiple_of = 8):
    tokenizer.pad_token = tokenizer.eos_token

    input_ids = tokenizer(
        batch[feature],
        return_tensors = "pt",
        return_attention_mask = False,
        padding = "max_length",
        max_length = max_length
    )["input_ids"][0]

    result_key = "labels"
    if feature != "label":
        result_key = "input_ids"

    result = {
        f"{result_key}": input_ids
    }

    return result

dataset = dataset.map(lambda x: tokenize(x, tokenizer, max_length = 256, feature = "context"), remove_columns = ["context"])
dataset = dataset.map(lambda x: tokenize(x, tokenizer, max_length = 256, feature = "label", eos = False), remove_columns = ["label"])

# COMMAND ----------

!rm -rf /tmp/persuasion4good
dataset.save_to_disk("/tmp/persuasion4good/dataset")

# COMMAND ----------

import datasets
dataset = datasets.load_from_disk("/tmp/persuasion4good/dataset/")

# COMMAND ----------

dataset

# COMMAND ----------

!mkdir /dbfs/tmp/persuasion4good
!cp -r /tmp/persuasion4good/dataset/ /dbfs/tmp/persuasion4good/dataset
