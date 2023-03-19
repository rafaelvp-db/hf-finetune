# Databricks notebook source
!pip install --upgrade pip && pip install --upgrade transformers wheel huggingface_hub evaluate pyarrow==10.0.1 datasets

# COMMAND ----------

import logging
logger = spark._jvm.org.apache.log4j
logging.getLogger("py4j.java_gateway").setLevel(logging.ERROR)

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Reading our data

# COMMAND ----------

# DBTITLE 1,Taking out columns that won't be used
from pyspark.sql import functions as F

df_train = spark.sql("select * from persuasiondb.train")
#df_train = spark.sql("select * from persuasiondb.train timestamp as of '2023-01-26 15:00'")
#df_train.write.saveAsTable("persuasiondb.train", mode = "overwrite", mergeSchema = True)

df_final = df_train \
  .drop(
    "conversation_id",
  ) \
  .withColumnRenamed(
    "context_trimmed",
    "context"
  )

df_final \
  .write \
  .option("overwriteSchema", "true") \
  .saveAsTable(
    "persuasiondb.context_final",
    mode = "overwrite",
    mergeSchema = True
  )

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Defining our Custom Dataset

# COMMAND ----------

import torch
from torch.utils.data import Dataset, DataLoader
from pyspark.storagelevel import StorageLevel
from transformers import AutoTokenizer
from typing import List
import pandas as pd
from abc import ABC, abstractmethod

class Deltaset(Dataset):
    def __init__(
      self,
      database,
      table,
      idx_col = "id",
      input_col = "text",
      label_col = "label",
      storage_level = StorageLevel.MEMORY_ONLY
    ):
      super().__init__()
      self.database = database
      self.table = table
      self.idx_col = idx_col
      self.input_col = input_col
      self.label_col = label_col
      self.storage_level = storage_level

    def load(self):
      query = f"SELECT {input_col} FROM {database}.{table}"
      self.dataframe = spark.sql(query)
      self.dataframe = self.dataframe.persist(self.storage_level)
      self.dataframe.count()

    @abstractmethod
    def sort(
      self,
      column,
      ascending = True
    ):
      

    def shuffle(
      self,

    )

    def replace(
      self,
    )

class PersuasionDataset(Deltaset):
    def __init__(
      self,
      database,
      table,
      input_col = "context",
      label_col = "label",
      idx_col = "id",
      storage_level = StorageLevel.MEMORY_ONLY
    ):
      super().__init__()
      self.input_col = input_col
      self.label_col = label_col
      self.idx_col = idx_col
      self.table = table
      self.from_eos_token = from_eos_token
      self.to_eos_token = to_eos_token

      query = f"""
        SELECT `{idx_col}`, `{input_col}`, `{label_col}`
        FROM {database}.{table}
      """
      self.df = spark.sql(query)
      self.df.persist(storage_level)
      self.length = self.df.count()
        
    def __len__(self):
      return self.length

    def __getitem__(self, idx):

      filter_query = f"{self.idx_col} = {idx}"
      df = self.df \
        .select(self.idx_col, self.input_col, self.label_col) \
        .filter(filter_query) \
        .withColumn(
          self.input_col,
          F.replace(
            F.col(self.input_col),
            self.from_eos_token,
            self.to_eos_token
          )
        ) \
        .toPandas()

      if len(df) == 0:
        return None

      input_string = df.loc[0, "context"].replace(self.eos_token, self.tokenizer.eos_token)
      context = self.tokenizer.eos_token + input_string
      label = df.loc[0, "label"]

      inputs = self.tokenizer(
        context,
        None,
        add_special_tokens = True,
        return_token_type_ids = True,
        truncation = True,
        max_length = self.max_length_input,
        padding = "max_length",
        return_tensors = "pt"
      )

      input_ids = inputs["input_ids"]
      attention_mask = inputs["attention_mask"]
      output = {
        "input_ids": input_ids,
        "attention_mask": attention_mask
      }
      
      encoded_label = self.tokenizer(
        label,
        None,
        add_special_tokens=True,
        return_token_type_ids=True,
        truncation=True,
        max_length = self.max_length_label,
        padding="max_length",
        return_tensors = "pt"
      )

      label_ids = encoded_label["input_ids"]
      output["labels"] = label_ids
        
      return output

# COMMAND ----------

# DBTITLE 1,Example: PersuasionDataset
ds = PersuasionDataset(
  database = "persuasiondb",
  table = "context_final",
)

# COMMAND ----------

ds[1]

# COMMAND ----------


