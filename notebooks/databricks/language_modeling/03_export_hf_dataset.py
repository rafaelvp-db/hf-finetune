# Databricks notebook source
!pip install --upgrade pip && pip install --upgrade transformers && pip install --upgrade wheel && pip install pyspark==3.3.0 huggingface_hub==0.7.0 evaluate==0.1.2 pyarrow
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
df_final = spark.sql("select * from persuasiondb.dialog_contextualized") \
  .drop(
    "conversation_id",
    "id",
    "turn",
    "agent"
  )

# COMMAND ----------

df_pandas = df_final.toPandas().fillna("<EOS>")
df_pandas["context"] = df_pandas.drop("label", axis=1).agg("<EOS>".join, axis=1)
df_pandas = df_pandas.loc[:, ["label", "context"]]
df_pandas.head()

# COMMAND ----------



# COMMAND ----------

from datasets import Dataset, DatasetDict
from datasets.splits import NamedSplit
from sklearn.model_selection import train_test_split
import pyarrow as pa

#Clean existing data
TARGET_DIR = "/tmp/persuasion4good"
!rm -rf {TARGET_DIR}

#Create training and testing set
df_train, df_test = train_test_split(df_pandas, test_size=0.15)

def convert_to_arrow(df: pd.DataFrame, preserve_index = False):

  schema = pa.schema([
    pa.field('label', pa.string()),
    pa.field('context', pa.string())],
    metadata={
      "context": "Conversation history.",
      "label": "Agent's response."
    }
  )
  table = pa.Table.from_pandas(df, preserve_index = preserve_index, schema = schema)
  return table

train_table = convert_to_arrow(df_train)
test_table = convert_to_arrow(df_test)
print(train_table.schema)

dataset = DatasetDict({
  "train": Dataset(
    arrow_table = train_table, 
    split = NamedSplit("train")
  ),
  "test": Dataset(
    arrow_table = test_table,
    split = NamedSplit("test")
  )
})

dataset

# COMMAND ----------

dataset.save_to_disk("/tmp/persuasion4good/dataset")
