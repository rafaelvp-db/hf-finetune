# Databricks notebook source
!pip install --upgrade pip && pip install --upgrade transformers && pip install --upgrade wheel && pip install pyspark==3.3.0 huggingface_hub==0.7.0 evaluate==0.1.2 pyarrow
!pip install --upgrade datasets

# COMMAND ----------

import logging
logger = spark._jvm.org.apache.log4j
logging.getLogger("py4j.java_gateway").setLevel(logging.ERROR)

# COMMAND ----------

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


from typing import List, Tuple, Dict

# COMMAND ----------

df = spark.sql("select * from ubuntu_contextualized").toPandas()
df = df.rename(columns = {"context": "label"})

# COMMAND ----------

df["context"] = df.drop("label", axis=1).agg("<EOS>".join, axis=1)
df = df.loc[:, ["label", "context"]]

# COMMAND ----------

df

# COMMAND ----------

!rm -rf /tmp/ubuntu

# COMMAND ----------

from datasets import Dataset, DatasetDict
from datasets.splits import NamedSplit
from sklearn.model_selection import train_test_split
import pyarrow as pa

df_train, df_test = train_test_split(df, test_size=0.15)

def convert_to_arrow(df: pd.DataFrame, preserve_index = False):

  schema = pa.schema([
    pa.field('label', pa.string()),
    pa.field('context', pa.string())],
    metadata={"context": "Conversation contents."})
  table = pa.Table.from_pandas(df, preserve_index = preserve_index, schema = schema)
  return table

train_table = convert_to_arrow(df_train)
test_table = convert_to_arrow(df_test)
print(train_table.schema)

# COMMAND ----------

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

# COMMAND ----------

dataset

# COMMAND ----------

dataset.save_to_disk("/tmp/ubuntu/dataset")

# COMMAND ----------


