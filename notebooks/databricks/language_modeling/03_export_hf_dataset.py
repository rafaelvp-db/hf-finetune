# Databricks notebook source
!pip install --upgrade pip && pip install --upgrade transformers && pip install --upgrade wheel && pip install pyspark==3.3.0 huggingface_hub==0.7.0 evaluate==0.1.2 datasets pyarrow

# COMMAND ----------

import logging
logger = spark._jvm.org.apache.log4j
logging.getLogger("py4j.java_gateway").setLevel(logging.ERROR)

# COMMAND ----------

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from transformers import AutoTokenizer, DataCollatorForLanguageModeling, TextDataset
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

from datasets import Dataset, DatasetDict
from datasets.splits import NamedSplit
from sklearn.model_selection import train_test_split

df_train, df_test = train_test_split(df, test_size=0.15)

"""dataset = DatasetDict({
  "train": Dataset(
    arrow_table = train_table, 
    split = NamedSplit("train")
  ),
  "test": Dataset(
    arrow_table = test_table,
    split = NamedSplit("test")
  )
})"""

dataset = DatasetDict({
  "train": Dataset.from_pandas(df_train, split = NamedSplit("train")),
  "test": Dataset.from_pandas(df_test, split = NamedSplit("test"))
})

# COMMAND ----------

dataset.save_to_disk("/tmp/ubuntu/hf_dataset/")
