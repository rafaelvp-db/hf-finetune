import torch
from torch.utils.data import Dataset, DataLoader
from pyspark.storagelevel import StorageLevel
from pyspark.sql import SparkSession
from transformers import AutoTokenizer
from typing import List
import pandas as pd

class PersuasionDataset(Dataset):
    def __init__(
      self,
      database,
      table,
      spark = None,
      input_col = "context",
      label_col = "label",
      idx_col = "id",
      eos_token = "<EOT>",
      pad_token = "<EOT>",
      storage_level = StorageLevel.MEMORY_ONLY,
      tokenizer_name_or_path = "microsoft/DialoGPT-Medium",
      max_length_input = 1024,
      max_length_label = 64
    ):
      super().__init__()
      self.spark = spark
      self._get_spark_session()
      self.input_col = input_col
      self.label_col = label_col
      self.tokenizer_name_or_path = tokenizer_name_or_path
      self.eos_token = eos_token
      self.pad_token = pad_token
      self.idx_col = idx_col
      self.table = table
      self.max_length_input = max_length_input
      self.max_length_label = max_length_label

      self._config_tokenizer()

      query = f"""
        SELECT `{idx_col}`, `{input_col}`, `{label_col}`
        FROM {database}.{table}
      """
      self.df = self.spark.sql(query)
      self.df.persist(storage_level)
      self.length = self.df.count()

    def _get_spark_session(self):
      
      if self.spark is None:
        self.spark = SparkSession.builder.getOrCreate()
      
    def _config_tokenizer(self):
      self.tokenizer = AutoTokenizer.from_pretrained(
        self.tokenizer_name_or_path,
        return_special_tokens_mask = True,
        padding_side = "left",
      )

      self.tokenizer.pad_token = self.tokenizer.eos_token
        
    def __len__(self):
      return self.length

    def __getitem__(self, idx):

      filter_query = f"{self.idx_col} = {idx}"
      df = self.df \
        .select(self.idx_col, self.input_col, self.label_col) \
        .filter(filter_query) \
        .toPandas()

      if len(df) == 0:
        return None

      input_string = df.loc[0, "context"].replace(self.eos_token, self.tokenizer.eos_token)
      context = self.tokenizer.eos_token + input_string
      label = df.loc[0, "label"]

      inputs = self.tokenizer(
        context,
        None,
        add_special_tokens=True,
        return_token_type_ids=True,
        truncation=True,
        max_length=self.max_length_input,
        padding="max_length",
        return_tensors = "pt"
      )

      input_ids = inputs["input_ids"]
      attention_mask = inputs["attention_mask"]
      output = {
        "input_ids": input_ids[0],
        "attention_mask": attention_mask[0]
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

      label_ids = encoded_label["input_ids"][0]
      output["labels"] = label_ids[0]
        
      return output