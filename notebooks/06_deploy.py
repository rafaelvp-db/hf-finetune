# Databricks notebook source
import mlflow
from mlflow.tracking import MlflowClient
import torch
import os

client = MlflowClient()

path = '/dbfs/FileStore/Users/rafael.pierre@databricks.com/persuasion4good/'

# COMMAND ----------

os.makedirs(path, exist_ok=True)

client.download_artifacts(
  run_id = "62ef2b8df9b1447db0d7c5d8a2c6bab9",
  path = "checkpoint-7500",
  dst_path = path
)

# COMMAND ----------

import torch
from transformers import (
  AutoModelForCausalLM,
  AutoTokenizer,
  PretrainedConfig
)

import os
import mlflow
from chatbot_wrapper import ChatbotWrapper

tokenizer_name = path
model_name = path
target_dir = "persuasion4good"

def save_model(
    model_name_or_path: str,
    tokenizer_name_or_path: str,
    model_name: str = "persuasion4good",
    artifact_path = target_dir
):
    model_path = artifact_path
    tokenizer_path = artifact_path

    artifacts = {
      "hf_model_path": model_name_or_path,
      "hf_tokenizer_path": tokenizer_name_or_path
    }
    mlflow_pyfunc_model_path = model_name

    model_info = None
    
    with mlflow.start_run() as run:
        model_info = mlflow.pyfunc.log_model(
            artifact_path = mlflow_pyfunc_model_path,
            python_model = ChatbotWrapper(artifact_path = "checkpoint-7500/artifacts/checkpoint-7500"),
            code_path = ["./chatbot_wrapper.py"],
            artifacts=artifacts,
            pip_requirements=[
              "numpy==1.20.1",
              "transformers==4.24.0",
              "torch==1.10.2"
            ]
        )
        
    return model_info

# COMMAND ----------

# DBTITLE 1,Save PyFunc Wrapped Model
model_info = save_model(
  model_name_or_path = model_name,
  tokenizer_name_or_path = tokenizer_name,
  model_name = "persuasion4good"
)

# COMMAND ----------

# DBTITLE 1,Register PyFunc Model and Promote to Staging
model_version_info = mlflow.register_model(model_uri = model_info.model_uri, name = "persuasion4good")

# COMMAND ----------

client.transition_model_version_stage(
  name = "persuasion4good",
  version = model_version_info.version,
  stage = "Staging"
)

# COMMAND ----------


