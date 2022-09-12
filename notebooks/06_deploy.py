# Databricks notebook source
from transformers.pytorch_utils import *
import mlflow
from mlflow.tracking import MlflowClient
import torch

client = MlflowClient()

!mkdir /tmp/model

client.download_artifacts(
  run_id = "4b2c833519dd481086f4e7b6d291d0e3",
  path = "checkpoint-8000",
  dst_path = "/tmp/model/"
)

# COMMAND ----------

import torch
from transformers import (
  AutoModelForCausalLM,
  AutoTokenizer,
  PretrainedConfig
)

target_dir = '/tmp/model/checkpoint-8000/artifacts/checkpoint-8000'
tokenizer_name = target_dir
model_name = target_dir

import os
import mlflow
from chatbot_wrapper import ChatbotWrapper

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
            python_model = ChatbotWrapper(),
            code_path = ["./chatbot_wrapper.py"],
            artifacts=artifacts,
            pip_requirements=[
              "numpy==1.20.1",
              "transformers==4.16.2",
              "torch==1.10.2"
            ]
        )
        
    return model_info

# COMMAND ----------

# DBTITLE 1,Save PyFunc Wrapped Model
artifacts = {
  "hf_model_path": model_name,
  "hf_tokenizer_path": tokenizer_name
}

mlflow_pyfunc_model_path = "model"
model_info = None

with mlflow.start_run() as run:
    model_info = mlflow.pyfunc.log_model(
        artifact_path = mlflow_pyfunc_model_path,
        python_model = ChatbotWrapper(),
        code_path = ["./chatbot_wrapper.py"],
        artifacts=artifacts,
        pip_requirements=["numpy==1.20.1", "transformers==4.16.2", "torch==1.10.2"]
    )

# COMMAND ----------

model_info

# COMMAND ----------


