import logging
import datasets
import torch
import mlflow
import torch

from transformers import (
  AutoTokenizer,
  AutoModelForCausalLM,
  DataCollatorWithPadding,
  Trainer,
  TrainingArguments,
  AutoConfig
)

TARGET_DIR = "/tmp/persuasion4good"
mlflow.set_tracking_uri("databricks")
mlflow.set_experiment(experiment_name = "/Users/rafael.pierre@databricks.com/persuasion4good")

#We will create a custom tokenization function leveraging microsoft/DialoGPT-Medium tokenizer

dataset = datasets.load_from_disk("/tmp/dataset")

model_name_or_path = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
config = AutoConfig.from_pretrained(model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(
  model_name_or_path,
  config = config
)

# By default, Hugging Face has an MLflow callback which is already switched on. 
# We just need to setup some parameters.

import os

os.environ["MLFLOW_EXPERIMENT_NAME"] = "/Users/rafael.pierre@databricks.com/persuasion4good"
os.environ["MLFLOW_FLATTEN_PARAMS"] = "1"
os.environ["HF_MLFLOW_LOG_ARTIFACTS"] = "1"
os.environ["MLFLOW_NESTED_RUN"] = "1"
os.environ["WANDB_DISABLED"] = "1"

for param in model.transformer.parameters():
    param.requires_grad = False

for param in model.lm_head.parameters():
    param.requires_grad = True
    
from evaluate import load

def compute_metrics(eval_preds):
    metric = load("perplexity", module_type="metric")
    results = metric.compute(predictions=eval_preds, model_id='gpt2')
    return results

from transformers import EarlyStoppingCallback, IntervalStrategy, DataCollatorForLanguageModeling

tokenizer.pad_token = tokenizer.eos_token
collator = DataCollatorForLanguageModeling(
  tokenizer = tokenizer,
  mlm = False,
  pad_to_multiple_of = 8
)

args = TrainingArguments(
    output_dir = f"{TARGET_DIR}/trainer/",
    evaluation_strategy = IntervalStrategy.STEPS, # "steps"
    eval_steps = 100, # Evaluation and Save happens every 50 steps
    save_total_limit = 5, # Only last 5 models are saved. Older ones are deleted.
    eval_accumulation_steps = 10,
    learning_rate = 5e-4,
    weight_decay = 0.1,
    adam_epsilon = 1e-8,
    warmup_steps = 0.0,
    max_grad_norm = 1.0,
    num_train_epochs = 10000.0,
    logging_steps = 10,
    no_cuda = False,
    overwrite_output_dir = True,
    seed = 42,
    local_rank = -1,
    fp16 = True,
    metric_for_best_model = 'eval_loss',
    greater_is_better = False,
    load_best_model_at_end = True,
    disable_tqdm = False,
    prediction_loss_only = True,
    report_to = ["mlflow"],
    deepspeed="ds_config.json"
)

trainer = Trainer(
    data_collator = collator,
    compute_metrics = compute_metrics,
    model = model.cuda(),
    args = args,
    train_dataset = dataset["train"],
    eval_dataset = dataset["test"],
    tokenizer = tokenizer,
    callbacks = [EarlyStoppingCallback(
        early_stopping_patience = 5,
        early_stopping_threshold = 0.001
    )]
)

torch.cuda.empty_cache()

with mlflow.start_run(nested = True) as run:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    trainer.train()
    metrics = trainer.evaluate()
    print(metrics)
    mlflow.log_metrics(metrics)
    model = trainer.model
    model_info = mlflow.pytorch.log_model(model, artifact_path = "model")