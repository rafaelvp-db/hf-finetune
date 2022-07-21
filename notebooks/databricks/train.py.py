# Databricks notebook source
!pip install transformers && pip install datasets

# COMMAND ----------

from datasets import load_dataset

dataset = load_dataset("yelp_review_full")
dataset["train"][100]

# COMMAND ----------

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_datasets = dataset.map(tokenize_function, batched=True)

# COMMAND ----------

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

# COMMAND ----------

from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=5)

# COMMAND ----------

from transformers import TrainingArguments

training_args = TrainingArguments(output_dir="test_trainer")

# COMMAND ----------

import numpy as np
from datasets import load_metric

metric = load_metric("accuracy")

# COMMAND ----------

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# COMMAND ----------

from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")

# COMMAND ----------

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)

# COMMAND ----------

trainer.train()

# COMMAND ----------

review = {"text": "I think this place is not that nice"}

tokenized_text = tokenizer(review["text"], padding="max_length", truncation=True, return_tensors='pt')
output = trainer.model(**tokenized_text)
output

# COMMAND ----------

review = {"text": "This place is terrible!"}

tokenized_text = tokenizer(review["text"], padding="max_length", truncation=True, return_tensors='pt')
output = trainer.model(**tokenized_text)
output

# COMMAND ----------

review = {"text": "This place is the best!"}

tokenized_text = tokenizer(review["text"], padding="max_length", truncation=True, return_tensors='pt')
output = trainer.model(**tokenized_text)
output

# COMMAND ----------


