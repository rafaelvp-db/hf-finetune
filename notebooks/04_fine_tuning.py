# Databricks notebook source
!cp cuda.sh /dbfs/FileStore/

# COMMAND ----------

!sudo find / -name 'libcudart.so.11.0'

# COMMAND ----------

!pip install --upgrade pip && pip install --upgrade cuda-python wheel huggingface_hub transformers deepspeed torch datasets

# append path to "LD_LIBRARY_PATH" in profile file
!export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.3/targets/x86_64-linux/lib

# COMMAND ----------

TARGET_DIR = "/dbfs/tmp/persuasion4good"

!rm -rf {TARGET_DIR}/trainer

# COMMAND ----------

from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
  "microsoft/DialoGPT-medium",
  return_special_tokens_mask = True,
  padding_side = "left"
)
print(f"Model max length: {tokenizer.model_max_length}")
print(f"Model max length single sentence: {tokenizer.max_len_single_sentence}")
print(f"Model max_model_input_sizes: {tokenizer.max_model_input_sizes}")

df_dataset = spark.sql("select * from persuasiondb.final_dataset")
df_pandas = df_dataset.toPandas()
df_train, df_test = train_test_split(df_pandas, test_size=0.15)

dataset = DatasetDict({
  "train": Dataset.from_pandas(df_train, split = "train"),
  "test": Dataset.from_pandas(df_test, split = "test")
})

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

dataset.save_to_disk("/tmp/dataset")

# COMMAND ----------

token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
url = spark.conf.get("spark.databricks.workspaceUrl")
dbutils.fs.put("file:///root/.databrickscfg", f"[DEFAULT]\nhost=https://{url}\ntoken = {token}", overwrite = True)

# COMMAND ----------

!deepspeed train.py

# COMMAND ----------


