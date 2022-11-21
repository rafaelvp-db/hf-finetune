# Databricks notebook source
TARGET_DIR = "/dbfs/tmp/persuasion4good"
URL = "https://gitlab.com/ucdavisnlp/persuasionforgood/-/raw/master/data/FullData/full_dialog.csv?inline=false"
!rm -rf {TARGET_DIR} && \
  mkdir {TARGET_DIR} && \
  curl {URL} \
  -o {TARGET_DIR}/full_dialog.csv
!ls -all {TARGET_DIR}

# COMMAND ----------

!head {TARGET_DIR}/full_dialog.csv

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, IntegerType, StringType, DateType, TimestampType

schema = StructType([ \
    StructField("id", IntegerType(),True), \
    StructField("utterance", StringType(),True), \
    StructField("turn", IntegerType(),True), \
    StructField("agent", IntegerType(),True), \
    StructField("conversation_id", StringType(),True), \
  ])

DBFS_DIR = TARGET_DIR.replace("/dbfs/", "/")

df = spark.read.format("csv").load(
  f"{DBFS_DIR}/full_dialog.csv",
  sep = ",",
  header = True,
  schema = schema
)

display(df.take(10))

# COMMAND ----------

spark.sql("CREATE DATABASE IF NOT EXISTS persuasiondb")
df = df.filter("agent is not NULL AND turn is NOT NULL")
df.write.saveAsTable("persuasiondb.full_dialog", mode = "overwrite")

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC select * from persuasiondb.full_dialog ORDER BY conversation_id, id
