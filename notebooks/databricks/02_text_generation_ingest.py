# Databricks notebook source
!tar -zxvf /dbfs/tmp/ubuntu/dialogs.tgz -C /dbfs/tmp/ubuntu/

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, StringType, DateType, TimestampType

schema = StructType([ \
    StructField("date_time", TimestampType(),True), \
    StructField("person1", StringType(),True), \
    StructField("person2", StringType(),True), \
    StructField("text", StringType(),True), \
  ])

df = spark.read.format("csv").load('dbfs:/tmp/ubuntu/dialogs/', sep = "\t", header = True, schema = schema)

# COMMAND ----------

df.write.saveAsTable("ubuntu_full")

# COMMAND ----------


