# Databricks notebook source
# MAGIC %sql
# MAGIC 
# MAGIC select * from ubuntu order by date_time asc

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window

ubuntu_df = spark.sql("select * from ubuntu")
w = Window.orderBy(F.col("person2"), F.col("date_time"))
ubuntu_lagged_df = ubuntu_df.withColumn("last1", F.lag(F.col("text"), 1).over(w))
display(ubuntu_lagged_df)

# COMMAND ----------



def clean(df):
  df = df.withColumn("context", F.lower(F.col("context")))
  return df

def create_contextualized_df(df, context_length = 5):

  w = Window.orderBy(F.col("date_time"))
  contextualized_df = df
  for i in range(1, context_length + 1):
      contextualized_df = contextualized_df.withColumn("context/{}".format(i), F.lag(F.col("context"), i).over(w))
  contextualized_df = contextualized_df.dropna(subset = [col for col in contextualized_df.columns if "person1" not in col])
  #contextualized_df = contextualized_df.filter("person2 != 'null'")
  #contextualized_df = contextualized_df.drop("person1", "person2", "date_time")
  return contextualized_df

# COMMAND ----------

df = spark.sql("select * from ubuntu").withColumnRenamed("text", "context")
df = clean(df)
context_df = create_contextualized_df(df)
context_df.count()

# COMMAND ----------

display(context_df)

# COMMAND ----------

context_df.write.saveAsTable("ubuntu_contextualized")
