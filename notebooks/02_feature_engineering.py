# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC ## Feature Engineering
# MAGIC <br/>
# MAGIC 
# MAGIC * In this notebook, we perform **feature engineering** over the dialogue data that we downloaded previously.
# MAGIC * Multi-turn conversational models require multiple columns for each of the parts of the **context**.
# MAGIC * The **context** in this case represents the different **utterances** that are part of the conversation.

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC select * from persuasiondb.full_dialog order by conversation_id, id asc

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### High Level Analysis
# MAGIC 
# MAGIC Let's look at how our dialogues are distributed in terms of number of **turns/utterances**.

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC select count(1) as num_turns
# MAGIC from persuasiondb.full_dialog
# MAGIC group by conversation_id

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC We have a long tail. Let's look into some basic summary statistics:

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC select 
# MAGIC   avg(num_turns),
# MAGIC   std(num_turns),
# MAGIC   max(num_turns),
# MAGIC   min(num_turns)
# MAGIC from (
# MAGIC   select count(1) as num_turns
# MAGIC   from persuasiondb.full_dialog
# MAGIC   group by conversation_id
# MAGIC )

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC * Each dialogue has on average 20 turns, with a standard deviation of approximately one. Our dialogues are somewhat homogeneous in terms of length.
# MAGIC * To train our model, we will have a label column which indicates the answers that we want our chatbot to give us.
# MAGIC * We will apply a rolling window function, where the label will be the last conversation turn up until that record, and the other columns will be the previous turns.
# MAGIC * For the sake of simplicity, let's work only with conversations which have 20 or less turns.

# COMMAND ----------

query = """select conversation_id from
  (select
     count(1) as num_turns,
     conversation_id
   from persuasiondb.full_dialog
   group by conversation_id
  )
where num_turns <= 20"""

df_conversations = spark.sql("select * from persuasiondb.full_dialog")
df_filtered = spark.sql(query).join(df_conversations, ["conversation_id"])
display(df_filtered)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Building Conversation Contexts

# COMMAND ----------

# Testing without shortening utterances

from pyspark.sql import window as W
from pyspark.sql import functions as F

CONTEXT_LENGTH = 10
df_context = df_filtered.withColumnRenamed("utterance", "label")
window  = W.Window.partitionBy("conversation_id").orderBy(F.col("turn"))

for i in range(1, CONTEXT_LENGTH + 1):
    df_context = df_context.withColumn(f"context/{i}", F.lag(F.lower(F.col("label")), i).over(window))

df_context = df_context.dropna()
spark.sql("drop table if exists persuasiondb.dialog_contextualized")
df_context.write.saveAsTable("persuasiondb.dialog_contextualized")
display(df_context)

# COMMAND ----------

!databricks configure --profile DEFAULT

# COMMAND ----------


