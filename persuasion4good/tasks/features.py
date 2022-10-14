from persuasion4good.common import Task
from pyspark.sql import functions as F
from pyspark.sql import window as W


class FeatureEngineeringTask(Task):

    def _filter(
        self,
        max_num_turns = 20
    ):
        query = f"""
            select conversation_id from
                (
                    select
                        count(1) as num_turns,
                        conversation_id
                    from {self.conf['db_name']}.{self.conf['raw_table']}
                    group by conversation_id
                )
                where num_turns <= {max_num_turns}
        """

        df_conversations = self.spark.sql(
            f"select * from {self.conf['db_name']}.{self.conf['raw_table']}"
        )

        df_filtered = self.spark.sql(query)\
            .join(df_conversations, ["conversation_id"]
        )

        return df_filtered

    def _create_context(
        self,
        df,
        context_length = 5
    ):

        df_exploded = df.drop("turn").withColumn(
            "shortened_utterance",
            F.explode(
                F.split(F.col("utterance"), "[\.|\?|\!\,]")
            )
        ).filter("length(shortened_utterance) > 0")

        df_exploded = df_exploded.withColumn(
            "turn",
            F.row_number().over(
                W.Window.orderBy('conversation_id', 'id')
            )
        ).drop("utterance", "id")

        df_context = df_exploded.withColumnRenamed("shortened_utterance", "label")
        window  = W.Window.partitionBy("conversation_id").orderBy(F.col("turn").desc())

        for i in range(1, context_length + 1):
            df_context = df_context.withColumn(
                f"context/{i}",
                F.lead(F.lower(F.col("label")), i)\
                    .over(window)
            )

        df_context = df_context\
            .where("agent = 0")\
            .orderBy(
                F.col("conversation_id"),
                F.col("id").desc()
            )

        df_context.write.saveAsTable(
            f"{self.conf['db_name']}.{self.conf['context_table']}",
            mode = "overwrite"
        )

    def launch(self):
        self.logger.info("Feature Engineering task started!")
        df = self._filter()
        self._create_context(df)
        self.logger.info("Feature Engineering task finished!")


# if you're using python_wheel_task, you'll need the entrypoint function to be used in setup.py
def entrypoint():  # pragma: no cover
    task = FeatureEngineeringTask()
    task.launch()


# if you're using spark_python_task, you'll need the __main__ block to start the code execution
if __name__ == "__main__":
    entrypoint()
