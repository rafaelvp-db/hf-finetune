from persuasion4good.common import Task



class FeatureEngineeringTask(Task):

    def _filter(
        self,
        max_num_turns = 20,
        output_table = "filtered_conversations"
    ):
        query = f"""
            select conversation_id from
                (
                    select
                        count(1) as num_turns,
                        conversation_id
                    from {self.conf['db_name']}.{self.conf['table_name']}
                    group by conversation_id
                )
                where num_turns <= {max_num_turns}
        """

        df_conversations = self.spark.sql(
            f"select * from {self.conf['db_name']}.{self.conf['table_name']}"
        )
        df_filtered = self.spark.sql(query)\
            .join(df_conversations, ["conversation_id"]
        )
        df_filtered.write.saveAsTable(
            f"{self.conf['db_name']}.{output_table}",
            mode = "overwrite"
        )

    def launch(self):
        self.logger.info("Feature Engineering task started!")
        df = self._filter()
        self.logger.info("Feature Engineering task finished!")


# if you're using python_wheel_task, you'll need the entrypoint function to be used in setup.py
def entrypoint():  # pragma: no cover
    task = FeatureEngineeringTask()
    task.launch()


# if you're using spark_python_task, you'll need the __main__ block to start the code execution
if __name__ == "__main__":
    entrypoint()
