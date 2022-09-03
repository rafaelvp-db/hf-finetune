import urllib

from persuasion4good.common import Task
import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql.types import (
    StructType,
    StructField,
    IntegerType,
    StringType
)

class GetDataTask(Task):
    def _get_data(self) -> DataFrame:
        target_dir = self.conf["target_dir"]
        csv_url = self.conf["csv_url"]
        self.logger.info(f"Fetching CSV data from {csv_url}...")

        try:
            urllib.request.urlretrieve(
                csv_url,
                target_dir
            )

            schema = StructType([ \
                StructField("id", IntegerType(),True), \
                StructField("utterance", StringType(),True), \
                StructField("turn", IntegerType(),True), \
                StructField("agent", IntegerType(),True), \
                StructField("conversation_id", StringType(),True), \
            ])

            dbfs_dir = target_dir.replace("/dbfs/", "/")
            df = self.spark.read.format("csv").load(
                dbfs_dir,
                sep = ",",
                header = True,
                schema = schema
            )

            return df

        except Exception as e:
            self.logger.error(f"Error downloading & loading data from {csv_url}:")
            self.logger.error(e)
        

    def _clean_data(
        self,
        df: DataFrame
    ) -> DataFrame:
        
        db_name = self.conf["db_name"]
        table_name = self.conf["table_name"]
        df = df.filter("agent is not NULL AND turn is NOT NULL")
        self.spark.sql(f"CREATE DATABASE IF NOT EXISTS {db_name}")
        df.write.saveAsTable(f"{db_name}.{table_name}", mode = "overwrite")

    def launch(self):
        self.logger.info("Launching sample ETL task")
        df = self._get_data()
        df = self._clean_data(df)
        self.logger.info("Sample ETL task finished!")


# if you're using python_wheel_task, you'll need the entrypoint function to be used in setup.py
def entrypoint():  # pragma: no cover
    task = GetDataTask()
    task.launch()


# if you're using spark_python_task, you'll need the __main__ block to start the code execution
if __name__ == "__main__":
    entrypoint()
