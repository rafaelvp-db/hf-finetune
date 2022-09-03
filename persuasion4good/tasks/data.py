import wget

from persuasion4good.common import Task
import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql.types import (
    StructType,
    StructField,
    IntegerType,
    StringType
)
from pathlib import Path

class GetDataTask(Task):
    def _get_data(self):
        target_dir = self.conf["target_dir"]
        csv_url = self.conf["csv_url"]
        self.logger.info(f"Fetching CSV data from {csv_url} into {target_dir}...")
        wget.download(csv_url, target_dir)

        if Path(target_dir).exists():
            self.logger.info(f"Successfully downloaded data!")
        else:
            raise FileNotFoundError("Error downloading data")
    
    def _parse_df(self) -> DataFrame:

        schema = StructType([ \
            StructField("id", IntegerType(),True), \
            StructField("utterance", StringType(),True), \
            StructField("turn", IntegerType(),True), \
            StructField("agent", IntegerType(),True), \
            StructField("conversation_id", StringType(),True), \
        ])

        dbfs_dir = self.conf["target_dir"].replace("/dbfs/", "/")
        if Path(dbfs_dir).exists():
            df = self.spark.read.format("csv").load(
                f"{dbfs_dir}",
                sep = ",",
                header = True,
                schema = schema
            )

            #Basic cleaning
            df = df.filter("agent is not NULL AND turn is NOT NULL")
            self.spark.sql(f"CREATE DATABASE IF NOT EXISTS {self.conf['db_name']}")
            self.logger.info("Saving into Delta...")
            df.write.saveAsTable(
                f"{self.conf['db_name']}.{self.conf['table_name']}",
                mode = "overwrite"
            )

        else:
            raise FileNotFoundError(f"File {self.conf['target_dir']} not found")


    def launch(self):
        self.logger.info("Launching sample ETL task")
        df = self._get_data()
        df = self._parse_df()
        self.logger.info("Sample ETL task finished!")


# if you're using python_wheel_task, you'll need the entrypoint function to be used in setup.py
def entrypoint():  # pragma: no cover
    task = GetDataTask()
    task.launch()


# if you're using spark_python_task, you'll need the __main__ block to start the code execution
if __name__ == "__main__":
    entrypoint()
