import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from persuasion4good.common import Task
from datasets import Dataset, DatasetDict
from transformers import TextDataset
from datasets.splits import NamedSplit
import pyarrow as pa
from pyspark.sql import DataFrame


class ExportDatasetTask(Task):


    def _filter_columns(self, context_size = 5):

        context_size = 5
        query_filter = " AND ".join(
            [
                f"""length(
                    spark_catalog
                        .{self.conf['db_name']}
                        .{self.conf['context_table']}
                        .`context/{i}`) > 0
                """
                for i in range(1, context_size + 1)
            ]
        )
        query = f"""
            select * 
            from {self.conf['db_name']}.{self.conf['context_table']}
            where {query_filter}
        """

        df_final = self.spark.sql(query) \
            .drop(
                "conversation_id",
                "id",
                "turn",
                "agent"
            )

        return df_final

    def _create_arrow_table(
        self,
        df: pd.DataFrame,
        preserve_index: bool = False
    ):

        schema = pa.schema([
            pa.field('label', pa.string()),
            pa.field('context', pa.string())],
            metadata={
                "context": "Conversation history.",
                "label": "Agent's response."
            }
        )
        table = pa.Table.from_pandas(
            df,
            preserve_index = preserve_index,
            schema = schema
        )

        return table

    def create_hf_dataset(self, df: DataFrame, test_size: float = 0.15) -> DatasetDict:

        pandas_df = df.toPandas()
        df_train, df_test = train_test_split(pandas_df, test_size=test_size)
        train_table = self._create_arrow_table(df_train)
        test_table = self._create_arrow_table(df_test)

        dataset = DatasetDict(
            {
                "train": Dataset(
                    arrow_table = train_table, 
                    split = NamedSplit("train")
                ),
                "test": Dataset(
                    arrow_table = test_table,
                    split = NamedSplit("test")
                )
            }
        )

        dataset.save_to_disk(self.conf["dataset_dir"])

    def launch(self):
        df = self._filter_columns()
        self.create_hf_dataset(df)

# if you're using python_wheel_task, you'll need the entrypoint function to be used in setup.py
def entrypoint():  # pragma: no cover
    task = ExportDatasetTask()
    task.launch()


# if you're using spark_python_task, you'll need the __main__ block to start the code execution
if __name__ == "__main__":
    entrypoint()





        