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

        target_columns = ["context", "label"]
        if sorted(list(df.columns)) != sorted(target_columns):
            error_msg = f"""
                Invalid columns in Pandas DF: {df.columns};
                Expecting: {target_columns}
            """
            raise ValueError(error_msg)

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

    def _collate(self, row, eos_string):
        collated = ""
        for i in range(0, len(row)):
            if row[i] and len(row[i]) > 0:
                collated += f"{row[i].lower()}{eos_string}"
        return collated

    def create_hf_dataset(
        self,
        df: DataFrame,
        test_size: float = 0.15,
        eos_string: str = "<EOS>"
    ) -> DatasetDict:

        df_pandas = df.toPandas()
        self.logger.info(f"Input Pandas DF for HF has {len(df_pandas)} rows")
        self.logger.info(f"Columns: {df_pandas.columns}")

        #Merge context colums into one, separated by EOS string
        df_pandas["context"] = df_pandas\
            .drop("label", axis=1)\
            .apply(
                lambda x:
                    self._collate(x, eos_string = eos_string), 
                axis=1
            )
        df_pandas["label"] = df_pandas["label"].str.lower()
        df_pandas = df_pandas.loc[:, ["label", "context"]]
        
        df_train, df_test = train_test_split(df_pandas, test_size=test_size)
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

        return dataset

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





        