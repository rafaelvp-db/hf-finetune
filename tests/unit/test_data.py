import pytest
from fixtures import (
    get_data_task,
    config,
    target_columns,
    csv_rows
)
from persuasion4good.tasks.data import GetDataTask
from pyspark.sql import SparkSession
from conftest import spark
from pathlib import Path
from typing import (
    Dict,
    List
)

def test_get_data(get_data_task: GetDataTask, spark: SparkSession, config: Dict):

    get_data_task._get_data()
    assert Path(config["target_dir"]).exists()


def test_parse_data(
    spark: SparkSession,
    csv_rows: List,
    config: Dict,
    get_data_task: GetDataTask,
    target_columns: List
):
    
    filepath = Path(config["target_dir"])
    filepath.write_text("\n".join(csv_rows))

    get_data_task._parse_df()

    df = spark.sql(
        f"""SELECT *
        FROM {config['db_name']}.{config['table_name']}
        """
    )

    assert sorted(target_columns) == sorted(df.columns)
    assert df.count() == 4
    

def test_launch(get_data_task: GetDataTask):

    get_data_task.launch()
