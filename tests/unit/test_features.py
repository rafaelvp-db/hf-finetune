from persuasion4good.tasks.features import FeatureEngineeringTask
from pyspark.sql import SparkSession
from conftest import spark
from fixtures import (
    feature_engineering_task,
    get_data_task,
    config,
    csv_rows
)
from typing import Dict
from pathlib import Path
import shutil


def test_filter(
    spark: SparkSession,
    get_data_task,
    feature_engineering_task,
    config: Dict
):
    
    output_table = config['output_table']
    db_name = config["db_name"]
    feature_engineering_task._filter(
        max_num_turns = 2,
        output_table = output_table
    )
    filtered_df = spark.sql(
        f"select * from {db_name}.{output_table}"
    )
    assert filtered_df.count() == 1

def test_launch(feature_engineering_task: FeatureEngineeringTask):

    feature_engineering_task.launch()



