from persuasion4good.tasks.export import ExportDatasetTask
from persuasion4good.tasks.features import FeatureEngineeringTask
from pyspark.sql import SparkSession
from conftest import spark
from fixtures import (
    feature_engineering_task,
    export_dataset_task,
    get_data_task,
    config,
    csv_rows
)
from typing import Dict
from pathlib import Path
import shutil


def test_context(
    feature_engineering_task,
    spark,
    config
):

    df = feature_engineering_task._filter(
        max_num_turns = 2
    )

    assert df.count() == 4

    feature_engineering_task._create_context(df, context_length = 2)
    context_df = spark.sql(
        f"select * from {config['db_name']}.{config['context_table']}"
    )

    assert context_df.count() > 0


def test_launch_feature(feature_engineering_task: FeatureEngineeringTask):
    feature_engineering_task.launch()


def test_export(spark, config):
    task = ExportDatasetTask(
        spark = spark,
        init_conf = config
    )

    df = task._filter_columns(context_size = 2)
    task.create_hf_dataset(df, test_size = 0.5)

    pass


