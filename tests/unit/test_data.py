from fixtures import config
from persuasion4good.tasks.data import GetDataTask
from pyspark.sql import SparkSession
from conftest import spark
from pathlib import Path
from typing import Dict

def test_get_data(spark: SparkSession, config: Dict):

    task = GetDataTask(
        spark = spark,
        init_conf = config
    )

    assert task
    
