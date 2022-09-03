import pytest
from typing import Dict
from persuasion4good.tasks.data import GetDataTask
from conftest import spark
from pathlib import Path


@pytest.fixture
def config() -> Dict:
    common_config = {
        "db_name": "persuasion4good",
        "table_name": "full_dialog",
        "target_dir": "/tmp/persuasion_full.csv",
        "csv_url": "https://www.stats.govt.nz/assets/Uploads/Business-employment-data/Business-employment-data-March-2022-quarter/Download-data/business-employment-data-march-2022-quarter-csv.zip"
    }

    return common_config


@pytest.fixture
def get_data_task(config, spark):

    task = GetDataTask(
        spark = spark,
        init_conf = config
    )

    return task

@pytest.fixture
def target_columns():

    columns = [
        "id",
        "utterance", 
        "turn",
        "agent",
        "conversation_id"
    ]

    return columns

