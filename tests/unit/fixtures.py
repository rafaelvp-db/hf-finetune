import pytest
from typing import Dict
from persuasion4good.tasks.data import GetDataTask
from persuasion4good.tasks.features import FeatureEngineeringTask
from conftest import spark
from pathlib import Path
import shutil


@pytest.fixture
def config() -> Dict:
    common_config = {
        "db_name": "persuasion4good",
        "table_name": "full_dialog",
        "output_table": "filtered_dialog",
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
def feature_engineering_task(config, spark):
    
    task = FeatureEngineeringTask(
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

@pytest.fixture
def csv_rows():

    csv_rows = [
        ",Unit,Turn,B4,B2",
        "0,Good morning. How are you doing today?,0,0,20180904-045349_715_live",
        "1,,,,20180904-045349_715_live",
        "2,Good morning. How are you doing today?,0,0,20180904-045349_716_live",
        "3,Good morning. How are you doing today?,0,0,20180904-045349_716_live",
        "4,Good morning. How are you doing today?,0,0,20180904-045349_716_live"
    ]

    return csv_rows
