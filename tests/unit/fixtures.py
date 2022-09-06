import pytest
from typing import Dict
from persuasion4good.tasks.data import GetDataTask
from persuasion4good.tasks.export import ExportDatasetTask
from persuasion4good.tasks.features import FeatureEngineeringTask
from conftest import spark


@pytest.fixture
def config() -> Dict:
    common_config = {
        "db_name": "persuasion4good",
        "raw_table": "full_dialog",
        "context_table": "context_dialog",
        "target_dir": "/tmp/persuasion_full.csv",
        "csv_url": "https://www.stats.govt.nz/assets/Uploads/Business-employment-data/Business-employment-data-March-2022-quarter/Download-data/business-employment-data-march-2022-quarter-csv.zip",
        "dataset_dir": "/tmp/hf/",
        "test_size": 0.5
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
def export_dataset_task(config, spark):
    
    task = ExportDatasetTask(
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
        "0,Good morning. How are you doing today?,0,0,20180904-045349_123_live",
        "0,Good morning. How are you doing today?,0,0,20180904-045349_456_live",
        "1,Good morning. Good.,0,0,20180904-045349_715_live",
        "2,Good morning. How are you doing today?,0,0,20180904-045349_717_live",
        "3,Good morning. Good.,0,0,20180904-045349_717_live",
        "4,,,,20180904-045349_715_live",
        "5,Good morning. How are you doing today?,0,0,20180904-045349_716_live",
        "6,Good morning. How are you doing today?,0,0,20180904-045349_716_live",
        "7,Good morning. How are you doing today?,0,0,20180904-045349_716_live"
    ]

    return csv_rows
