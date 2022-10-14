from persuasion4good.tasks.finetune import FinetuningTask
from conftest import spark
from fixtures.fixtures import (
    feature_engineering_task,
    export_dataset_task,
    get_data_task,
    config,
    csv_rows
)

def test_module():

    task = FinetuningTask()
    assert task


