import pytest
from typing import Dict

@pytest.fixture
def config() -> Dict:
    common_config = {
        "db_name": "persuasion4good",
        "table_name": "full_dialog"
    }

    return common_config