import os

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import pandas as pd
import pytest

from src.common import PROCESSED


@pytest.fixture(scope="session")
def accounts():
    return pd.read_parquet(PROCESSED / "accounts.parquet")
