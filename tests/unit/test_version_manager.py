"""Unit tests for DatasetVersionManager (`src/data/version_manager.py`)."""

from pathlib import Path
import json
import pandas as pd
import pytest

from src.data.version_manager import DatasetVersionManager
from src.utils.config_manager import ConfigManager


def test_version_manager_lifecycle(temp_workspace: Path) -> None:
    """Verify incrementing versions (`v1`, `v2`), metadata manifest generation, and immutability protection."""
    cfg = ConfigManager(config_dir=str(temp_workspace / "config"))
    base_dir = temp_workspace / "test_datasets_synthetic"
    mgr = DatasetVersionManager(base_dir=str(base_dir), config=cfg)

    # Initial version should be v1
    assert mgr.get_latest_version_number() == 0
    assert mgr.get_next_version_id() == "v1"

    df1 = pd.DataFrame({
        "incident_number": ["INC1", "INC2"],
        "category": ["Core Banking", "Payment Systems"],
        "priority": [1, 2],
        "state": [6, 6],
        "made_sla": [True, False],
        "resolution_time_hours": [2.5, 14.0]
    })

    # Save v1
    d_path1, m_path1, v_id1 = mgr.save_versioned_dataset(df1, seed=123, file_format="csv")
    assert v_id1 == "v1"
    assert d_path1.exists()
    assert m_path1.exists()
    assert mgr.get_latest_version_number() == 1

    # Verify metadata contents
    meta1 = mgr.load_metadata("v1")
    assert meta1 is not None
    assert meta1["dataset_version"] == "v1"
    assert meta1["num_rows"] == 2
    assert meta1["schema_version"] == "1.5.0"

    # Save v2
    d_path2, m_path2, v_id2 = mgr.save_versioned_dataset(df1, seed=456, file_format="parquet")
    assert v_id2 == "v2"
    assert mgr.get_latest_version_number() == 2

    # Verify history file tracking
    history = mgr.list_all_versions()
    assert len(history) == 2
    assert history[0]["version"] == "v1"
    assert history[1]["version"] == "v2"

    # Verify immutability violation raise
    with pytest.raises(FileExistsError):
        mgr.save_versioned_dataset(df1, custom_version="v1")
