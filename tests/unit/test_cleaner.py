"""Unit tests for Enterprise Data Cleaner (`src/preprocessing/cleaner.py`)."""

from pathlib import Path
import json
import pandas as pd
import pytest

from src.preprocessing.cleaner import EnterpriseDataCleaner
from src.data.feature_registry import FeatureRegistry


@pytest.fixture
def dirty_incident_df() -> pd.DataFrame:
    """Create synthetic dirty dataset with duplicates, invalid timestamps, and outliers."""
    return pd.DataFrame({
        "incident_number": ["INC000001", "INC000001", "INC000002", "INC000003", "INC000004"],
        "opened_at": ["2025-01-10 10:00:00", "2025-01-10 12:00:00", "2025-01-10 14:00:00", "2025-01-10 16:00:00", "2025-01-10 18:00:00"],
        "resolved_at": ["2025-01-10 08:00:00", "2025-01-10 13:00:00", "2025-01-10 15:00:00", "2025-01-10 17:00:00", "2025-01-10 19:00:00"],
        "closed_at": ["2025-01-11 10:00:00", "2025-01-11 12:00:00", "2025-01-11 14:00:00", "2025-01-11 16:00:00", "2025-01-11 18:00:00"],
        "short_description": [None, "Login failure", "Network lag  ", None, "Server error"],
        "description": ["Router down", None, "High ping  ", "ORA-00001", "CPU 100%"],
        "category": ["Network", "InvalidCategory", "Database", "Software", "Hardware"],
        "assignment_group": ["Network Support", "UnknownGroup", "Database Support", "Application Support", "Hardware Support"],
        "priority": [0, 2, 6, 3, 4],  # 0 and 6 are out of bounds [1,5]
        "reassignment_count": [0, 1, 50, 2, 0],  # 50 is extreme outlier above 15
        "reopen_count": [0, 0, 0, 25, 1],  # 25 is extreme outlier above 8
        "made_sla": [True, False, "True", 0, 1]
    })


def test_cleaner_initialization() -> None:
    """Test initializing cleaner."""
    cleaner = EnterpriseDataCleaner()
    assert cleaner.registry is not None


def test_cleaner_clean_dataset(dirty_incident_df: pd.DataFrame, tmp_path: Path) -> None:
    """Test full cleaning pipeline: duplicates, timestamps, categories, outliers, and audit reports."""
    cleaner = EnterpriseDataCleaner()
    output_dir = str(tmp_path / "reports")

    clean_df, audit_log = cleaner.clean_dataset(
        df=dirty_incident_df,
        output_dir=output_dir,
        strict_mode=False
    )

    # Verify duplicate removal (INC000001 had 2 rows, should now be 1)
    assert len(clean_df) == 4
    assert clean_df["incident_number"].nunique() == 4

    # Verify missing value handling (short_description None filled with 'Not Provided')
    assert clean_df["short_description"].isna().sum() == 0
    assert "Not Provided" in clean_df["short_description"].values
    assert clean_df["description"].isna().sum() == 0

    # Verify priority bounds [1, 5]
    assert clean_df["priority"].min() >= 1
    assert clean_df["priority"].max() <= 5

    # Verify outlier winsorization (reassignment_count 50 should be clipped <= 15)
    assert clean_df["reassignment_count"].max() <= 15
    assert clean_df["reopen_count"].max() <= 8

    # Verify invalid category is preserved (as domain validation is removed)
    assert "InvalidCategory" in clean_df["category"].values

    # Verify timestamp progression corrected (resolved_at < opened_at for first row corrected)
    op = pd.to_datetime(clean_df["opened_at"])
    res = pd.to_datetime(clean_df["resolved_at"])
    assert (res >= op).all()

    # Verify audit reports exist and contain exact transformations
    assert (Path(output_dir) / "cleaning_report.json").exists()
    assert (Path(output_dir) / "cleaning_report.md").exists()

    with open(Path(output_dir) / "cleaning_report.json", "r", encoding="utf-8") as f:
        data = json.load(f)
        assert data["initial_record_count"] == 5
        assert data["final_record_count"] == 4
        assert data["records_removed_total"] == 1
        assert len(data["transformations"]) >= 4
