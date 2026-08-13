"""Unit tests for Enterprise Dataset Splitter (`src/preprocessing/splitter.py`)."""

from pathlib import Path
import json
import pandas as pd
import pytest

from src.preprocessing.splitter import DatasetSplitter


@pytest.fixture
def engineered_incident_df() -> pd.DataFrame:
    """Create synthetic engineered dataframe for split testing."""
    return pd.DataFrame({
        "number": [f"INC0000{i:02d}" for i in range(1, 31)],
        "opened_at": [f"2025-01-01 {i%24:02d}:00:00" for i in range(1, 31)],
        "priority": [1, 2, 3, 4, 5, 2] * 5,
        "assignment_group": ["Network Support"] * 10 + ["Database Support"] * 10 + ["App Support"] * 10,
        "opened_at_hour": [i % 24 for i in range(1, 31)],
        "is_weekend": [0, 1] * 15
    })


def test_splitter_initialization() -> None:
    """Test initialization."""
    splitter = DatasetSplitter()
    assert splitter.random_state == 42
    assert splitter.registry is not None


def test_splitter_stratified_split(engineered_incident_df: pd.DataFrame, tmp_path: Path) -> None:
    """Test stratified splitting, zero leakage verification, and CSV exports."""
    splitter = DatasetSplitter()
    output_dir = str(tmp_path / "processed")
    report_dir = str(tmp_path / "reports")

    train_df, val_df, test_df, report = splitter.split_dataset(
        df=engineered_incident_df,
        strategy="stratified",
        target_column="assignment_group",
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        output_dir=output_dir,
        report_dir=report_dir
    )

    # Verify counts (30 total -> 21 train, ~4 val, ~5 test)
    assert len(train_df) + len(val_df) + len(test_df) == 30
    assert len(train_df) == int(30 * 0.70)

    # Verify zero leakage status
    assert report["leakage_verification"]["status"] == "PASS_ZERO_LEAKAGE"

    # Verify files saved
    assert (Path(output_dir) / "train.csv").exists()
    assert (Path(output_dir) / "val.csv").exists()
    assert (Path(output_dir) / "test.csv").exists()
    assert (Path(output_dir) / "metadata.json").exists()
    assert (Path(report_dir) / "split_report.json").exists()
    assert (Path(report_dir) / "split_report.md").exists()


def test_splitter_time_based_split(engineered_incident_df: pd.DataFrame, tmp_path: Path) -> None:
    """Test time-based splitting."""
    splitter = DatasetSplitter()
    output_dir = str(tmp_path / "processed_time")
    report_dir = str(tmp_path / "reports_time")

    train_df, val_df, test_df, report = splitter.split_dataset(
        df=engineered_incident_df,
        strategy="time_based",
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        output_dir=output_dir,
        report_dir=report_dir
    )

    # Verify chronological order across splits
    assert pd.to_datetime(train_df["opened_at"]).max() <= pd.to_datetime(val_df["opened_at"]).min()
    assert pd.to_datetime(val_df["opened_at"]).max() <= pd.to_datetime(test_df["opened_at"]).min()
    assert report["leakage_verification"]["status"] == "PASS_ZERO_LEAKAGE"
