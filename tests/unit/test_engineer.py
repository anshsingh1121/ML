"""Unit tests for Enterprise Feature Engineering Engine (`src/preprocessing/engineer.py`)."""

from pathlib import Path
import json
import pandas as pd
import pytest

from src.preprocessing.engineer import FeatureEngineeringEngine
from src.data.feature_registry import FeatureRegistry
from src.data.feature_lineage import FeatureLineageTracker


@pytest.fixture
def clean_incident_df() -> pd.DataFrame:
    """Create synthetic clean dataframe for feature engineering testing."""
    return pd.DataFrame({
        "number": [f"INC00000{i}" for i in range(1, 11)],
        "opened_at": [
            "2025-01-10 08:30:00",  # Friday business hours
            "2025-01-11 14:00:00",  # Saturday weekend
            "2025-01-15 02:00:00",  # Wednesday night off hours
            "2025-07-04 11:00:00",  # July 4th Holiday
            "2025-12-25 16:00:00",  # Christmas Holiday
        ] * 2,
        "resolved_at": [
            "2025-01-10 10:30:00",  # 2.0 hours
            "2025-01-11 18:00:00",  # 4.0 hours
            "2025-01-15 08:00:00",  # 6.0 hours
            "2025-07-04 12:00:00",  # 1.0 hours
            "2025-12-26 16:00:00",  # 24.0 hours
        ] * 2,
        "short_description": ["Router down and switch dead"] * 5 + ["Oracle database login failure"] * 5,
        "description": ["Router port gigabit 0/1 down after power cut"] * 5 + ["ORA-00001 unique constraint violated on login session"] * 5,
        "priority": [1, 2, 3, 4, 2] * 2,
        "business_impact": [1, 2, 2, 3, 1] * 2,
        "urgency": [1, 2, 2, 3, 2] * 2,
        "category": ["Network", "Network", "Database", "Software", "Hardware"] * 2,
        "assignment_group": ["Network Support", "Network Support", "Database Support", "App Support", "Hardware Support"] * 2,
        "duplicate_incident": [None, "INC000000", None, None, None] * 2,
        "parent_incident": ["INC999999", None, None, None, None] * 2,
        "problem_flag": [True, False, False, True, False] * 2,
        "knowledge_linked": [True, False, True, False, False] * 2
    })


def test_engineer_initialization() -> None:
    """Test initializing FeatureEngineeringEngine with singletons."""
    engine = FeatureEngineeringEngine()
    assert engine.registry is not None
    assert engine.lineage is not None
    assert engine.holidays is not None


def test_engineer_features_pipeline(clean_incident_df: pd.DataFrame, tmp_path: Path) -> None:
    """Test full feature generation across datetime, interaction, text, flags, and frequency encodings."""
    # Reset singletons before test to verify clean sync
    registry = FeatureRegistry.get_instance()
    lineage = FeatureLineageTracker.get_instance()
    engine = FeatureEngineeringEngine(registry=registry, lineage=lineage)

    output_dir = str(tmp_path / "reports")
    eng_df, report = engine.engineer_features(df=clean_incident_df, output_dir=output_dir)

    # 1. Verify new features added to DataFrame
    assert "opened_at_hour" in eng_df.columns
    assert "opened_at_hour_sin" in eng_df.columns
    assert "is_weekend" in eng_df.columns
    assert "is_business_hours" in eng_df.columns
    assert "is_holiday" in eng_df.columns
    assert "priority_x_business_impact" in eng_df.columns
    assert "resolution_time_hours" in eng_df.columns
    assert "short_description_word_count" in eng_df.columns
    assert "is_duplicate" in eng_df.columns
    assert "has_parent_incident" in eng_df.columns
    assert "assignment_group_freq" in eng_df.columns

    # 2. Verify specific feature calculations
    assert eng_df.loc[0, "opened_at_hour"] == 8
    assert eng_df.loc[0, "is_business_hours"] == 1
    assert eng_df.loc[1, "is_weekend"] == 1  # Saturday
    assert eng_df.loc[3, "is_holiday"] == 1  # July 4th
    assert eng_df.loc[4, "is_holiday"] == 1  # Christmas
    assert eng_df.loc[0, "priority_x_business_impact"] == 1
    assert eng_df.loc[0, "resolution_time_hours"] == 2.0
    assert eng_df.loc[0, "short_description_word_count"] == 5

    # 3. Verify Registry automatic sync
    hour_def = registry.get_feature("opened_at_hour_sin")
    assert hour_def is not None
    assert hour_def.data_type == "float"
    assert hour_def.encoding_strategy == "sine_cosine"

    res_def = registry.get_feature("resolution_time_hours")
    assert res_def is not None
    assert res_def.target_leakage_classification == "blocked"

    # 4. Verify Feature Lineage automatic sync
    ancestry = lineage.get_ancestry_chain("priority_x_business_impact")
    assert "priority" in ancestry
    assert "business_impact" in ancestry

    # 5. Verify exported reports
    assert (Path(output_dir) / "feature_engineering_report.json").exists()
    assert (Path(output_dir) / "feature_engineering_report.md").exists()

    with open(Path(output_dir) / "feature_engineering_report.json", "r", encoding="utf-8") as f:
        data = json.load(f)
        assert data["status"] == "CERTIFIED_ENGINEERED"
        assert data["total_new_features"] > 15
