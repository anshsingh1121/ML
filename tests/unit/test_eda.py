"""Unit tests for Enterprise EDA Engine (`src/preprocessing/eda.py`)."""

from pathlib import Path
import json
import pandas as pd
import pytest

from src.preprocessing.eda import EnterpriseEDAEngine
from src.data.feature_registry import FeatureRegistry


@pytest.fixture
def sample_incident_df() -> pd.DataFrame:
    """Create synthetic incident dataframe for EDA testing."""
    return pd.DataFrame({
        "number": [f"INC00000{i}" for i in range(1, 21)],
        "opened_at": [f"2025-01-10 0{i%9}:15:00" for i in range(1, 21)],
        "resolved_at": [f"2025-01-10 1{i%9}:15:00" for i in range(1, 21)],
        "closed_at": [f"2025-01-11 1{i%9}:15:00" for i in range(1, 21)],
        "short_description": ["Network router interface down"] * 10 + ["Database deadlock on login table"] * 10,
        "description": ["Router gigabitethernet0/1 link protocol down"] * 10 + ["ORA-00060 deadlock detected while executing insert query"] * 10,
        "close_notes": ["Replaced optical transceiver"] * 10 + ["Killed blocking session"] * 10,
        "category": ["Network"] * 10 + ["Database"] * 10,
        "subcategory": ["Router"] * 10 + ["Oracle"] * 10,
        "assignment_group": ["Network Support"] * 10 + ["Database Support"] * 10,
        "priority": [1, 2, 3, 4] * 5,
        "business_impact": [1, 2, 2, 3] * 5,
        "urgency": [1, 2, 2, 3] * 5,
        "urgency": [1, 2, 2, 3] * 5,
        "reassignment_count": [0, 1, 0, 2] * 5,
        "reopen_count": [0, 0, 1, 0] * 5,
        "made_sla": [True, True, False, True] * 5,
        "problem_flag": [False, False, True, False] * 5,
        "knowledge_linked": [True, False, False, True] * 5
    })


def test_eda_engine_initialization() -> None:
    """Test initializing EDA engine with singleton registry."""
    engine = EnterpriseEDAEngine()
    assert engine.registry is not None
    assert engine.validator is not None


def test_eda_engine_analyze_dataset(sample_incident_df: pd.DataFrame, tmp_path: Path) -> None:
    """Test complete automated dataset analysis, chart generation, and report exports."""
    engine = EnterpriseEDAEngine()
    output_dir = str(tmp_path / "reports")
    
    results = engine.analyze_dataset(
        df=sample_incident_df,
        target_column="assignment_group",
        output_dir=output_dir,
        generate_figures=True
    )

    assert results["dataset_summary"]["total_records"] == 20
    assert results["dataset_summary"]["total_columns"] == len(sample_incident_df.columns)
    assert "priority" in results["numerical_analysis"]
    assert "category" in results["categorical_analysis"]
    assert "made_sla" in results["boolean_analysis"]
    assert "opened_at" in results["datetime_analysis"]
    assert "short_description" in results["text_analysis"]

    # Verify figures generated
    fig_dir = Path(output_dir) / "figures"
    assert fig_dir.exists()
    assert (fig_dir / "01_category_distribution.png").exists()
    assert (fig_dir / "02_priority_vs_sla.png").exists()
    assert (fig_dir / "03_hourly_arrival.png").exists()
    assert (fig_dir / "04_numerical_correlation.png").exists()
    assert (fig_dir / "05_text_word_counts.png").exists()

    # Verify exported reports
    assert (Path(output_dir) / "eda_report.json").exists()
    assert (Path(output_dir) / "eda_report.md").exists()
    assert (Path(output_dir) / "eda_report.html").exists()

    # Verify JSON content matches dictionary
    with open(Path(output_dir) / "eda_report.json", "r", encoding="utf-8") as f:
        data = json.load(f)
        assert data["dataset_summary"]["total_records"] == 20
