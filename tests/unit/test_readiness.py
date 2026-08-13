"""Unit tests for MLReadinessEvaluator (`src/data/readiness.py`)."""

from pathlib import Path
import pandas as pd
import pytest


from src.data.readiness import MLReadinessEvaluator
from src.utils.config_manager import ConfigManager


def test_readiness_evaluation(temp_workspace: Path) -> None:
    """Verify MLReadinessEvaluator computes accurate stats and generates reports."""
    cfg = ConfigManager(config_dir=str(temp_workspace / "config"))
    df = pd.DataFrame({
        "number": [f"INC{i}" for i in range(10)],
        "assignment_group": ["GroupA", "GroupA", "GroupB", "GroupB", "GroupC", "GroupC", "GroupA", "GroupB", "GroupC", "GroupA"],
        "category": ["Hardware", "Software"] * 5,
        "priority": [1, 2, 3, 4, 5] * 2,
        "made_sla": [True, False] * 5,
        "resolution_time_hours": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    })

    readiness = MLReadinessEvaluator(config=cfg)
    report = readiness.evaluate_dataset(
        df=df,
        target_column="assignment_group",
        save_report=True,
        report_dir=str(temp_workspace / "reports")
    )

    assert report["total_records"] == 10
    assert report["primary_target"] == "assignment_group"
    assert "assignment_group" in report["class_imbalance"]
    assert "shannon_entropy" in report["class_imbalance"]["assignment_group"]
    assert report["target_leakage"]["has_leakage_risks"] is True
    assert len(report["recommended_preprocessing"]) > 0

    assert (temp_workspace / "reports" / "ml_readiness_report.json").exists()
    assert (temp_workspace / "reports" / "ml_readiness_report.md").exists()


def test_readiness_edge_cases(temp_workspace: Path) -> None:
    """Verify MLReadinessEvaluator handles empty or edge-case dataframes gracefully."""
    cfg = ConfigManager(config_dir=str(temp_workspace / "config"))
    readiness = MLReadinessEvaluator(config=cfg)

    # Empty df
    empty_df = pd.DataFrame()
    res_empty = readiness.evaluate_dataset(empty_df, save_report=False)
    assert res_empty["total_records"] == 0

    # Df with no numeric columns for correlation or single numeric
    tiny_df = pd.DataFrame({"number": ["INC1"], "priority": [1], "short_description": ["Hello"]})
    res_tiny = readiness.evaluate_dataset(tiny_df, save_report=False)
    assert res_tiny["total_records"] == 1
