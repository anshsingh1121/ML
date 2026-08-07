"""Unit tests for DatasetValidator (`src/data/validation.py`)."""

from pathlib import Path
import pandas as pd
import pytest


from src.data.validation import DatasetValidator, CheckResult
from src.utils.config_manager import ConfigManager


def test_validator_on_clean_generated_dataset(temp_workspace: Path) -> None:
    """Verify DatasetValidator passes all checks on clean synthetic data."""
    cfg = ConfigManager(config_dir=str(temp_workspace / "config"))
    df = pd.DataFrame({
        "incident_number": [f"INC{i:03d}" for i in range(100)],
        "opened_at": [f"2026-03-14 12:00:00" for _ in range(100)],
        "resolved_at": [f"2026-03-14 14:00:00" for _ in range(100)],
        "closed_at": [f"2026-03-15 14:00:00" for _ in range(100)],
        "priority": [3] * 100,
        "category": ["Hardware"] * 100,
        "assignment_group": ["Hardware_Team"] * 100,
        "short_description": ["Server down"] * 100,
        "description": ["The server is down and needs fixing."] * 100,
        "made_sla": [True] * 100,
        "sla_status": ["Met"] * 100,
        "resolution_time_hours": [2.0] * 100,
        "cmdb_ci": ["ci_01"] * 100,
        "business_service": ["Internal IT"] * 100
    })

    validator = DatasetValidator(config=cfg)
    summary = validator.validate_dataset(df, save_report=True, report_dir=str(temp_workspace / "reports"))

    assert summary["total_records"] == 100
    assert summary["is_valid"] is True
    assert summary["passed_checks"] == summary["total_checks"]
    assert (temp_workspace / "reports" / "validation_report.json").exists()
    assert (temp_workspace / "reports" / "validation_report.md").exists()


def test_validator_detects_anomalies(temp_workspace: Path) -> None:
    """Verify DatasetValidator correctly flags missing values, duplicate IDs, and timestamp errors."""
    cfg = ConfigManager(config_dir=str(temp_workspace / "config"))
    
    # Create corrupted dataframe
    corrupted_data = pd.DataFrame({
        "incident_number": ["INC001", "INC001", "INC003"],  # Duplicate INC001
        "opened_at": ["2026-03-14 12:00:00", "2026-03-14 14:00:00", "2026-03-14 15:00:00"],
        "resolved_at": ["2026-03-14 10:00:00", "2026-03-14 16:00:00", None],  # Resolved before opened for INC001
        "closed_at": [None, None, None],
        "priority": [1, 99, 3],  # Invalid priority 99
        "category": ["Core Banking", "", "Security"],  # Empty category
        "assignment_group": ["Core_Banking_L2", "UNKNOWN_TEAM", "SOC_Security_L3"],
        "short_description": ["Valid description", "", "Another valid"],  # Empty short_description
        "description": ["Details", "More details", None],  # Missing description
        "made_sla": [True, False, True],
        "sla_status": ["Met", "Breached", "Met"],
        "resolution_time_hours": [-5.0, 2.0, 1.0],  # Negative resolution time
        "cmdb_ci": ["ci_01", "", "ci_03"],
        "business_service": ["SWIFT Payments", None, "ATM Network"]
    })

    validator = DatasetValidator(config=cfg)
    summary = validator.validate_dataset(corrupted_data, save_report=False)

    assert summary["is_valid"] is False
    assert summary["failed_checks"] > 0

    # Verify specific check results
    checks_dict = {c["rule_name"]: c for c in summary["checks"]}
    assert checks_dict["Duplicate Incident Numbers"]["passed"] is False
    assert checks_dict["Invalid Timestamps"]["passed"] is False
    assert checks_dict["Invalid Categories"]["passed"] is False
    assert checks_dict["Invalid Priorities"]["passed"] is False
    assert checks_dict["Empty Short Descriptions"]["passed"] is False
    assert checks_dict["Empty Descriptions"]["passed"] is False
