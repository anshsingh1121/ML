"""
Unit tests for Enterprise Random Forest Trainer & Custom Transformers (`v1.5.0`).
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import pytest
import joblib

from src.data.feature_registry import FeatureRegistry
from src.ml.model_registry import ModelRegistry
from src.ml.random_forest.transformers import (
    DataFrameSelector,
    EnterpriseFeatureExtractor,
    FrequencyEncoder,
    SmoothedTargetEncoder,
)
from src.ml.random_forest.trainer import EnterpriseRandomForestTrainer


@pytest.fixture
def synthetic_incidents_csv(tmp_path: Path) -> Tuple[Path, Path, Path]:
    """Create synthetic train.csv, val.csv, and test.csv partitions for fast ML testing."""
    n = 100
    np.random.seed(42)
    df = pd.DataFrame({
        "incident_number": [f"INC{10000+i}" for i in range(n)],
        "priority": np.random.choice([1, 2, 3, 4], size=n),
        "impact": np.random.choice([1, 2, 3], size=n),
        "urgency": np.random.choice([1, 2, 3], size=n),
        "state": np.random.choice([1, 2, 6], size=n),
        "contact_type": np.random.choice(["Alert", "Phone", "Self-service", "Email"], size=n),
        "category": np.random.choice(["Software", "Network", "Hardware"], size=n),
        "subcategory": np.random.choice(["VPN", "Database", "Memory", "CPU"], size=n),
        "business_service": np.random.choice(["Retail Banking", "Payment Gateway", "Mobile Banking"], size=n),
        "location": np.random.choice(["New York", "Charlotte", "Raleigh"], size=n),
        "cmdb_ci": np.random.choice(["db_server_01", "app_server_02"], size=n),
        "vendor": np.random.choice(["Cisco", "Oracle", "Microsoft"], size=n),
        "reassignment_count": np.random.randint(0, 5, size=n),
        "reopen_count": np.random.randint(0, 2, size=n),
        "opened_at": pd.date_range("2026-01-01", periods=n, freq="1h").astype(str),
        "assignment_group": np.random.choice(["L2_Network", "L3_Database", "L1_ServiceDesk"], size=n),
        "resolution_time_hours": np.random.uniform(0.5, 48.0, size=n).round(2),
        "resolved_at": pd.date_range("2026-01-02", periods=n, freq="1h").astype(str)  # Blocked leakage col
    })

    train_p = tmp_path / "train.csv"
    val_p = tmp_path / "val.csv"
    test_p = tmp_path / "test.csv"

    df.iloc[:60].to_csv(train_p, index=False)
    df.iloc[60:80].to_csv(val_p, index=False)
    df.iloc[80:].to_csv(test_p, index=False)

    return train_p, val_p, test_p


def test_custom_transformers() -> None:
    """Test DataFrameSelector, EnterpriseFeatureExtractor, FrequencyEncoder, and SmoothedTargetEncoder."""
    df = pd.DataFrame({
        "priority": [1, 2, 3, 4],
        "impact": [2, 2, 3, 1],
        "category": ["A", "A", "B", "C"],
        "subcategory": ["X", "X", "Y", "Z"],
        "opened_at": ["2026-01-01 10:00:00", "2026-01-01 14:00:00", "2026-01-02 08:00:00", "2026-01-03 16:00:00"]
    })

    # DataFrameSelector
    selector = DataFrameSelector(attribute_names=["priority", "category"], return_array=False)
    out_sel = selector.fit_transform(df)
    assert list(out_sel.columns) == ["priority", "category"]

    # EnterpriseFeatureExtractor
    extractor = EnterpriseFeatureExtractor()
    out_ext = extractor.fit_transform(df)
    assert "priority_x_impact" in out_ext.columns
    assert "opened_at_hour_sin" in out_ext.columns

    # FrequencyEncoder
    freq_enc = FrequencyEncoder(columns=["category", "subcategory"])
    out_freq = freq_enc.fit_transform(df)
    assert out_freq["category"].iloc[0] == 0.5  # 'A' appears 2 out of 4 times

    # SmoothedTargetEncoder
    y = pd.Series([10.0, 20.0, 30.0, 40.0])
    target_enc = SmoothedTargetEncoder(columns=["category"], smoothing=2.0)
    out_target = target_enc.fit_transform(df, y)
    assert out_target["category"].dtype == float


def test_trainer_target_leakage_interlock() -> None:
    """Verify that passing a blocked target leakage column raises ValueError immediately."""
    trainer = EnterpriseRandomForestTrainer()
    with pytest.raises(ValueError, match="Blocked target leakage feature"):
        trainer._verify_no_target_leakage(["priority", "resolved_at"])


def test_trainer_build_preprocessing_pipeline(synthetic_incidents_csv: Tuple[Path, Path, Path]) -> None:
    """Test zero-leakage scikit-learn preprocessing pipeline construction."""
    train_p, _, _ = synthetic_incidents_csv
    df = pd.read_csv(train_p)
    trainer = EnterpriseRandomForestTrainer()
    predictors = FeatureRegistry.get_instance().get_random_forest_predictors()

    pipe = trainer.build_preprocessing_pipeline(df, predictors)
    out = pipe.fit_transform(df)
    assert out.shape[0] == len(df)
    assert out.shape[1] > 0


def test_train_baselines_and_compare(synthetic_incidents_csv: Tuple[Path, Path, Path]) -> None:
    """Test multi-baseline model comparison (Decision Tree, Random Forest, Extra Trees)."""
    train_p, val_p, _ = synthetic_incidents_csv
    df_train = pd.read_csv(train_p)
    df_val = pd.read_csv(val_p)

    trainer = EnterpriseRandomForestTrainer()
    predictors = FeatureRegistry.get_instance().get_random_forest_predictors()

    pipelines_dict, best_name = trainer.train_baselines_and_compare(
        df_train, df_train["assignment_group"],
        df_val, df_val["assignment_group"],
        predictors, target_type="assignment_group"
    )

    assert "DecisionTree" in pipelines_dict
    assert "RandomForest" in pipelines_dict
    assert "ExtraTrees" in pipelines_dict
    assert best_name in pipelines_dict

    # Check report files
    assert Path("reports/model_comparison_assignment_group.json").exists()
    assert Path("reports/model_comparison_assignment_group.md").exists()


def test_train_classifier_and_regressor(synthetic_incidents_csv: Tuple[Path, Path, Path]) -> None:
    """Test primary classification and regression training, joblib persistence, and ModelRegistry interlock."""
    train_p, val_p, _ = synthetic_incidents_csv
    trainer = EnterpriseRandomForestTrainer()

    # Classification
    clf_path = trainer.train_classifier(train_path=str(train_p), val_path=str(val_p), target_col="assignment_group", compare_baselines=False)
    assert clf_path.exists()
    pipe_clf = joblib.load(clf_path)
    assert hasattr(pipe_clf, "predict")

    meta_clf = ModelRegistry.get_instance().get_model_metadata("random_forest_assignment_group")
    assert meta_clf is not None
    assert meta_clf.target_variable == "assignment_group"

    # Regression
    reg_path = trainer.train_regressor(train_path=str(train_p), val_path=str(val_p), target_col="resolution_time_hours", compare_baselines=False)
    assert reg_path.exists()
    pipe_reg = joblib.load(reg_path)
    assert hasattr(pipe_reg, "predict")

    meta_reg = ModelRegistry.get_instance().get_model_metadata("random_forest_resolution_time_hours")
    assert meta_reg is not None
    assert meta_reg.target_variable == "resolution_time_hours"
