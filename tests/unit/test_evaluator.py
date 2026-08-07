"""
Unit tests for Model Evaluator (`v1.5.0`).
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline

from src.ml.model_registry import ModelRegistry
from src.ml.random_forest.evaluator import ModelEvaluator
from src.ml.random_forest.transformers import EnterpriseFeatureExtractor


@pytest.fixture
def dummy_pipeline_and_test_data(tmp_path: Path) -> Tuple[Path, Path, Path]:
    """Create dummy classification and regression models and test.csv file."""
    np.random.seed(42)
    n = 50
    df_test = pd.DataFrame({
        "priority": np.random.choice([1, 2, 3, 4], size=n),
        "impact": np.random.choice([1, 2, 3], size=n),
        "urgency": np.random.choice([1, 2, 3], size=n),
        "reassignment_count": np.random.randint(0, 5, size=n),
        "category": np.random.choice(["Software", "Network", "Hardware"], size=n),
        "assignment_group": np.random.choice(["L2_Network", "L3_Database", "L1_ServiceDesk"], size=n),
        "resolution_time_hours": np.random.uniform(1.0, 48.0, size=n)
    })
    test_p = tmp_path / "test.csv"
    df_test.to_csv(test_p, index=False)

    # Train a quick dummy classification pipeline using all authorized predictors
    from src.data.feature_registry import FeatureRegistry
    from src.ml.random_forest.trainer import EnterpriseRandomForestTrainer
    trainer = EnterpriseRandomForestTrainer()
    predictors = FeatureRegistry.get_instance().get_random_forest_predictors()
    for col in predictors:
        if col not in df_test.columns:
            df_test[col] = 0
    X = df_test[predictors]
    y_clf = df_test["assignment_group"]
    prep = trainer.build_preprocessing_pipeline(X, predictors)
    clf_pipe = Pipeline([
        ("preprocessing", prep),
        ("estimator", RandomForestClassifier(n_estimators=10, random_state=42))
    ])
    clf_pipe.fit(X, y_clf)
    clf_model_p = tmp_path / "dummy_clf.pkl"
    joblib.dump(clf_pipe, clf_model_p)

    # Train a quick dummy regression pipeline
    y_reg = df_test["resolution_time_hours"]
    prep_reg = trainer.build_preprocessing_pipeline(X, predictors)
    reg_pipe = Pipeline([
        ("preprocessing", prep_reg),
        ("estimator", RandomForestRegressor(n_estimators=10, random_state=42))
    ])
    reg_pipe.fit(X, np.log1p(y_reg))
    reg_model_p = tmp_path / "dummy_reg.pkl"
    joblib.dump(reg_pipe, reg_model_p)

    return clf_model_p, reg_model_p, test_p


def test_evaluate_classification(dummy_pipeline_and_test_data: Tuple[Path, Path, Path], tmp_path: Path) -> None:
    """Test classification evaluation, ROC curves, confusion matrix, and reports."""
    clf_p, _, test_p = dummy_pipeline_and_test_data
    rep_dir = tmp_path / "reports"
    evaluator = ModelEvaluator(reports_dir=rep_dir)

    metrics = evaluator.evaluate_classification(
        model_key_or_path=str(clf_p),
        test_path=str(test_p),
        target_col="assignment_group"
    )
    assert "accuracy" in metrics
    assert "f1_weighted" in metrics
    assert (rep_dir / "classification_report.md").exists()
    assert (rep_dir / "confusion_matrix.png").exists()
    assert (rep_dir / "feature_importance.csv").exists()


def test_evaluate_regression(dummy_pipeline_and_test_data: Tuple[Path, Path, Path], tmp_path: Path) -> None:
    """Test regression evaluation (`RMSE`, `MAE`, `R2`) and importance export."""
    _, reg_p, test_p = dummy_pipeline_and_test_data
    rep_dir = tmp_path / "reports"
    evaluator = ModelEvaluator(reports_dir=rep_dir)

    metrics = evaluator.evaluate_regression(
        model_key_or_path=str(reg_p),
        test_path=str(test_p),
        target_col="resolution_time_hours"
    )
    assert "rmse_hours" in metrics
    assert "mae_hours" in metrics
    assert "r2_score" in metrics
    assert (rep_dir / "regression_report.md").exists()
    assert (rep_dir / "feature_importance_resolution_time_hours.png").exists()
