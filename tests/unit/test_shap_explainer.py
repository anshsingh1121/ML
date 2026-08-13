"""
Unit tests for SHAP Explainable AI & Structured Prediction Engine (`v1.5.0`).
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple
import joblib
import numpy as np
import pandas as pd
import pytest
from catboost import CatBoostClassifier
from sklearn.pipeline import Pipeline

from src.data.feature_registry import FeatureRegistry
from src.ml.explainability.shap_explainer import SHAPIntelligenceExplainer
from src.ml.catboost.trainer import EnterpriseCatBoostTrainer


@pytest.fixture
def dummy_model_and_data(tmp_path: Path) -> Tuple[Path, Path, pd.DataFrame]:
    """Create dummy model pipeline and dataset for SHAP testing."""
    np.random.seed(42)
    n = 30
    df = pd.DataFrame({
        "priority": np.random.choice([1, 2, 3, 4], size=n),
        "business_impact": np.random.choice([1, 2, 3], size=n),
        "urgency": np.random.choice([1, 2, 3], size=n),
        "reassignment_count": np.random.randint(0, 5, size=n),
        "category": np.random.choice(["Software", "Network", "Hardware"], size=n),
        "assignment_group": np.random.choice(["L2_Network", "L3_Database", "L1_ServiceDesk"], size=n)
    })
    test_p = tmp_path / "test.csv"
    df.to_csv(test_p, index=False)

    trainer = EnterpriseCatBoostTrainer()
    predictors = FeatureRegistry.get_instance().get_catboost_predictors()
    for col in predictors:
        if col not in df.columns:
            if "text" in col:
                df[col] = "safetoken"
            else:
                df[col] = 0

    X = df[predictors]
    y = df["assignment_group"]
    from catboost import CatBoostClassifier
    
    prep = trainer.build_preprocessing_pipeline(X, predictors)
    X_trans = prep.fit_transform(X, y)
    text_idx = X_trans.shape[1] - 1
    
    pipe = Pipeline([
        ("preprocessing", prep),
        ("estimator", CatBoostClassifier(iterations=5, random_seed=42, verbose=0))
    ])
    pipe.fit(X, y, estimator__text_features=[text_idx])
    model_p = tmp_path / "dummy_shap.pkl"
    joblib.dump(pipe, model_p)

    return model_p, test_p, df


def test_explain_global(dummy_model_and_data: Tuple[Path, Path, pd.DataFrame], tmp_path: Path) -> None:
    """Test global SHAP summary values and chart export (`shap_bar.png`)."""
    model_p, test_p, _ = dummy_model_and_data
    rep_dir = tmp_path / "reports"
    explainer = SHAPIntelligenceExplainer(reports_dir=rep_dir)

    importances = explainer.explain_global(
        model_key_or_path=str(model_p),
        test_path=str(test_p),
        sample_size=15,
        target_col="assignment_group"
    )
    assert isinstance(importances, dict)
    assert len(importances) > 0
    assert (rep_dir / "shap_bar.png").exists()
    assert (rep_dir / "shap_summary.png").exists()


def test_explain_prediction(dummy_model_and_data: Tuple[Path, Path, pd.DataFrame], tmp_path: Path) -> None:
    """Test local SHAP inference and structured prediction metadata export."""
    model_p, _, df = dummy_model_and_data
    rep_dir = tmp_path / "reports"
    explainer = SHAPIntelligenceExplainer(reports_dir=rep_dir)

    batch_input = df.head(5).to_dict(orient="records")
    results = explainer.explain_prediction(
        record_or_batch=batch_input,
        model_key_or_path=str(model_p),
        target_col="assignment_group"
    )
    assert isinstance(results, list)
    assert len(results) == 5
    assert "predicted_class" in results[0]
    assert "confidence_score" in results[0]
    assert "top_contributing_features" in results[0]
    assert "feature_importances" in results[0]
    assert (rep_dir / "prediction_metadata.json").exists()
    assert (rep_dir / "prediction_metadata.csv").exists()


def test_explain_regression(dummy_model_and_data: Tuple[Path, Path, pd.DataFrame], tmp_path: Path) -> None:
    """Test global and local SHAP diagnostics for regression target (`resolution_time_hours`)."""
    from catboost import CatBoostRegressor
    _, test_p, df = dummy_model_and_data
    df_reg = df.copy()
    df_reg["resolution_time_hours"] = np.random.uniform(1.0, 48.0, size=len(df_reg))
    test_reg_p = tmp_path / "test_reg.csv"
    df_reg.to_csv(test_reg_p, index=False)

    trainer = EnterpriseCatBoostTrainer()
    predictors = FeatureRegistry.get_instance().get_catboost_predictors()
    for col in predictors:
        if col not in df_reg.columns:
            if "text" in col:
                df_reg[col] = "safetoken"
            else:
                df_reg[col] = 0
                
    X = df_reg[predictors]
    y = df_reg["resolution_time_hours"]
    prep = trainer.build_preprocessing_pipeline(X, predictors)
    pipe = Pipeline([
        ("preprocessing", prep),
        ("estimator", CatBoostRegressor(iterations=5, random_seed=42, verbose=0))
    ])
    
    prep.fit(X, y)
    X_trans = prep.transform(X)
    text_idx = X_trans.shape[1] - 1
    
    pipe.fit(X, y, estimator__text_features=[text_idx])
    model_reg_p = tmp_path / "dummy_reg_shap.pkl"
    joblib.dump(pipe, model_reg_p)

    rep_dir = tmp_path / "reports_reg"
    explainer = SHAPIntelligenceExplainer(reports_dir=rep_dir)
    importances = explainer.explain_global(
        model_key_or_path=str(model_reg_p),
        test_path=str(test_reg_p),
        sample_size=15,
        target_col="resolution_time_hours"
    )
    assert isinstance(importances, dict)
    assert len(importances) > 0

    batch_input = df_reg.head(3).to_dict(orient="records")
    results = explainer.explain_prediction(
        record_or_batch=batch_input,
        model_key_or_path=str(model_reg_p),
        target_col="resolution_time_hours"
    )
    assert isinstance(results, list)
    assert "predicted_value" in results[0]


def test_explain_registry_key(tmp_path: Path) -> None:
    """Test loading model from ModelRegistry by key string format (`name:tag`)."""
    from src.ml.model_registry import ModelRegistry
    reg = ModelRegistry.get_instance()
    meta = reg.get_model_metadata("catboost_assignment_group", "latest")
    if meta and Path(meta.model_file_path).exists():
        explainer = SHAPIntelligenceExplainer(reports_dir=tmp_path / "reports_key")
        pipe, path = explainer._resolve_pipeline("catboost_assignment_group:latest")
        assert path == Path(meta.model_file_path)

