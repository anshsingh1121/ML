"""
Unit tests for Hyperparameter Optimizer (`v1.5.0`).
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import pytest

from src.data.feature_registry import FeatureRegistry
from src.ml.random_forest.hpo import HyperparameterOptimizer


@pytest.fixture
def sample_data() -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Create quick sample classification and regression data for HPO testing."""
    np.random.seed(42)
    n = 60
    df = pd.DataFrame({
        "priority": np.random.choice([1, 2, 3, 4], size=n),
        "impact": np.random.choice([1, 2, 3], size=n),
        "urgency": np.random.choice([1, 2, 3], size=n),
        "reassignment_count": np.random.randint(0, 5, size=n),
        "category": np.random.choice(["Software", "Network", "Hardware"], size=n),
    })
    y_clf = pd.Series(np.random.choice(["L2_Network", "L3_Database", "L1_ServiceDesk"], size=n))
    y_reg = pd.Series(np.random.uniform(1.0, 48.0, size=n))
    return df, y_clf, y_reg


def test_hpo_classifier(sample_data: Tuple[pd.DataFrame, pd.Series, pd.Series]) -> None:
    """Test HyperparameterOptimizer classification parameter tuning."""
    X, y_clf, _ = sample_data
    hpo = HyperparameterOptimizer()
    
    best_params = hpo.optimize_classifier(X, y_clf, target_col="assignment_group", n_iter=1, cv_folds=2)
    assert isinstance(best_params, dict)
    assert "n_estimators" in best_params
    assert Path("reports/hpo_comparison_assignment_group.json").exists()
    assert Path("reports/hpo_comparison_assignment_group.md").exists()


def test_hpo_regressor(sample_data: Tuple[pd.DataFrame, pd.Series, pd.Series]) -> None:
    """Test HyperparameterOptimizer regression parameter tuning."""
    X, _, y_reg = sample_data
    hpo = HyperparameterOptimizer()
    
    best_params = hpo.optimize_regressor(X, y_reg, target_col="resolution_time_hours", n_iter=1, cv_folds=2)
    assert isinstance(best_params, dict)
    assert "n_estimators" in best_params
    assert Path("reports/hpo_comparison_resolution_time_hours.json").exists()
    assert Path("reports/hpo_comparison_resolution_time_hours.md").exists()
