"""
Enterprise Hyperparameter Optimization Engine (`v1.5.0`).

Performs configurable `GridSearchCV` and `RandomizedSearchCV` tuning across zero-leakage
scikit-learn `Pipeline` objects (`RandomForestClassifier` & `RandomForestRegressor`).
Exports tuning comparison logs and best parameter distributions for production persistence.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from src.data.feature_registry import FeatureRegistry
from src.ml.random_forest.trainer import EnterpriseRandomForestTrainer
from src.utils.config_manager import ConfigManager
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CatBoostClassifierWrapper(CatBoostClassifier):
    def fit(self, X, y=None, **kwargs):
        # Dynamically attach the last column as text_features to handle CV shifts
        if "text_features" not in kwargs:
            kwargs["text_features"] = [X.shape[1] - 1]
        return super().fit(X, y, **kwargs)

class CatBoostRegressorWrapper(CatBoostRegressor):
    def fit(self, X, y=None, **kwargs):
        if "text_features" not in kwargs:
            kwargs["text_features"] = [X.shape[1] - 1]
        return super().fit(X, y, **kwargs)

class HyperparameterOptimizer:
    """
    Enterprise Hyperparameter Optimizer across zero-leakage scikit-learn pipelines.
    Supports StratifiedKFold classification tuning and KFold regression tuning.
    """

    def __init__(self, config_path: Optional[str] = None) -> None:
        self.cfg = ConfigManager(config_path)
        self.feat_reg = FeatureRegistry.get_instance()
        self.trainer = EnterpriseRandomForestTrainer(config_path)

        self.reports_dir = Path("reports")
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        self.hpo_cfg = self.cfg.get("hyperparameter_optimization", {})

    def _prepare_pipeline_param_grid(self, raw_grid: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        """Prefix parameter names with 'estimator__' to match sklearn Pipeline structure."""
        pipeline_grid = {}
        for k, v in raw_grid.items():
            key = f"estimator__{k}" if not k.startswith("estimator__") else k
            # Clean null / None string representations if present
            cleaned_vals = [None if val in ["null", "None", None] else val for val in v]
            pipeline_grid[key] = cleaned_vals
        return pipeline_grid

    def optimize_classifier(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        target_col: str = "assignment_group",
        n_iter: Optional[int] = None,
        cv_folds: Optional[int] = None
    ) -> Dict[str, Any]:
        """Run StratifiedKFold hyperparameter search across CatBoostClassifier pipeline."""
        start_t = time.time()
        logger.info(f"Initiating Hyperparameter Optimization for classifier target: '{target_col}'...")

        predictors = self.feat_reg.get_random_forest_predictors()
        prep_pipeline = self.trainer.build_preprocessing_pipeline(X_train, predictors)
        
        # Determine text feature index after preprocessing
        estimator = CatBoostClassifierWrapper(random_seed=42, verbose=0)
        full_pipe = Pipeline([
            ("preprocessing", prep_pipeline),
            ("estimator", estimator)
        ])

        raw_grid = {
            "iterations": [100, 200, 300],
            "depth": [4, 6, 8],
            "learning_rate": [0.05, 0.1]
        }
        param_grid = self._prepare_pipeline_param_grid(raw_grid)

        method = self.hpo_cfg.get("method", "randomized_search").lower()
        cv = int(self.hpo_cfg.get("cv_folds", 5))
        iters = int(self.hpo_cfg.get("n_iter", 20))
        scoring = self.hpo_cfg.get("scoring", "f1_weighted")

        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

        if method == "grid_search":
            logger.info(f"Running GridSearchCV across {cv} folds (scoring={scoring})...")
            search = GridSearchCV(full_pipe, param_grid=param_grid, cv=skf, scoring=scoring, n_jobs=-1, verbose=1)
        else:
            logger.info(f"Running RandomizedSearchCV across {cv} folds ({iters} iterations, scoring={scoring})...")
            search = RandomizedSearchCV(full_pipe, param_distributions=param_grid, n_iter=iters, cv=skf, scoring=scoring, random_state=42, n_jobs=-1, verbose=1)

        X_train_clean = self.trainer._get_safe_predictor_matrix(X_train, predictors)
        search.fit(X_train_clean, y_train.astype(str))
        dur = time.time() - start_t

        best_params_clean = {k.replace("estimator__", ""): v for k, v in search.best_params_.items()}
        logger.info(f"HPO completed in {dur:.2f}s! Best {scoring}: {search.best_score_:.4f}")
        logger.info(f"Best Parameters identified: {best_params_clean}")

        self._export_hpo_report(search, best_params_clean, target_col, dur, scoring)
        return best_params_clean

    def optimize_regressor(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        target_col: str = "resolution_time_hours",
        n_iter: Optional[int] = None,
        cv_folds: Optional[int] = None
    ) -> Dict[str, Any]:
        """Run KFold hyperparameter search across CatBoostRegressor pipeline."""
        start_t = time.time()
        logger.info(f"Initiating Hyperparameter Optimization for regressor target: '{target_col}'...")

        predictors = self.feat_reg.get_random_forest_predictors()
        prep_pipeline = self.trainer.build_preprocessing_pipeline(X_train, predictors)

        # Determine text feature index after preprocessing
        estimator = CatBoostRegressorWrapper(random_seed=42, verbose=0, loss_function="MAE")
        full_pipe = Pipeline([
            ("preprocessing", prep_pipeline),
            ("estimator", estimator)
        ])

        raw_grid = {
            "iterations": [100, 200, 300],
            "depth": [4, 6, 8],
            "learning_rate": [0.05, 0.1]
        }
        param_grid = self._prepare_pipeline_param_grid(raw_grid)

        method = self.hpo_cfg.get("method", "randomized_search").lower()
        cv = int(self.hpo_cfg.get("cv_folds", 5))
        iters = int(self.hpo_cfg.get("n_iter", 20))
        scoring = "neg_mean_absolute_error"

        kf = KFold(n_splits=cv, shuffle=True, random_state=42)

        if method == "grid_search":
            search = GridSearchCV(full_pipe, param_grid=param_grid, cv=kf, scoring=scoring, n_jobs=-1, verbose=1)
        else:
            search = RandomizedSearchCV(full_pipe, param_distributions=param_grid, n_iter=iters, cv=kf, scoring=scoring, random_state=42, n_jobs=-1, verbose=1)

        X_train_clean = self.trainer._get_safe_predictor_matrix(X_train, predictors)
        search.fit(X_train_clean, y_train)
        dur = time.time() - start_t

        best_params_clean = {k.replace("estimator__", ""): v for k, v in search.best_params_.items()}
        logger.info(f"Regression HPO completed in {dur:.2f}s! Best {scoring}: {search.best_score_:.4f}")
        logger.info(f"Best Parameters identified: {best_params_clean}")

        self._export_hpo_report(search, best_params_clean, target_col, dur, scoring)
        return best_params_clean

    def _export_hpo_report(self, search_obj: Any, best_params: Dict[str, Any], target: str, duration: float, scoring: str) -> None:
        """Export detailed HPO candidates comparison table to JSON and Markdown."""
        json_path = self.reports_dir / f"hpo_comparison_{target}.json"
        md_path = self.reports_dir / f"hpo_comparison_{target}.md"

        cv_results = search_obj.cv_results_
        top_indices = np.argsort(cv_results["rank_test_score"])[:10]  # Top 10 configurations

        candidates = []
        for idx in top_indices:
            candidates.append({
                "rank": int(cv_results["rank_test_score"][idx]),
                "mean_score": round(float(cv_results["mean_test_score"][idx]), 4),
                "std_score": round(float(cv_results["std_test_score"][idx]), 4),
                "params": {k.replace("estimator__", ""): str(v) for k, v in cv_results["params"][idx].items()}
            })

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({
                "target_variable": target,
                "scoring_metric": scoring,
                "tuning_duration_sec": round(duration, 2),
                "best_score": round(float(search_obj.best_score_), 4),
                "best_params": {k: str(v) for k, v in best_params.items()},
                "top_10_candidates": candidates
            }, f, indent=2)

        lines = [
            f"# Hyperparameter Optimization Report (`{target}`)",
            "**Organization:** First Citizens Bank — Incident Intelligence Platform (IIP)  ",
            f"**Tuning Duration:** {duration:.2f}s (`{len(cv_results['params'])}` configurations evaluated)  ",
            f"**Scoring Metric:** `{scoring}` | **Best Score Achieved:** `{search_obj.best_score_:.4f}`  \n",
            "---",
            "\n## Top 10 Parameter Configurations\n",
            "| Rank | Mean CV Score | Std Dev | Hyperparameter Configuration |",
            "|:---:|:---:|:---:|---|",
        ]

        for c in candidates:
            badge = "🏆 Rank 1" if c["rank"] == 1 else f"Rank {c['rank']}"
            lines.append(f"| **{badge}** | **{c['mean_score']}** | `±{c['std_score']}` | `{json.dumps(c['params'])}` |")

        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        logger.info(f"Exported HPO report to {json_path} & {md_path}")
