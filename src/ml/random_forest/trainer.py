"""
Enterprise Random Forest Trainer (`v1.5.0`).

Orchestrates data loading, zero-leakage preprocessing pipeline construction (`scikit-learn`),
multi-baseline model training (Decision Tree, Random Forest, Extra Trees, and optional XGBoost/LightGBM),
target leakage interlock verification, `joblib` Pipeline persistence, and formal registration inside `ModelRegistry`.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from src.utils import robust_read_csv
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from src.data.feature_lineage import FeatureLineageTracker
from src.data.feature_registry import FeatureRegistry
from src.data.pipeline_contracts import PipelineContractValidator
from src.ml.model_registry import ModelRegistry
from src.ml.random_forest.transformers import EnterpriseFeatureExtractor, FrequencyEncoder
from src.utils.config_manager import ConfigManager
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EnterpriseRandomForestTrainer:
    """
    Enterprise-grade Random Forest and baseline ML trainer for First Citizens Bank Incident Intelligence Platform.
    Ensures zero target leakage, complete preprocessing + estimator pipeline persistence, and model registry compliance.
    """

    def __init__(self, config_path: Optional[str] = None) -> None:
        self.cfg = ConfigManager(config_path)
        self.feat_reg = FeatureRegistry.get_instance()
        self.lineage = FeatureLineageTracker.get_instance()
        self.validator = PipelineContractValidator()
        self.model_reg = ModelRegistry.get_instance()

        self.models_dir = Path("models")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir = Path("reports")
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def _verify_no_target_leakage(self, feature_names: List[str]) -> None:
        """
        Interlock: Verify that no unauthorized or blocked target leakage columns enter the predictor matrix.
        """
        for feat_name in feature_names:
            feat_def = self.feat_reg.get_feature(feat_name)
            if feat_def and feat_def.target_leakage_classification == "blocked":
                err_msg = f"[SECURITY INTERLOCK] Blocked target leakage feature '{feat_name}' rejected from ML predictor set!"
                logger.error(err_msg)
                raise ValueError(err_msg)
        logger.debug("Verified zero target leakage across all authorized ML predictor columns.")

    def _get_safe_predictor_matrix(self, df: pd.DataFrame, predictors: List[str]) -> pd.DataFrame:
        """Ensure all predictor columns exist in the dataframe before slicing, initializing missing with safe defaults."""
        df_clean = df.copy()
        for col in predictors:
            if col not in df_clean.columns:
                df_clean[col] = "UNKNOWN" if col in ["category", "subcategory", "business_service", "location", "cmdb_ci", "vendor", "contact_type"] else 0
        return df_clean[predictors]

    def build_preprocessing_pipeline(self, X: pd.DataFrame, predictors: List[str]) -> Pipeline:
        """
        Construct a self-contained scikit-learn preprocessing pipeline for zero-leakage inference.
        Partition predictors into frequency encoded, one-hot encoded, and numerical scaled branches.
        """
        self._verify_no_target_leakage(predictors)

        # Categorize features based on FeatureRegistry rules or dtypes
        freq_cols = []
        onehot_cols = []
        num_cols = []

        for col in predictors:
            if col not in X.columns:
                continue
            
            # Text columns are handled separately by TfidfVectorizer via combined_text
            if col in ["short_description", "description"]:
                continue
                
            feat_def = self.feat_reg.get_feature(col)
            strategy = feat_def.encoding_strategy if feat_def else "none"

            if strategy in ["frequency", "label", "target"]:
                freq_cols.append(col)
            elif strategy in ["one_hot", "ordinal"]:
                onehot_cols.append(col)
            else:
                # Numerical or cyclic shift columns
                num_cols.append(col)

        # Include columns added dynamically by EnterpriseFeatureExtractor inside num_cols branch
        if "priority" in predictors and "impact" in predictors and "priority_x_impact" not in num_cols:
            num_cols.append("priority_x_impact")
        if "priority" in predictors and "urgency" in predictors and "priority_x_urgency" not in num_cols:
            num_cols.append("priority_x_urgency")
        if "opened_at" in predictors:
            for cyclic_col in ["opened_at_hour_sin", "opened_at_hour_cos", "opened_at_dayofweek_sin", "opened_at_dayofweek_cos"]:
                if cyclic_col not in num_cols:
                    num_cols.append(cyclic_col)

        # Build column transformer with verbose_feature_names_out=False to preserve clean technical lineage
        transformers = []
        if freq_cols:
            transformers.append(("freq", FrequencyEncoder(columns=freq_cols), freq_cols))
        if onehot_cols:
            transformers.append(("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False), onehot_cols))
        if num_cols:
            transformers.append(("num", SimpleImputer(strategy="median"), num_cols))
            
        # Pass raw unstructured text directly into the pipeline for CatBoost Native NLP Engine
        transformers.append(("text_raw", "passthrough", ["combined_text"]))

        col_trans = ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)

        prep_pipeline = Pipeline([
            ("extractor", EnterpriseFeatureExtractor()),
            ("col_transform", col_trans)
        ])

        logger.info(f"Built zero-leakage preprocessing pipeline across {len(predictors)} predictors (Freq: {len(freq_cols)}, OneHot: {len(onehot_cols)}, Num: {len(num_cols)}).")
        return prep_pipeline

    def train_baselines_and_compare(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        predictors: List[str],
        target_type: str = "assignment_group"
    ) -> Tuple[Dict[str, Pipeline], str]:
        """
        Train multiple baseline models (Decision Tree, Random Forest, Extra Trees, optional XGBoost/LightGBM).
        Compare validation metrics, generate formal reports, and automatically identify the best-performing model.
        """
        logger.info(f"Initiating multi-baseline comparison for target '{target_type}' across {len(predictors)} predictors...")
        X_train = self._get_safe_predictor_matrix(X_train, predictors)
        X_val = self._get_safe_predictor_matrix(X_val, predictors)
        prep_pipeline = self.build_preprocessing_pipeline(X_train, predictors)
        prep_pipeline.fit(X_train, y_train)

        X_train_trans = prep_pipeline.transform(X_train)
        X_val_trans = prep_pipeline.transform(X_val)

        text_idx = X_train_trans.shape[1] - 1

        is_classification = (target_type in ["assignment_group", "category", "priority"])
        models_dict: Dict[str, Any] = {}

        if is_classification:
            rf_cfg = self.cfg.get(f"models.{target_type}.params", {})
            models_dict["CatBoost"] = CatBoostClassifier(
                iterations=rf_cfg.get("n_estimators", 300),
                depth=rf_cfg.get("max_depth", 6),
                learning_rate=0.1,
                verbose=0,
                random_seed=42,
                text_features=[text_idx]
            )
        else:
            # Regression for resolution_time_hours
            rf_cfg = self.cfg.get("models.resolution_time.params", {})
            models_dict["CatBoost"] = CatBoostRegressor(
                iterations=rf_cfg.get("n_estimators", 150),
                depth=rf_cfg.get("max_depth", 6),
                learning_rate=0.1,
                verbose=0,
                random_seed=42,
                text_features=[text_idx]
            )

        comparison_results = []
        fitted_pipelines: Dict[str, Pipeline] = {}
        best_score = -float("inf") if is_classification else float("inf")
        best_model_name = "CatBoost"

        for name, estimator in models_dict.items():
            start_t = time.time()
            try:
                # Fit estimator on transformed features
                estimator.fit(X_train_trans, y_train)
                train_dur = time.time() - start_t

                # Evaluate on validation set
                preds = estimator.predict(X_val_trans)

                if is_classification:
                    acc = accuracy_score(y_val, preds)
                    f1 = f1_score(y_val, preds, average="weighted", zero_division=0)
                    score = f1
                    if score > best_score:
                        best_score = score
                        best_model_name = name
                    metrics = {"accuracy": round(acc, 4), "f1_weighted": round(f1, 4)}
                else:
                    # Inverse log1p transform if regression targets were log-transformed
                    preds_inv = np.expm1(np.clip(preds, 0, 15)) if target_type == "resolution_time_hours" else preds
                    y_val_inv = np.expm1(np.clip(y_val, 0, 15)) if target_type == "resolution_time_hours" else y_val
                    rmse = float(np.sqrt(mean_squared_error(y_val_inv, preds_inv)))
                    mae = float(mean_absolute_error(y_val_inv, preds_inv))
                    r2 = float(r2_score(y_val_inv, preds_inv))
                    score = rmse
                    if score < best_score:
                        best_score = score
                        best_model_name = name
                    metrics = {"rmse": round(rmse, 4), "mae": round(mae, 4), "r2": round(r2, 4)}

                # Build full scikit-learn pipeline for this baseline
                full_pipe = Pipeline([
                    ("preprocessing", prep_pipeline),
                    ("estimator", estimator)
                ])
                fitted_pipelines[name] = full_pipe

                comparison_results.append({
                    "model": name,
                    "target_variable": target_type,
                    "train_duration_sec": round(train_dur, 2),
                    "primary_score": round(score, 4),
                    "metrics": metrics,
                    "status": "PASS"
                })
                logger.info(f"Baseline [{name}]: {metrics} (Trained in {train_dur:.2f}s)")

            except Exception as e:
                logger.error(f"Failed to train baseline [{name}]: {e}")
                comparison_results.append({
                    "model": name,
                    "target_variable": target_type,
                    "train_duration_sec": 0.0,
                    "primary_score": 0.0,
                    "metrics": {"error": str(e)},
                    "status": "FAIL"
                })

        # Export comparison report
        self._export_baseline_comparison_report(comparison_results, best_model_name, target_type)
        logger.info(f"Multi-baseline comparison complete! Best Model identified: [{best_model_name}] (Score: {best_score:.4f})")
        return fitted_pipelines, best_model_name

    def _export_baseline_comparison_report(self, results: List[Dict[str, Any]], best_model: str, target: str) -> None:
        """Export multi-baseline comparison results to markdown and json reports."""
        json_path = self.reports_dir / f"model_comparison_{target}.json"
        md_path = self.reports_dir / f"model_comparison_{target}.md"

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"target_variable": target, "best_model": best_model, "results": results}, f, indent=2)

        lines = [
            f"# Multi-Baseline Model Comparison Report (`{target}`)",
            "**Organization:** First Citizens Bank — Incident Intelligence Platform (IIP)  ",
            f"**Evaluation Timestamp:** {datetime.now().isoformat()}  ",
            f"**Best Performing Model Identified:** `🏆 {best_model}`  \n",
            "---",
            "\n## Baseline Model Performance Matrix\n",
            "| Model Name | Target Variable | Training Time (s) | Primary Score | Metrics Summary | Status |",
            "|---|---|:---:|:---:|---|:---:|"
        ]

        for r in results:
            badge = "🏆 BEST" if r["model"] == best_model else "🟢 PASS" if r["status"] == "PASS" else "🔴 FAIL"
            lines.append(
                f"| `{r['model']}` | `{r['target_variable']}` | {r['train_duration_sec']}s | "
                f"**{r['primary_score']}** | `{json.dumps(r['metrics'])}` | {badge} |"
            )

        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        logger.info(f"Exported baseline comparison reports to {json_path} & {md_path}")

    def train_classifier(
        self,
        train_path: Optional[str] = None,
        val_path: Optional[str] = None,
        target_col: str = "assignment_group",
        compare_baselines: bool = True
    ) -> Path:
        """
        Train primary CatBoost (and multi-baseline) classification pipeline on triage predictors.
        Persist complete sklearn Pipeline to disk (`models/catboost_assignment_group.pkl`), and register inside ModelRegistry.
        """
        start_t = time.time()
        train_file = Path(train_path or "data/processed/train.csv")
        val_file = Path(val_path or "data/processed/val.csv")

        if not train_file.exists() or not val_file.exists():
            raise FileNotFoundError(f"Training dataset partitions missing! Ensure {train_file} and {val_file} exist.")

        logger.info(f"Loading training data from {train_file} and validation data from {val_file}...")
        df_train = robust_read_csv(train_file)
        df_val = robust_read_csv(val_file)

        predictors = self.feat_reg.get_random_forest_predictors()
        # Verify no target leakage
        self._verify_no_target_leakage(predictors)

        X_train = self._get_safe_predictor_matrix(df_train, predictors)
        y_train = df_train[target_col].astype(str)
        X_val = self._get_safe_predictor_matrix(df_val, predictors)
        y_val = df_val[target_col].astype(str)

        if compare_baselines:
            pipelines_dict, best_name = self.train_baselines_and_compare(X_train, y_train, X_val, y_val, predictors, target_type=target_col)
            # Primary model is CatBoost or Best Baseline as requested
            primary_pipeline = pipelines_dict.get("CatBoost", pipelines_dict[best_name])
        else:
            prep = self.build_preprocessing_pipeline(X_train, predictors)
            X_train_trans = prep.fit_transform(X_train, y_train)
            text_idx = X_train_trans.shape[1] - 1
            
            rf_cfg = self.cfg.get(f"models.{target_col}.params", {})
            estimator = CatBoostClassifier(
                iterations=rf_cfg.get("n_estimators", 300),
                depth=rf_cfg.get("max_depth", 6),
                learning_rate=0.1,
                verbose=0,
                random_seed=42,
                text_features=[text_idx]
            )
            primary_pipeline = Pipeline([("preprocessing", prep), ("estimator", estimator)])
            primary_pipeline.fit(X_train, y_train)

        # Evaluate final pipeline on validation fold
        val_preds = primary_pipeline.predict(X_val)
        acc = accuracy_score(y_val, val_preds)
        f1 = f1_score(y_val, val_preds, average="weighted", zero_division=0)
        train_dur = time.time() - start_t

        # Persist complete pipeline
        out_path = self.models_dir / f"catboost_{target_col}.pkl"
        joblib.dump(primary_pipeline, out_path)
        logger.info(f"Successfully persisted complete classification pipeline to: {out_path}")

        # Register inside ModelRegistry
        hyperparams = primary_pipeline.named_steps["estimator"].get_params() if "estimator" in primary_pipeline.named_steps else {}
        # Clean non-serializable objects from params dict
        hyperparams_clean = {k: str(v) if not isinstance(v, (int, float, str, bool, type(None))) else v for k, v in hyperparams.items()}

        self.model_reg.register_model(
            model_name=f"catboost_{target_col}",
            version="v1.0.0",
            training_dataset_uri=str(train_file),
            dataset_version="v2.0.0-alpha",
            hyperparameters=hyperparams_clean,
            metrics={"accuracy": round(acc, 4), "f1_weighted": round(f1, 4), "training_duration_sec": round(train_dur, 2)},
            features_used=predictors,
            target_variable=target_col,
            model_file_path=str(out_path),
            status="Active"
        )

        logger.info(f"Registered model catboost_{target_col}:v1.0.0 (Acc: {acc:.4f}, F1: {f1:.4f})")
        return out_path

    def train_regressor(
        self,
        train_path: Optional[str] = None,
        val_path: Optional[str] = None,
        target_col: str = "resolution_time_hours",
        compare_baselines: bool = True
    ) -> Path:
        """
        Train primary CatBoost (and multi-baseline) regression pipeline on triage predictors.
        Applies log1p target transformation (`np.log1p`) to normalize right-skewed resolution windows.
        Persist complete sklearn Pipeline to disk (`models/catboost_resolution_time_hours.pkl`), and register inside ModelRegistry.
        """
        start_t = time.time()
        train_file = Path(train_path or "data/processed/train.csv")
        val_file = Path(val_path or "data/processed/val.csv")

        if not train_file.exists() or not val_file.exists():
            raise FileNotFoundError(f"Training dataset partitions missing! Ensure {train_file} and {val_file} exist.")

        logger.info(f"Loading training data from {train_file} and validation data from {val_file}...")
        df_train = robust_read_csv(train_file)
        df_val = robust_read_csv(val_file)

        predictors = self.feat_reg.get_random_forest_predictors()
        
        # Inject Assignment Group for SLA context during resolution time regression
        if "assignment_group" not in predictors:
            predictors.append("assignment_group")
            
        self._verify_no_target_leakage(predictors)

        X_train = self._get_safe_predictor_matrix(df_train, predictors)
        y_train_raw = pd.to_numeric(df_train[target_col], errors="coerce").fillna(4.0)
        y_train_log = np.log1p(np.clip(y_train_raw, 0, 168))  # log1p transformation

        X_val = self._get_safe_predictor_matrix(df_val, predictors)
        y_val_raw = pd.to_numeric(df_val[target_col], errors="coerce").fillna(4.0)
        y_val_log = np.log1p(np.clip(y_val_raw, 0, 168))

        if compare_baselines:
            pipelines_dict, best_name = self.train_baselines_and_compare(X_train, y_train_log, X_val, y_val_log, predictors, target_type=target_col)
            primary_pipeline = pipelines_dict.get("CatBoost", pipelines_dict[best_name])
        else:
            prep = self.build_preprocessing_pipeline(X_train, predictors)
            X_train_trans = prep.fit_transform(X_train, y_train_log)
            text_idx = X_train_trans.shape[1] - 1
            
            rf_cfg = self.cfg.get("models.resolution_time.params", {})
            estimator = CatBoostRegressor(
                iterations=rf_cfg.get("n_estimators", 150),
                depth=rf_cfg.get("max_depth", 6),
                learning_rate=0.1,
                verbose=0,
                random_seed=42,
                text_features=[text_idx]
            )
            primary_pipeline = Pipeline([("preprocessing", prep), ("estimator", estimator)])
            primary_pipeline.fit(X_train, y_train_log)

        # Evaluate on validation fold (inverse transform log1p via expm1)
        val_preds_log = primary_pipeline.predict(X_val)
        val_preds_inv = np.expm1(val_preds_log)
        y_val_inv = np.expm1(y_val_log)

        rmse = float(np.sqrt(mean_squared_error(y_val_inv, val_preds_inv)))
        mae = float(mean_absolute_error(y_val_inv, val_preds_inv))
        r2 = float(r2_score(y_val_inv, val_preds_inv))
        train_dur = time.time() - start_t

        out_path = self.models_dir / f"catboost_{target_col}.pkl"
        joblib.dump(primary_pipeline, out_path)
        logger.info(f"Successfully persisted complete regression pipeline to: {out_path}")

        hyperparams = primary_pipeline.named_steps["estimator"].get_params() if "estimator" in primary_pipeline.named_steps else {}
        hyperparams_clean = {k: str(v) if not isinstance(v, (int, float, str, bool, type(None))) else v for k, v in hyperparams.items()}

        self.model_reg.register_model(
            model_name=f"catboost_{target_col}",
            version="v1.0.0",
            training_dataset_uri=str(train_file),
            dataset_version="v2.0.0-alpha",
            hyperparameters=hyperparams_clean,
            metrics={"rmse": round(rmse, 4), "mae": round(mae, 4), "r2": round(r2, 4), "training_duration_sec": round(train_dur, 2)},
            features_used=predictors,
            target_variable=target_col,
            model_file_path=str(out_path),
            status="Active"
        )

        logger.info(f"Registered model random_forest_{target_col}:v1.0.0 (RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f})")
        return out_path
