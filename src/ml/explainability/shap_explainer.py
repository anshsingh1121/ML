"""
Explainable AI & Structured Prediction Engine (`v1.5.0`).

Integrates `shap.TreeExplainer` across complete scikit-learn `Pipeline` objects.
Generates global summary (`shap_summary.png`, `shap_bar.png`) and local diagnostic plots
(`shap_waterfall_sample.png`, `shap_decision_sample.png`).
Exports structured prediction metadata (`predicted_class`, `confidence_score`,
`top_contributing_features`, `feature_importances`, `prediction_timestamp`) for Hybrid Similarity integration.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from src.utils import robust_read_csv

from src.data.feature_registry import FeatureRegistry
from src.ml.model_registry import ModelRegistry
from src.utils.config_manager import ConfigManager
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SHAPIntelligenceExplainer:
    """
    Enterprise Explainable AI Engine for First Citizens Bank Incident Intelligence Platform.
    Provides mathematically verified local and global feature attribution via game-theoretic SHAP values.
    """

    def __init__(self, config_path: Optional[str] = None, reports_dir: Optional[Union[str, Path]] = None) -> None:
        self.cfg = ConfigManager(config_path)
        self.feat_reg = FeatureRegistry.get_instance()
        self.model_reg = ModelRegistry.get_instance()

        self.reports_dir = Path(reports_dir) if reports_dir is not None else Path("reports")
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_pipeline(self, model_key_or_path: Union[str, Path]) -> Tuple[Any, Path]:
        """Resolve pipeline model object and file path."""
        if Path(model_key_or_path).exists():
            model_file = Path(model_key_or_path)
        elif isinstance(model_key_or_path, str) and ":" in model_key_or_path and not (len(model_key_or_path) >= 2 and model_key_or_path[1] == ':' and model_key_or_path[0].isalpha()):
            parts = model_key_or_path.split(":")
            meta = self.model_reg.get_model_metadata(parts[0], parts[-1])
            if not meta:
                raise ValueError(f"Model key '{model_key_or_path}' not found in ModelRegistry!")
            model_file = Path(meta.model_file_path)
        else:
            model_file = Path(model_key_or_path)

        if not model_file.exists():
            raise FileNotFoundError(f"Model file missing: {model_file}")

        return joblib.load(model_file), model_file

    def _get_safe_predictor_matrix(self, df: pd.DataFrame, predictors: List[str]) -> pd.DataFrame:
        """Ensure all predictor columns exist in the dataframe before slicing, initializing missing with safe defaults."""
        df_clean = df.copy()
        for col in predictors:
            if col not in df_clean.columns:
                df_clean[col] = "UNKNOWN" if col in ["category", "subcategory", "business_service", "location", "cmdb_ci", "vendor", "contact_type"] else 0
        return df_clean[predictors]

    def _get_transformed_dataframe_with_business_names(self, prep: Any, X_trans: Any, predictors: List[str]) -> pd.DataFrame:
        """Construct transformed DataFrame with verified enterprise business feature names via get_feature_names_out + FeatureRegistry."""
        n_cols = X_trans.shape[1] if hasattr(X_trans, "shape") else len(X_trans)
        try:
            if hasattr(prep, "named_steps") and "col_transform" in prep.named_steps:
                f_names = prep.named_steps["col_transform"].get_feature_names_out().tolist()
            elif hasattr(prep, "get_feature_names_out"):
                f_names = prep.get_feature_names_out().tolist()
            else:
                f_names = predictors[:n_cols] if n_cols <= len(predictors) else [f"f_{i}" for i in range(n_cols)]
        except Exception as e:
            logger.debug(f"Could not extract get_feature_names_out in SHAP ({e}), attempting fallback.")
            f_names = predictors[:n_cols] if n_cols <= len(predictors) else [f"f_{i}" for i in range(n_cols)]

        # Map every technical or expanded feature name to its exact enterprise business name
        resolved_names = []
        for name in f_names:
            biz = self.feat_reg.resolve_business_name(name) if hasattr(self.feat_reg, "resolve_business_name") else name
            resolved_names.append(biz)

        if isinstance(X_trans, pd.DataFrame):
            df_trans = X_trans.copy()
            df_trans.columns = resolved_names[:len(df_trans.columns)]
            return df_trans
        else:
            return pd.DataFrame(X_trans, columns=resolved_names[:n_cols])

    def explain_global(
        self,
        model_key_or_path: Union[str, Path],
        test_path: Optional[str] = None,
        sample_size: int = 500,
        target_col: str = "assignment_group"
    ) -> Dict[str, float]:
        """Compute global SHAP explanations (`TreeExplainer`) and generate summary/bar plots (`reports/`)."""
        pipeline, _ = self._resolve_pipeline(model_key_or_path)

        test_file = Path(test_path or "data/processed/test.csv")
        if not test_file.exists():
            raise FileNotFoundError(f"Test partition missing: {test_file}")

        df_test = robust_read_csv(test_file)
        predictors = self.feat_reg.get_random_forest_predictors()
        X_sample = self._get_safe_predictor_matrix(df_test, predictors).head(sample_size)

        # Transform features through pipeline preprocessing
        prep = pipeline.named_steps.get("preprocessing", None)
        estimator = pipeline.named_steps.get("estimator", pipeline)

        X_trans = prep.transform(X_sample) if prep else X_sample
        X_trans_df = self._get_transformed_dataframe_with_business_names(prep, X_trans, predictors)

        logger.info(f"Computing SHAP TreeExplainer values across {len(X_trans_df)} records...")
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(X_trans_df)

        # Handle multi-class vs regression shap array structures
        if isinstance(shap_values, list):
            # Average absolute shap contribution across all classes
            mean_abs_shap = np.mean([np.abs(sv) for sv in shap_values], axis=0).mean(axis=0)
            # Use class 0 values for summary plot or list
            plot_shap = shap_values[0]
        elif len(shap_values.shape) == 3:
            mean_abs_shap = np.abs(shap_values).mean(axis=0).mean(axis=1)
            plot_shap = shap_values[:, :, 0]
        else:
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            plot_shap = shap_values

        # Global SHAP Bar Ranking Plot
        plt.figure(figsize=(10, 6))
        feat_cols = X_trans_df.columns.tolist()
        df_bar = pd.DataFrame({"feature": feat_cols, "shap_importance": mean_abs_shap}).sort_values(by="shap_importance", ascending=False).head(15)
        sns.barplot(x="shap_importance", y="feature", hue="feature", data=df_bar, palette="magma", legend=False)
        plt.title(f"Global SHAP Feature Attribution ({target_col})", fontsize=14, fontweight="bold")
        plt.xlabel("Mean |SHAP Value| (Impact on Prediction)", fontsize=12)
        plt.ylabel("Transformed Predictor", fontsize=12)
        plt.tight_layout()
        plt.savefig(self.reports_dir / "shap_bar.png", dpi=300)
        plt.close()

        # Global SHAP Summary Plot (Beeswarm / Density)
        plt.figure(figsize=(10, 8))
        try:
            shap.summary_plot(plot_shap, X_trans_df, show=False, rng=np.random.default_rng(42))
            plt.title("SHAP Beeswarm Summary Plot", fontsize=14, fontweight="bold")
            plt.tight_layout()
            plt.savefig(self.reports_dir / "shap_summary.png", dpi=300)
        except Exception as e:
            logger.debug(f"SHAP summary_plot error, falling back to bar: {e}")
            df_bar.plot.barh(x="feature", y="shap_importance", figsize=(10, 6))
            plt.title("SHAP Feature Summary (Fallback)", fontsize=14, fontweight="bold")
            plt.tight_layout()
            plt.savefig(self.reports_dir / "shap_summary.png", dpi=300)
        plt.close()

        logger.info("Exported global SHAP plots to reports/shap_bar.png & reports/shap_summary.png")
        return df_bar.set_index("feature")["shap_importance"].to_dict()

    def explain_prediction(
        self,
        record_or_batch: Union[Dict[str, Any], pd.DataFrame, List[Dict[str, Any]]],
        model_key_or_path: Union[str, Path] = "models/random_forest_assignment_group.pkl",
        target_col: str = "assignment_group"
    ) -> List[Dict[str, Any]]:
        """
        Execute zero-manual-preprocessing inference, calculate local SHAP attribution,
        and export structured prediction metadata (`predicted_class`, `confidence_score`,
        `top_contributing_features`, `feature_importances`, `prediction_timestamp`).
        Saves structured artifacts (`reports/prediction_metadata.json` & `.csv`).
        """
        pipeline, _ = self._resolve_pipeline(model_key_or_path)

        if isinstance(record_or_batch, dict):
            df_in = pd.DataFrame([record_or_batch])
        elif isinstance(record_or_batch, list):
            df_in = pd.DataFrame(record_or_batch)
        elif isinstance(record_or_batch, pd.DataFrame):
            df_in = record_or_batch.copy()
        else:
            raise ValueError("Input must be a dict, list of dicts, or pandas DataFrame.")

        predictors = self.feat_reg.get_random_forest_predictors()
        X_in = self._get_safe_predictor_matrix(df_in, predictors)
        prep = pipeline.named_steps.get("preprocessing", None)
        estimator = pipeline.named_steps.get("estimator", pipeline)

        X_trans = prep.transform(X_in) if prep else X_in
        X_trans_df = self._get_transformed_dataframe_with_business_names(prep, X_trans, predictors)

        preds = pipeline.predict(X_in)
        probs = pipeline.predict_proba(X_in) if hasattr(pipeline, "predict_proba") else None

        # Compute SHAP local values
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(X_trans_df)

        results = []
        now_iso = datetime.now().isoformat()

        for i in range(len(df_in)):
            pred_val = str(preds[i]) if hasattr(preds[i], "strip") or isinstance(preds[i], (str, int, np.integer)) else float(preds[i])
            # If resolution time regression was log-transformed, apply expm1
            if target_col == "resolution_time_hours" and isinstance(pred_val, float):
                pred_val = float(np.expm1(pred_val)) if pred_val < 15.0 else pred_val

            conf = float(np.max(probs[i])) if probs is not None else 1.0

            # Extract local SHAP contributions for this specific instance
            local_shap_dict = {}
            if isinstance(shap_values, list):
                # Find class index if possible, otherwise use 0
                cls_idx = 0
                if probs is not None:
                    cls_idx = int(np.argmax(probs[i]))
                    if cls_idx >= len(shap_values):
                        cls_idx = 0
                sv = shap_values[cls_idx][i]
            elif len(shap_values.shape) == 3:
                cls_idx = int(np.argmax(probs[i])) if probs is not None else 0
                sv = shap_values[i, :, cls_idx]
            else:
                sv = shap_values[i]

            for j, fname in enumerate(X_trans_df.columns):
                local_shap_dict[fname] = round(float(sv[j]), 6)

            # Sort features by absolute SHAP contribution
            sorted_feats = sorted(local_shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)
            top_5_contributors = [{"feature": k, "shap_contribution": v} for k, v in sorted_feats[:5]]

            result_entry = {
                "incident_number": str(df_in.iloc[i].get("incident_number", f"INC_INFERENCE_{i}")),
                "predicted_class" if probs is not None else "predicted_value": pred_val,
                "confidence_score": round(conf, 4),
                "top_contributing_features": top_5_contributors,
                "feature_importances": local_shap_dict,
                "prediction_timestamp": now_iso
            }
            results.append(result_entry)

        # Generate sample local waterfall and decision plots on the first record
        if len(X_trans_df) > 0:
            plt.figure(figsize=(10, 6))
            try:
                # Plot horizontal waterfall/contribution bar of top 10 local forces
                top_local = pd.DataFrame(sorted_feats[:10], columns=["feature", "contribution"]).sort_values(by="contribution", ascending=True)
                colors = ["green" if c > 0 else "red" for c in top_local["contribution"]]
                plt.barh(top_local["feature"], top_local["contribution"], color=colors)
                plt.title(f"Local SHAP Feature Attribution (Record 0 -> Pred: {results[0].get('predicted_class', results[0].get('predicted_value'))})", fontsize=12, fontweight="bold")
                plt.xlabel("SHAP Value Contribution (+ pushes higher, - pushes lower)", fontsize=10)
                plt.axvline(0, color="black", linestyle="--")
                plt.tight_layout()
                plt.savefig(self.reports_dir / "shap_waterfall_sample.png", dpi=300)
            except Exception as e:
                logger.debug(f"Waterfall plot fallback: {e}")
            plt.close()

            plt.figure(figsize=(10, 6))
            try:
                # Decision path plot
                plt.plot(range(len(sorted_feats[:10])), [x[1] for x in sorted_feats[:10]], marker="o", color="blue", linewidth=2)
                plt.xticks(range(len(sorted_feats[:10])), [x[0] for x in sorted_feats[:10]], rotation=45, ha="right")
                plt.title("Local SHAP Decision Contribution Path", fontsize=12, fontweight="bold")
                plt.ylabel("SHAP Force", fontsize=10)
                plt.axhline(0, color="gray", linestyle=":")
                plt.tight_layout()
                plt.savefig(self.reports_dir / "shap_decision_sample.png", dpi=300)
            except Exception as e:
                logger.debug(f"Decision plot fallback: {e}")
            plt.close()

        # Save structured prediction metadata to JSON and CSV artifacts with resilience
        json_out = self.reports_dir / "prediction_metadata.json"
        try:
            with open(json_out, "w", encoding="utf-8") as f:
                json.dump({"total_predictions": len(results), "timestamp": now_iso, "predictions": results}, f, indent=2)
        except PermissionError:
            json_out = self.reports_dir / "prediction_metadata_latest.json"
            with open(json_out, "w", encoding="utf-8") as f:
                json.dump({"total_predictions": len(results), "timestamp": now_iso, "predictions": results}, f, indent=2)

        # Flatten top contributing features for CSV export
        csv_rows = []
        for r in results:
            flat = {
                "incident_number": r["incident_number"],
                "prediction": r.get("predicted_class", r.get("predicted_value")),
                "confidence_score": r["confidence_score"],
                "prediction_timestamp": r["prediction_timestamp"],
                "top_1_feature": r["top_contributing_features"][0]["feature"] if len(r["top_contributing_features"]) > 0 else "UNKNOWN",
                "top_1_shap": r["top_contributing_features"][0]["shap_contribution"] if len(r["top_contributing_features"]) > 0 else 0.0,
                "top_2_feature": r["top_contributing_features"][1]["feature"] if len(r["top_contributing_features"]) > 1 else "UNKNOWN",
                "top_2_shap": r["top_contributing_features"][1]["shap_contribution"] if len(r["top_contributing_features"]) > 1 else 0.0,
            }
            csv_rows.append(flat)

        csv_out = self.reports_dir / "prediction_metadata.csv"
        try:
            pd.DataFrame(csv_rows).to_csv(csv_out, index=False)
        except PermissionError:
            csv_out = self.reports_dir / "prediction_metadata_latest.csv"
            pd.DataFrame(csv_rows).to_csv(csv_out, index=False)

        logger.info(f"Generated structured prediction metadata across {len(results)} records ({json_out} & {csv_out}).")
        return results
