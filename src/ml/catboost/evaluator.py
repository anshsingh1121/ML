"""
Enterprise Model Evaluation & Feature Importance Engine (`v1.5.0`).

Evaluates complete scikit-learn `Pipeline` objects across test partitions (`data/processed/test.csv`).
Computes multi-class Top-K accuracy, weighted/macro precision/recall/F1, ROC-AUC (`ovr`),
regression RMSE/MAE/$R^2$, Confusion Matrix/ROC charts (`reports/`), and Feature Importance rankings.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/CLI compatibility
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from src.utils import robust_read_csv
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    top_k_accuracy_score,
)
from sklearn.preprocessing import label_binarize

from src.data.feature_registry import FeatureRegistry
from src.ml.model_registry import ModelRegistry
from src.utils.config_manager import ConfigManager
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelEvaluator:
    """
    Enterprise ML Evaluator for classification (`assignment_group`) and regression (`resolution_time_hours`).
    Generates professional visual plots and structured Feature Importance audit artifacts.
    """

    def __init__(self, config_path: Optional[str] = None, reports_dir: Optional[Union[str, Path]] = None) -> None:
        self.cfg = ConfigManager(config_path)
        self.feat_reg = FeatureRegistry.get_instance()
        self.model_reg = ModelRegistry.get_instance()

        self.reports_dir = Path(reports_dir) if reports_dir is not None else Path("reports")
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def _get_safe_predictor_matrix(self, df: pd.DataFrame, predictors: List[str]) -> pd.DataFrame:
        """Ensure all predictor columns exist in the dataframe before slicing, initializing missing with safe defaults."""
        df_clean = df.copy()
        for col in predictors:
            # Exclude text NLP pipelines
            if col in ["short_description", "description", "u_describe_customer_impact", "close_notes"]:
                continue
            if col not in df_clean.columns:
                df_clean[col] = "UNKNOWN" if col in ["category", "subcategory", "cmdb_ci", "u_caused_by", "u_development_release_id", "u_vendor_ticket_ref"] else 0
        return df_clean[predictors]

    def _resolve_pipeline_and_test_data(
        self,
        model_key_or_path: Union[str, Path],
        test_path: Optional[str] = None,
        target_col: str = "assignment_group"
    ) -> Tuple[Any, pd.DataFrame, pd.Series, List[str]]:
        """Resolve model pipeline file path and load predictors/targets from test partition."""
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

        logger.info(f"Loading model pipeline from {model_file}...")
        pipeline = joblib.load(model_file)

        test_file = Path(test_path or "data/processed/test.csv")
        if not test_file.exists():
            raise FileNotFoundError(f"Test partition missing: {test_file}")

        df_test = robust_read_csv(test_file)
        predictors = self.feat_reg.get_catboost_predictors()
        X_test = self._get_safe_predictor_matrix(df_test, predictors)
        y_test = df_test[target_col]

        return pipeline, X_test, y_test, predictors

    def extract_and_export_feature_importances(
        self,
        pipeline: Any,
        predictors: List[str],
        target_col: str = "assignment_group"
    ) -> pd.DataFrame:
        """Extract tree feature importances, align with FeatureRegistry business names, and export CSV/MD/PNG."""
        estimator = pipeline.named_steps.get("estimator", pipeline)
        if not hasattr(estimator, "feature_importances_"):
            logger.warning(f"Estimator {type(estimator)} does not expose `feature_importances_`. Skipping feature importance ranking.")
            return pd.DataFrame()

        importances = estimator.feature_importances_
        # If preprocessing expanded or selected columns, match length cleanly
        n_feats = len(importances)
        try:
            if hasattr(pipeline, "named_steps") and "preprocessing" in pipeline.named_steps:
                prep_step = pipeline.named_steps["preprocessing"]
                if hasattr(prep_step, "named_steps") and "col_transform" in prep_step.named_steps:
                    feat_names = prep_step.named_steps["col_transform"].get_feature_names_out().tolist()
                elif hasattr(prep_step, "get_feature_names_out"):
                    feat_names = prep_step.get_feature_names_out().tolist()
                else:
                    feat_names = [f"feature_{i}" for i in range(n_feats)]
            elif hasattr(pipeline, "named_steps") and "col_transform" in pipeline.named_steps:
                feat_names = pipeline.named_steps["col_transform"].get_feature_names_out().tolist()
            elif hasattr(pipeline, "get_feature_names_out"):
                feat_names = pipeline.get_feature_names_out().tolist()
            else:
                feat_names = [f"feature_{i}" for i in range(n_feats)]
        except Exception as e:
            logger.warning(f"Could not extract get_feature_names_out ({e}), falling back to positional indices.")
            feat_names = [f"feature_{i}" for i in range(n_feats)]

        rows = []
        for i, name in enumerate(feat_names):
            biz_name = self.feat_reg.resolve_business_name(name) if hasattr(self.feat_reg, "resolve_business_name") else name
            rows.append({
                "rank": 0,
                "technical_name": name,
                "business_name": biz_name,
                "importance_score": round(float(importances[i]), 6),
                "importance_pct": round(float(importances[i]) * 100.0, 2)
            })

        df_imp = pd.DataFrame(rows).sort_values(by="importance_score", ascending=False).reset_index(drop=True)
        df_imp["rank"] = df_imp.index + 1

        # Export CSV with PermissionError resilience
        csv_path = self.reports_dir / f"feature_importance_{target_col}.csv"
        try:
            df_imp.to_csv(csv_path, index=False)
        except PermissionError:
            csv_path = self.reports_dir / f"feature_importance_{target_col}_latest.csv"
            df_imp.to_csv(csv_path, index=False)
            logger.warning(f"Primary CSV file locked by another process; exported to {csv_path}")

        # Also save standard feature_importance.csv for default assignment_group
        if target_col == "assignment_group":
            try:
                df_imp.to_csv(self.reports_dir / "feature_importance.csv", index=False)
            except PermissionError:
                df_imp.to_csv(self.reports_dir / "feature_importance_latest.csv", index=False)

        # Export Markdown
        md_path = self.reports_dir / f"feature_importance_{target_col}.md"
        lines = [
            f"# Enterprise Feature Importance Ranking (`{target_col}`)",
            "**Organization:** First Citizens Bank — Incident Intelligence Platform (IIP)  ",
            f"**Total Predictors Evaluated:** `{len(df_imp)}`  \n",
            "---",
            "\n## Top 20 Contributing Attributes\n",
            "| Rank | Technical Name | Business Name | Importance Score | Relative Share (%) |",
            "|:---:|---|---|:---:|:---:|"
        ]

        for _, row in df_imp.head(20).iterrows():
            badge = "🔥 Top Driver" if row["rank"] <= 3 else f"#{row['rank']}"
            lines.append(f"| **{badge}** | `{row['technical_name']}` | {row['business_name']} | `{row['importance_score']:.6f}` | **{row['importance_pct']}%** |")

        try:
            with open(md_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
        except PermissionError:
            md_path_alt = self.reports_dir / f"feature_importance_{target_col}_latest.md"
            with open(md_path_alt, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))

        # Render Horizontal Bar PNG Chart
        plt.figure(figsize=(10, 6))
        top15 = df_imp.head(15)
        sns.barplot(x="importance_pct", y="business_name", hue="business_name", data=top15, palette="crest", legend=False)
        plt.title(f"Top 15 Feature Importances ({target_col})", fontsize=14, fontweight="bold")
        plt.xlabel("Relative Contribution Share (%)", fontsize=12)
        plt.ylabel("Business Attribute", fontsize=12)
        plt.tight_layout()

        png_path = self.reports_dir / f"feature_importance_{target_col}.png"
        try:
            plt.savefig(png_path, dpi=300)
        except PermissionError:
            png_path = self.reports_dir / f"feature_importance_{target_col}_latest.png"
            plt.savefig(png_path, dpi=300)
        if target_col == "assignment_group":
            try:
                plt.savefig(self.reports_dir / "feature_importance.png", dpi=300)
            except PermissionError:
                plt.savefig(self.reports_dir / "feature_importance_latest.png", dpi=300)
        plt.close()

        logger.info(f"Exported Feature Importance ranking across {len(df_imp)} predictors to {csv_path}, {md_path}, and {png_path}")
        return df_imp

    def evaluate_classification(
        self,
        model_key_or_path: Union[str, Path],
        test_path: Optional[str] = None,
        target_col: str = "assignment_group"
    ) -> Dict[str, Any]:
        """Run complete classification evaluation, rendering Confusion Matrix, ROC curves, and formal metrics."""
        pipeline, X_test, y_test, predictors = self._resolve_pipeline_and_test_data(model_key_or_path, test_path, target_col)
        y_test_str = y_test.astype(str)

        preds = pipeline.predict(X_test)
        acc = accuracy_score(y_test_str, preds)
        prec_macro = precision_score(y_test_str, preds, average="macro", zero_division=0)
        prec_weighted = precision_score(y_test_str, preds, average="weighted", zero_division=0)
        rec_macro = recall_score(y_test_str, preds, average="macro", zero_division=0)
        rec_weighted = recall_score(y_test_str, preds, average="weighted", zero_division=0)
        f1_mac = f1_score(y_test_str, preds, average="macro", zero_division=0)
        f1_weight = f1_score(y_test_str, preds, average="weighted", zero_division=0)

        top3_acc = acc
        roc_auc = 0.0
        classes = sorted(y_test_str.unique().tolist())

        if hasattr(pipeline, "predict_proba"):
            probs = pipeline.predict_proba(X_test)
            try:
                k_val = min(3, len(classes) - 1)
                if k_val >= 1 and probs.shape[1] > k_val:
                    top3_acc = top_k_accuracy_score(y_test_str, probs, k=k_val)
            except Exception as e:
                logger.debug(f"Could not calculate Top-K accuracy: {e}")

            try:
                if len(classes) == 2:
                    roc_auc = roc_auc_score(y_test_str, probs[:, 1])
                elif len(classes) > 2:
                    roc_auc = roc_auc_score(y_test_str, probs, multi_class="ovr", average="weighted")
            except Exception as e:
                logger.debug(f"Could not calculate multi-class ROC-AUC: {e}")

            # Plot ROC Curves for Top 3 most frequent classes
            plt.figure(figsize=(8, 6))
            try:
                y_bin = label_binarize(y_test_str, classes=classes)
                for i, cls_name in enumerate(classes[:5]):
                    if y_bin.shape[1] > i and probs.shape[1] > i:
                        fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
                        plt.plot(fpr, tpr, label=f"{cls_name[:15]}")
                plt.plot([0, 1], [0, 1], "k--", label="Random Chance")
                plt.title("Multi-Class ROC Curves (Top Squads)", fontsize=14, fontweight="bold")
                plt.xlabel("False Positive Rate", fontsize=12)
                plt.ylabel("True Positive Rate", fontsize=12)
                plt.legend(loc="lower right")
                plt.tight_layout()
                plt.savefig(self.reports_dir / "roc_curve.png", dpi=300)
            except Exception as e:
                logger.debug(f"ROC plot generation skipped: {e}")
            plt.close()

        # Plot Confusion Matrix
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_test_str, preds, labels=classes)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
        plt.title(f"Confusion Matrix ({target_col})", fontsize=14, fontweight="bold")
        plt.xlabel("Predicted Squad", fontsize=12)
        plt.ylabel("True Squad", fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(self.reports_dir / "confusion_matrix.png", dpi=300)
        plt.close()

        # Extract Feature Importances
        self.extract_and_export_feature_importances(pipeline, predictors, target_col)

        metrics = {
            "accuracy": round(float(acc), 4),
            "top_3_accuracy": round(float(top3_acc), 4),
            "precision_weighted": round(float(prec_weighted), 4),
            "precision_macro": round(float(prec_macro), 4),
            "recall_weighted": round(float(rec_weighted), 4),
            "recall_macro": round(float(rec_macro), 4),
            "f1_weighted": round(float(f1_weight), 4),
            "f1_macro": round(float(f1_mac), 4),
            "roc_auc_weighted": round(float(roc_auc), 4),
            "test_sample_count": len(y_test_str)
        }

        # Save formal report
        report_md = self.reports_dir / "classification_report.md"
        lines = [
            f"# Enterprise Classification Evaluation Report (`{target_col}`)",
            "**Organization:** First Citizens Bank — Incident Intelligence Platform (IIP)  ",
            f"**Evaluation Timestamp:** {datetime.now().isoformat()}  ",
            f"**Test Sample Size:** `{len(y_test_str):,}` records  \n",
            "---",
            "\n## Key Performance Indicators\n",
            "| Metric Name | Score | Banking SLA Interpretation |",
            "|---|:---:|---|",
            f"| **Top-1 Exact Accuracy** | `{metrics['accuracy'] * 100:.2f}%` | Percentage of tickets routed immediately to the exact right L1/L2 squad. |",
            f"| **Top-3 Triage Accuracy** | `{metrics['top_3_accuracy'] * 100:.2f}%` | Percentage where correct squad is within the top 3 AI recommendations. |",
            f"| **Weighted F1-Score** | `{metrics['f1_weighted']:.4f}` | Balanced harmonic mean across high-volume and specialized engineering squads. |",
            f"| **ROC-AUC (Weighted OVR)** | `{metrics['roc_auc_weighted']:.4f}` | Discriminative ranking capability across multi-class assignment boundaries. |\n",
            "---",
            "\n## Certified Visual Audit Charts\n",
            "- Confusion Matrix: `reports/confusion_matrix.png`",
            "- ROC Curves: `reports/roc_curve.png`",
            "- Feature Importance: `reports/feature_importance.png`"
        ]
        with open(report_md, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        with open(self.reports_dir / "classification_report.json", "w", encoding="utf-8") as f:
            json.dump({"target_variable": target_col, "metrics": metrics, "timestamp": datetime.now().isoformat()}, f, indent=2)

        logger.info(f"Classification evaluation complete! Acc={acc:.4f}, Top3Acc={top3_acc:.4f}, F1={f1_weight:.4f}")
        return metrics

    def evaluate_regression(
        self,
        model_key_or_path: Union[str, Path],
        test_path: Optional[str] = None,
        target_col: str = "resolution_time_hours"
    ) -> Dict[str, Any]:
        """Run complete regression evaluation (`MAE`, `RMSE`, `R2`) on actual resolution hours."""
        pipeline, X_test, y_test, predictors = self._resolve_pipeline_and_test_data(model_key_or_path, test_path, target_col)
        y_test_num = pd.to_numeric(y_test, errors="coerce").fillna(4.0)

        preds_raw = pipeline.predict(X_test)
        # If the model was trained on log1p targets, apply inverse expm1
        preds_inv = np.expm1(np.clip(preds_raw, 0, 15)) if preds_raw.max() < 15.0 else preds_raw

        mae = mean_absolute_error(y_test_num, preds_inv)
        mse = mean_squared_error(y_test_num, preds_inv)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test_num, preds_inv)

        self.extract_and_export_feature_importances(pipeline, predictors, target_col)

        metrics = {
            "mae_hours": round(float(mae), 4),
            "mse": round(float(mse), 4),
            "rmse_hours": round(float(rmse), 4),
            "r2_score": round(float(r2), 4),
            "test_sample_count": len(y_test_num)
        }

        report_md = self.reports_dir / "regression_report.md"
        lines = [
            f"# Enterprise Regression Evaluation Report (`{target_col}`)",
            "**Organization:** First Citizens Bank — Incident Intelligence Platform (IIP)  ",
            f"**Evaluation Timestamp:** {datetime.now().isoformat()}  ",
            f"**Test Sample Size:** `{len(y_test_num):,}` records  \n",
            "---",
            "\n## Regression Performance Metrics\n",
            "| Metric Name | Score | Operational Significance |",
            "|---|:---:|---|",
            f"| **Mean Absolute Error (MAE)** | `{metrics['mae_hours']} hours` | Average absolute deviation in predicting ticket resolution effort. |",
            f"| **Root Mean Squared Error (RMSE)** | `{metrics['rmse_hours']} hours` | Penalizes large estimation errors and SLA prediction misses. |",
            f"| **R² Variance Explained** | `{metrics['r2_score']:.4f}` | Proportion of MTTR variance explained by triage predictors. |\n",
            "---",
            "\n## Certified Visual Artifacts\n",
            f"- Feature Importance Ranking: `reports/feature_importance_{target_col}.png`"
        ]
        with open(report_md, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        with open(self.reports_dir / "regression_report.json", "w", encoding="utf-8") as f:
            json.dump({"target_variable": target_col, "metrics": metrics, "timestamp": datetime.now().isoformat()}, f, indent=2)

        logger.info(f"Regression evaluation complete! RMSE={rmse:.4f} hrs, MAE={mae:.4f} hrs, R2={r2:.4f}")
        return metrics
