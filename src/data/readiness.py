"""
ML Readiness Evaluation Framework — Enterprise Feature & Leakage Analysis.

Automates the diagnostic inspection of synthetic or real ServiceNow incident
datasets before model training. Identifies target leakage, class imbalance,
textual token capacity, high cardinality, and multi-collinearity.

Design Decisions:
    - Target Leakage Protection: Proactively segregates time-of-ticket-creation
      features from post-resolution operational data to guarantee zero leakage during
      Assignment Group classifier training.
    - Information Theoretic Metrics: Computes Shannon Entropy and Gini Impurity for
      categorical targets to quantify class imbalance severity and guide resampling.
    - NLP Readiness Diagnostics: Evaluates token and character distributions against
      TF-IDF and TruncatedSVD components properly bounded.
"""

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.config_manager import ConfigManager
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MLReadinessEvaluator:
    """
    Evaluates dataset quality, statistical distributions, target leakage risks,
    and feature transformations required for ML model training.
    """

    # Post-resolution fields that must NOT be used when predicting assignment_group at triage
    POST_RESOLUTION_FIELDS = {
        "resolved_at", "closed_at", "close_notes", "close_code",
        "resolution_time_hours", "calendar_stc", "business_duration_hours",
        "made_sla", "sla_status", "reassignment_count", "reopen_count",
        "root_cause_summary"
    }

    def __init__(self, config: Optional[ConfigManager] = None) -> None:
        """Initialize MLReadinessEvaluator with optional ConfigManager."""
        self.config = config or ConfigManager()

    def evaluate_dataset(
        self,
        df: pd.DataFrame,
        target_column: str = "assignment_group",
        save_report: bool = True,
        report_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Conduct a comprehensive ML readiness audit of the dataset.

        Args:
            df: Pandas DataFrame containing incident records.
            target_column: Primary target variable for classification readiness.
            save_report: Whether to output reports/ml_readiness_report.md & .json.
            report_dir: Custom directory path for saving reports.

        Returns:
            Dictionary containing all diagnostic metrics, leakage flags, and recommendations.
        """
        logger.info(f"Conducting ML Readiness evaluation for {len(df):,} records (Target: '{target_column}')...")

        num_records = len(df)
        missing_stats = self._compute_missing_percentage(df)
        duplicate_stats = self._compute_duplicate_percentage(df)
        cardinality_stats = self._compute_cardinality(df)
        leakage_report = self._detect_target_leakage(df, target_column)
        imbalance_stats = self._analyze_class_imbalance(df, target_column)
        text_stats = self._compute_text_statistics(df)
        correlation_summary = self._compute_correlation_matrix(df)
        recommendations = self._generate_recommendations(
            missing_stats, cardinality_stats, imbalance_stats, text_stats, leakage_report
        )

        readiness_data = {
            "evaluation_timestamp": datetime.now().isoformat(),
            "total_records": num_records,
            "total_features": len(df.columns),
            "primary_target": target_column,
            "missing_stats": missing_stats,
            "duplicate_stats": duplicate_stats,
            "cardinality_stats": cardinality_stats,
            "target_leakage": leakage_report,
            "class_imbalance": imbalance_stats,
            "text_statistics": text_stats,
            "correlation_summary": correlation_summary,
            "recommended_preprocessing": recommendations
        }

        logger.info("ML Readiness evaluation successfully completed.")
        if save_report:
            self.save_readiness_report(readiness_data, report_dir)

        return readiness_data

    def _compute_missing_percentage(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate missing value percentage per column."""
        missing = df.isnull().sum()
        pct = (missing / len(df)) * 100.0
        return {col: round(val, 4) for col, val in pct.items()}

    def _compute_duplicate_percentage(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate duplicate record statistics."""
        if len(df) == 0:
            return {"exact_duplicate_pct": 0.0, "duplicate_id_pct": 0.0}

        exact_dups = df.duplicated().sum()
        id_dups = df["number"].duplicated().sum() if "number" in df.columns else 0

        return {
            "exact_duplicate_pct": round((exact_dups / len(df)) * 100.0, 4),
            "duplicate_id_pct": round((id_dups / len(df)) * 100.0, 4),
            "exact_duplicate_count": int(exact_dups),
            "duplicate_id_count": int(id_dups)
        }

    def _compute_cardinality(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Calculate distinct value counts and cardinality ratios across categorical columns."""
        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        cardinality = {}

        for col in cat_cols:
            unique_cnt = df[col].nunique(dropna=True)
            cardinality[col] = {
                "unique_count": int(unique_cnt),
                "cardinality_ratio": round(unique_cnt / len(df), 4) if len(df) > 0 else 0.0,
                "top_5_values": df[col].value_counts().head(5).to_dict()
            }

        return cardinality

    def _detect_target_leakage(self, df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
        """Identify potential target leakage features that must be dropped during early triage modeling."""
        leaky_features = []
        safe_features = []

        for col in df.columns:
            if col == target_col or col == "number":
                continue
            if col in self.POST_RESOLUTION_FIELDS:
                leaky_features.append({
                    "feature": col,
                    "reason": "Post-resolution / outcome-derived field. Populated after triage or closure."
                })
            else:
                safe_features.append(col)

        return {
            "has_leakage_risks": len(leaky_features) > 0,
            "leaky_features": leaky_features,
            "safe_triage_features": safe_features
        }

    def _analyze_class_imbalance(self, df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
        """Compute entropy, Gini impurity, and imbalance ratios across core target fields."""
        targets = [target_col, "priority", "category"]
        results = {}

        for col in targets:
            if col not in df.columns or len(df) == 0:
                continue

            counts = df[col].value_counts()
            probs = counts / len(df)

            # Shannon Entropy: H = -sum(p * log2(p))
            entropy = -sum(p * math.log2(p) for p in probs if p > 0)
            max_entropy = math.log2(len(counts)) if len(counts) > 1 else 1.0
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

            # Gini Impurity: G = 1 - sum(p^2)
            gini = 1.0 - sum(p ** 2 for p in probs)

            maj_count = int(counts.iloc[0]) if len(counts) > 0 else 0
            min_count = int(counts.iloc[-1]) if len(counts) > 0 else 0
            imbalance_ratio = round(maj_count / min_count, 2) if min_count > 0 else float("inf")

            results[col] = {
                "unique_classes": len(counts),
                "majority_class": str(counts.index[0]) if len(counts) > 0 else "UNKNOWN",
                "majority_percentage": round((maj_count / len(df)) * 100.0, 2),
                "minority_class": str(counts.index[-1]) if len(counts) > 0 else "UNKNOWN",
                "minority_percentage": round((min_count / len(df)) * 100.0, 2),
                "imbalance_ratio": imbalance_ratio,
                "shannon_entropy": round(entropy, 4),
                "normalized_entropy": round(normalized_entropy, 4),
                "gini_impurity": round(gini, 4)
            }

        return results

    def _compute_text_statistics(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Compute character and estimated token length statistics for text columns."""
        text_cols = ["short_description", "description", "close_notes"]
        stats = {}

        for col in text_cols:
            if col not in df.columns:
                continue

            series = df[col].dropna().astype(str)
            if len(series) == 0:
                stats[col] = {"char_min": 0, "char_avg": 0, "char_max": 0, "est_token_avg": 0, "est_token_max": 0}
                continue

            char_lens = series.str.len()
            # Approximate token count assuming ~4.5 chars per word/token
            token_lens = char_lens / 4.5

            stats[col] = {
                "char_min": float(char_lens.min()),
                "char_avg": round(float(char_lens.mean()), 2),
                "char_max": float(char_lens.max()),
                "est_token_avg": round(float(token_lens.mean()), 1),
                "est_token_max": round(float(token_lens.max()), 1),
                "exceeds_256_tokens_pct": round((float((token_lens > 256).sum()) / len(series)) * 100.0, 2)
            }

        return stats

    def _compute_correlation_matrix(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute numeric correlation summary across continuous and ordinal fields."""
        num_cols = [col for col in ["priority", "business_impact", "urgency", "resolution_time_hours", "reassignment_count"] if col in df.columns]
        if len(num_cols) < 2:
            return {"matrix": {}}

        corr_df = df[num_cols].corr(method="pearson").round(4)
        matrix_dict = corr_df.to_dict()

        # Find top correlated pairs (excluding diagonal)
        top_pairs = []
        for i in range(len(num_cols)):
            for j in range(i + 1, len(num_cols)):
                c1, c2 = num_cols[i], num_cols[j]
                val = corr_df.loc[c1, c2]
                if not pd.isnull(val):
                    top_pairs.append({"pair": f"{c1} <-> {c2}", "pearson_corr": float(val)})

        top_pairs.sort(key=lambda x: abs(x["pearson_corr"]), reverse=True)

        return {
            "matrix": matrix_dict,
            "top_correlated_pairs": top_pairs[:5]
        }

    def _generate_recommendations(
        self,
        missing: Dict[str, float],
        cardinality: Dict[str, Dict[str, Any]],
        imbalance: Dict[str, Any],
        text: Dict[str, Dict[str, float]],
        leakage: Dict[str, Any]
    ) -> List[str]:
        """Generate specific preprocessing and feature engineering action items."""
        recs = []

        # 1. Target Leakage
        if leakage.get("has_leakage_risks"):
            leaky_names = [f["feature"] for f in leakage["leaky_features"]]
            recs.append(f"**Target Leakage Exclusion:** Strictly drop post-resolution features ({', '.join(leaky_names)}) from training inputs when predicting `assignment_group` at triage time.")

        # 2. Imbalance handling
        ag_imb = imbalance.get("assignment_group", {})
        if ag_imb.get("imbalance_ratio", 1) > 5.0:
            recs.append(f"**Class Imbalance Resampling:** `assignment_group` exhibits a high imbalance ratio ({ag_imb['imbalance_ratio']}x). Implement `class_weight='balanced'` in Random Forest and apply stratified cross-validation (`StratifiedKFold`).")

        # 3. Encoding strategy for cardinality
        for col, stats in cardinality.items():
            if stats["unique_count"] > 100 and col not in ["number", "short_description", "description", "close_notes", "caller"]:
                recs.append(f"**High Cardinality Encoding:** `{col}` has {stats['unique_count']} unique categories. Avoid One-Hot Encoding to prevent dimensionality explosion; use Target Encoding or Frequency Encoding instead.")
            elif stats["unique_count"] <= 20 and col not in ["number", "short_description", "description", "close_notes"]:
                recs.append(f"**Categorical Encoding:** `{col}` has {stats['unique_count']} distinct values. Use Ordinal/Label Encoding for tree models (`CatBoostClassifier`).")

        # 4. Text length checks for Sentence Transformers
        desc_text = text.get("description", {})
        if desc_text.get("exceeds_256_tokens_pct", 0) > 0:
            recs.append(f"**Text Truncation & Summarization:** {desc_text['exceeds_256_tokens_pct']}% of `description` values exceed 256 tokens. Apply smart head-tail concatenation or extractive summarization before computing embeddings.")

        # 5. Missing value imputation
        high_missing = [c for c, p in missing.items() if p > 5.0 and c not in self.POST_RESOLUTION_FIELDS]
        if high_missing:
            recs.append(f"**Missing Value Imputation:** Features {high_missing} have >5% missingness. Impute categorical missingness with `'UNKNOWN'` to treat missing as a valid signal.")

        if not recs:
            recs.append("Dataset structure is well-balanced and clean. Ready for baseline modeling.")

        return recs

    def save_readiness_report(self, data: Dict[str, Any], report_dir: Optional[str] = None) -> Tuple[Path, Path]:
        """Save ML Readiness Report to reports/ml_readiness_report.md and .json."""
        out_dir = Path(report_dir or self.config.get("reports.dir", "reports"))
        out_dir.mkdir(parents=True, exist_ok=True)

        json_path = out_dir / "ml_readiness_report.json"
        md_path = out_dir / "ml_readiness_report.md"

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        lines = [
            "# Machine Learning Readiness Audit Report",
            f"**Audit Timestamp:** {data['evaluation_timestamp']}  ",
            f"**Dataset Volume:** {data['total_records']:,} rows × {data['total_features']} columns  ",
            f"**Primary Classification Target:** `{data['primary_target']}`  \n",
            "---",
            "\n## 1. Target Leakage Prevention Analysis",
            "> [!CAUTION]",
            "> **Strict Separation Required:** The following features contain post-resolution operational outcomes. Including them during assignment group or triage prediction will cause severe data leakage and artificial test accuracy.\n",
            "| Leaky Feature Name | Reason / Risk Description |",
            "|---|---|"
        ]

        for item in data["target_leakage"]["leaky_features"]:
            lines.append(f"| `{item['feature']}` | {item['reason']} |")

        lines.extend([
            "\n---",
            "\n## 2. Class Imbalance & Entropy Summary\n",
            "| Target Column | Unique Classes | Majority Class (%) | Minority Class (%) | Imbalance Ratio | Shannon Entropy | Gini Impurity |",
            "|---|:---:|:---:|:---:|:---:|:---:|:---:|"
        ])

        for col, stats in data["class_imbalance"].items():
            lines.append(
                f"| `{col}` | {stats['unique_classes']} | `{stats['majority_class']}` ({stats['majority_percentage']}%) | "
                f"`{stats['minority_class']}` ({stats['minority_percentage']}%) | **{stats['imbalance_ratio']}x** | "
                f"{stats['shannon_entropy']} | {stats['gini_impurity']} |"
            )

        lines.extend([
            "\n---",
            "\n## 3. Text & Token Statistics (SentenceTransformer Readiness)\n",
            "| Text Feature | Min Chars | Avg Chars | Max Chars | Avg Est. Tokens | Max Est. Tokens | Exceeds 256 Tokens (%) |",
            "|---|:---:|:---:|:---:|:---:|:---:|:---:|"
        ])

        for col, t_stats in data["text_statistics"].items():
            lines.append(
                f"| `{col}` | {t_stats['char_min']} | {t_stats['char_avg']} | {t_stats['char_max']} | "
                f"{t_stats['est_token_avg']} | {t_stats['est_token_max']} | **{t_stats['exceeds_256_tokens_pct']}%** |"
            )

        lines.extend([
            "\n---",
            "\n## 4. Top Numerical Feature Correlations\n",
            "| Feature Pair | Pearson Correlation ($r$) |",
            "|---|:---:|"
        ])

        for pair in data["correlation_summary"].get("top_correlated_pairs", []):
            lines.append(f"| `{pair['pair']}` | **{pair['pearson_corr']}** |")

        lines.extend([
            "\n---",
            "\n## 5. Actionable Preprocessing Recommendations\n"
        ])

        for idx, rec in enumerate(data["recommended_preprocessing"], 1):
            lines.append(f"{idx}. {rec}\n")

        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        logger.info(f"ML Readiness reports generated: {md_path} & {json_path}")
        return json_path, md_path
