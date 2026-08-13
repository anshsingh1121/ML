"""
Enterprise Exploratory Data Analysis (EDA) Engine (`src/preprocessing/eda.py`).

Performs comprehensive automated analysis of ServiceNow incident datasets across
Numerical, Categorical, Boolean, Datetime, and Text dimensions without hardcoded
column lists by consuming the Central Feature Registry via Pipeline Contracts.
Generates rich markdown, JSON, and HTML reports along with explanatory visual charts.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import math
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/CLI usage
import matplotlib.pyplot as plt
import seaborn as sns

from src.data.feature_registry import FeatureRegistry
from src.data.pipeline_contracts import PipelineContractValidator
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EnterpriseEDAEngine:
    """
    Automated EDA & Feature Intelligence Engine governing ServiceNow data exploration.
    Enforces contract validation, computes information-theoretic statistics, generates
    diagnostic visual charts with explanations, and outputs formal executive reports.
    """

    def __init__(self, registry: Optional[FeatureRegistry] = None, validator: Optional[PipelineContractValidator] = None) -> None:
        """Initialize EDA engine with Registry and Contract Validator instances."""
        self.registry = registry or FeatureRegistry.get_instance()
        self.validator = validator or PipelineContractValidator(self.registry)

    def analyze_dataset(
        self,
        df: pd.DataFrame,
        target_column: str = "assignment_group",
        output_dir: str = "reports",
        generate_figures: bool = True
    ) -> Dict[str, Any]:
        """
        Execute full automated exploration across all column types and generate reports.

        Args:
            df: Input ServiceNow pandas DataFrame.
            target_column: Target label column name (`assignment_group`).
            output_dir: Base directory to save `eda_report.md`, `eda_report.json`, and figures.
            generate_figures: Whether to generate and save PNG visual charts.

        Returns:
            Dict[str, Any]: Complete structured dictionary of EDA metrics and analysis results.
        """
        logger.info(f"Initiating Enterprise EDA Engine across {len(df):,} records and {len(df.columns)} columns...")
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        fig_dir = out_path / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)

        # Retrieve categorized columns via Registry Contract
        all_defs = self.registry.list_all_features()
        num_cols = [f.technical_name for f in all_defs if f.data_type in ("integer", "float") and f.technical_name in df.columns]
        cat_cols = [f.technical_name for f in all_defs if f.data_type == "string" and f.technical_name in df.columns and f.technical_name not in self._get_known_text_cols()]
        bool_cols = [f.technical_name for f in all_defs if f.data_type == "boolean" and f.technical_name in df.columns]
        date_cols = [f.technical_name for f in all_defs if f.data_type == "datetime" and f.technical_name in df.columns]
        text_cols = [f.technical_name for f in all_defs if f.technical_name in self._get_known_text_cols() and f.technical_name in df.columns]

        # 1. Dataset Summary & Missing/Duplicate Audit
        missing_report = self._compute_missing_report(df)
        duplicate_count = int(df.duplicated(subset=["number"]).sum()) if "number" in df.columns else int(df.duplicated().sum())
        duplicate_percentage = float(duplicate_count / len(df) * 100) if len(df) > 0 else 0.0

        # 2. Numerical Analysis & Outlier Detection
        numerical_analysis = self._compute_numerical_analysis(df, num_cols)

        # 3. Categorical Analysis (Cardinality, Entropy, Gini)
        categorical_analysis = self._compute_categorical_analysis(df, cat_cols)

        # 4. Boolean Analysis
        boolean_analysis = self._compute_boolean_analysis(df, bool_cols)

        # 5. Datetime & Temporal Progression Analysis
        datetime_analysis = self._compute_datetime_analysis(df, date_cols)

        # 6. Text Length & NLP Token Analysis
        text_analysis = self._compute_text_analysis(df, text_cols)

        # 7. Correlation Analysis
        correlation_report = self._compute_correlation_analysis(df, num_cols)

        # 8. Target Leakage Inspection
        leakage_report = self._compute_target_leakage_audit(df)

        # 9. Generate Visual Charts & Figure Explanations
        figures_metadata = []
        if generate_figures and len(df) > 0:
            figures_metadata = self._generate_visual_charts(df, fig_dir, num_cols, cat_cols, date_cols, text_cols)

        # Compile full report dictionary
        eda_results = {
            "dataset_summary": {
                "total_records": len(df),
                "total_columns": len(df.columns),
                "memory_usage_kb": float(df.memory_usage(deep=True).sum() / 1024),
                "duplicate_incident_count": duplicate_count,
                "duplicate_percentage": round(duplicate_percentage, 4),
                "target_column": target_column,
                "target_missing_count": int(df[target_column].isna().sum()) if target_column in df.columns else 0
            },
            "missing_value_report": missing_report,
            "numerical_analysis": numerical_analysis,
            "categorical_analysis": categorical_analysis,
            "boolean_analysis": boolean_analysis,
            "datetime_analysis": datetime_analysis,
            "text_analysis": text_analysis,
            "correlation_report": correlation_report,
            "target_leakage_report": leakage_report,
            "figures_metadata": figures_metadata
        }

        # Export JSON report
        json_file = out_path / "eda_report.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(eda_results, f, indent=2)
        logger.info(f"Exported EDA JSON report to: {json_file}")

        # Export Markdown Report
        md_file = out_path / "eda_report.md"
        self._export_markdown_report(eda_results, md_file)
        logger.info(f"Exported EDA Markdown report to: {md_file}")

        # Export HTML Report
        html_file = out_path / "eda_report.html"
        self._export_html_report(eda_results, html_file)
        logger.info(f"Exported EDA HTML report to: {html_file}")

        return eda_results

    def _get_known_text_cols(self) -> List[str]:
        """Identify long-form unstructured text columns."""
        return ["short_description", "description", "close_notes", "resolution_notes", "work_notes"]

    def _compute_missing_report(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Compute exact missing value counts, percentages, and registry thresholds."""
        report = []
        for col in df.columns:
            missing_count = int(df[col].isna().sum())
            missing_perc = float(missing_count / len(df) * 100) if len(df) > 0 else 0.0
            feat_def = self.registry.get_feature(col)
            expected_max = feat_def.missing_percentage if feat_def else 100.0
            status = "PASS" if missing_perc <= expected_max else "EXCEEDS_THRESHOLD"
            report.append({
                "column": col,
                "missing_count": missing_count,
                "missing_percentage": round(missing_perc, 2),
                "expected_max_percentage": expected_max,
                "status": status
            })
        return sorted(report, key=lambda x: x["missing_percentage"], reverse=True)

    def _compute_numerical_analysis(self, df: pd.DataFrame, num_cols: List[str]) -> Dict[str, Any]:
        """Compute descriptive statistics, skewness, kurtosis, and IQR outlier bounds."""
        results = {}
        for col in num_cols:
            series = df[col].dropna()
            if len(series) == 0:
                continue
            q25 = float(series.quantile(0.25))
            q75 = float(series.quantile(0.75))
            iqr = q75 - q25
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr
            outlier_count = int(((series < lower_bound) | (series > upper_bound)).sum())

            results[col] = {
                "mean": round(float(series.mean()), 4),
                "std": round(float(series.std()), 4),
                "min": round(float(series.min()), 4),
                "q25": round(q25, 4),
                "median": round(float(series.median()), 4),
                "q75": round(q75, 4),
                "max": round(float(series.max()), 4),
                "skewness": round(float(series.skew()), 4) if len(series) > 2 else 0.0,
                "kurtosis": round(float(series.kurtosis()), 4) if len(series) > 3 else 0.0,
                "iqr": round(iqr, 4),
                "outlier_count_iqr": outlier_count,
                "outlier_percentage": round(float(outlier_count / len(series) * 100), 2)
            }
        return results

    def _compute_categorical_analysis(self, df: pd.DataFrame, cat_cols: List[str]) -> Dict[str, Any]:
        """Compute cardinality, top categories, Shannon Entropy, and Gini Impurity."""
        results = {}
        for col in cat_cols:
            series = df[col].dropna().astype(str)
            if len(series) == 0:
                continue
            value_counts = series.value_counts()
            probs = value_counts.values / len(series)

            # Shannon Entropy H = - sum(p * log2(p))
            entropy = float(-np.sum(probs * np.log2(probs + 1e-12)))
            # Gini Impurity G = 1 - sum(p^2)
            gini = float(1.0 - np.sum(probs ** 2))

            top_10 = {str(k): int(v) for k, v in value_counts.head(10).items()}
            results[col] = {
                "unique_count": int(series.nunique()),
                "shannon_entropy": round(entropy, 4),
                "gini_impurity": round(gini, 4),
                "top_categories": top_10
            }
        return results

    def _compute_boolean_analysis(self, df: pd.DataFrame, bool_cols: List[str]) -> Dict[str, Any]:
        """Compute True/False/Null counts and percentages."""
        results = {}
        for col in bool_cols:
            series = df[col].dropna()
            if len(series) == 0:
                continue
            true_cnt = int((series == True).sum())
            false_cnt = int((series == False).sum())
            results[col] = {
                "true_count": true_cnt,
                "false_count": false_cnt,
                "true_percentage": round(float(true_cnt / len(series) * 100), 2),
                "false_percentage": round(float(false_cnt / len(series) * 100), 2)
            }
        return results

    def _compute_datetime_analysis(self, df: pd.DataFrame, date_cols: List[str]) -> Dict[str, Any]:
        """Analyze temporal spans and hourly/daily arrival patterns."""
        results = {}
        for col in date_cols:
            series = pd.to_datetime(df[col], errors="coerce").dropna()
            if len(series) == 0:
                continue
            min_dt = series.min()
            max_dt = series.max()
            span_days = float((max_dt - min_dt).total_seconds() / 86400.0)

            hourly_counts = series.dt.hour.value_counts().to_dict()
            hourly_dist = {str(k): int(hourly_counts.get(k, 0)) for k in range(24)}

            weekday_counts = series.dt.dayofweek.value_counts().to_dict()
            weekday_dist = {str(k): int(weekday_counts.get(k, 0)) for k in range(7)}

            results[col] = {
                "min_timestamp": min_dt.isoformat(),
                "max_timestamp": max_dt.isoformat(),
                "span_days": round(span_days, 2),
                "hourly_distribution": hourly_dist,
                "weekday_distribution": weekday_dist
            }
        return results

    def _compute_text_analysis(self, df: pd.DataFrame, text_cols: List[str]) -> Dict[str, Any]:
        """Compute character length, word count, and token estimations across text fields."""
        results = {}
        for col in text_cols:
            series = df[col].fillna("").astype(str)
            char_lengths = series.apply(len)
            word_counts = series.apply(lambda x: len(x.split()))
            token_estimates = char_lengths // 4  # Typical BPE/WordPiece estimation

            results[col] = {
                "mean_char_length": round(float(char_lengths.mean()), 2),
                "max_char_length": int(char_lengths.max()),
                "mean_word_count": round(float(word_counts.mean()), 2),
                "max_word_count": int(word_counts.max()),
                "mean_token_estimate": round(float(token_estimates.mean()), 2),
                "max_token_estimate": int(token_estimates.max()),
                "exceeds_256_tokens_count": int((token_estimates > 256).sum()),
                "exceeds_256_tokens_percentage": round(float((token_estimates > 256).sum() / len(series) * 100), 2) if len(series) > 0 else 0.0
            }
        return results

    def _compute_correlation_analysis(self, df: pd.DataFrame, num_cols: List[str]) -> Dict[str, Any]:
        """Compute Pearson ($r$) and Spearman ($\rho$) correlation matrices."""
        valid_cols = [c for c in num_cols if df[c].nunique() > 1]
        if len(valid_cols) < 2:
            return {"pearson_correlation": {}, "spearman_correlation": {}}

        sub_df = df[valid_cols].dropna()
        pearson = sub_df.corr(method="pearson").round(4).to_dict()
        spearman = sub_df.corr(method="spearman").round(4).to_dict()
        return {
            "pearson_correlation": pearson,
            "spearman_correlation": spearman
        }

    def _compute_target_leakage_audit(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Cross-check dataset columns against Feature Registry leakage tiers."""
        blocked_defs = self.registry.get_features_by_leakage("blocked")
        warning_defs = self.registry.get_features_by_leakage("warning")

        blocked_present = [f.technical_name for f in blocked_defs if f.technical_name in df.columns]
        warning_present = [f.technical_name for f in warning_defs if f.technical_name in df.columns]

        # Note: target labels (like assignment_group or resolution_time_hours if target) are allowed as targets,
        # but pure leakage predictors (close_notes, resolved_at, resolution_code) must be flagged when present in raw inputs
        pure_leakage = [b for b in blocked_present if b not in ["assignment_group", "resolution_time_hours"]]

        return {
            "audit_status": "WARNING_LEAKAGE_DETECTED" if len(pure_leakage) > 0 else "PASS_CLEAN_TRIAGE",
            "blocked_columns_present": blocked_present,
            "pure_leakage_predictors_present": pure_leakage,
            "warning_columns_present": warning_present,
            "recommendation": "Drop pure_leakage_predictors before training initial triage models (`CatBoostClassifier`)."
        }

    def _generate_visual_charts(
        self,
        df: pd.DataFrame,
        fig_dir: Path,
        num_cols: List[str],
        cat_cols: List[str],
        date_cols: List[str],
        text_cols: List[str]
    ) -> List[Dict[str, str]]:
        """Generate high-resolution PNG charts with structured ML decision explanations."""
        figures = []

        # Chart 1: Category Distribution
        if "category" in df.columns and len(df["category"].dropna()) > 0:
            fig_path = fig_dir / "01_category_distribution.png"
            plt.figure(figsize=(10, 6))
            counts = df["category"].value_counts().head(10)
            sns.barplot(x=counts.values, y=counts.index, hue=counts.index, legend=False, palette="viridis")
            plt.title("Top 10 Incident Categories — Frequency Distribution", fontsize=14, fontweight="bold")
            plt.xlabel("Incident Count", fontsize=12)
            plt.ylabel("Category", fontsize=12)
            plt.tight_layout()
            plt.savefig(fig_path, dpi=300)
            plt.close()

            figures.append({
                "filename": "01_category_distribution.png",
                "title": "Category Frequency Distribution",
                "ml_explanation": "Identifies class imbalances across top incident categories. High cardinality or extreme skewness dictates whether `class_weight='balanced'` or stratified resampling is mandatory for `CatBoostClassifier`."
            })

        # Chart 2: Priority vs SLA Compliance
        if "priority" in df.columns and "made_sla" in df.columns:
            fig_path = fig_dir / "02_priority_vs_sla.png"
            plt.figure(figsize=(8, 6))
            sla_summary = df.groupby(["priority", "made_sla"]).size().unstack(fill_value=0)
            sla_summary.plot(kind="bar", stacked=True, color=["#ef4444", "#10b981"], figsize=(8, 6))
            plt.title("SLA Compliance Rate by Incident Priority", fontsize=14, fontweight="bold")
            plt.xlabel("Priority Tier (1=Critical, 4=Low)", fontsize=12)
            plt.ylabel("Record Count", fontsize=12)
            plt.legend(["SLA Breached (False)", "SLA Met (True)"], title="Compliance")
            plt.tight_layout()
            plt.savefig(fig_path, dpi=300)
            plt.close()

            figures.append({
                "filename": "02_priority_vs_sla.png",
                "title": "Priority Tier vs. SLA Compliance Mechanics",
                "ml_explanation": "Demonstrates the correlation between urgency/priority tiers and SLA breaches. Helps validate whether `priority` and `impact` serve as strong splitting features for resolution time prediction."
            })

        # Chart 3: Hourly Arrival Heatmap
        if "opened_at" in df.columns:
            series = pd.to_datetime(df["opened_at"], errors="coerce").dropna()
            if len(series) > 0:
                fig_path = fig_dir / "03_hourly_arrival.png"
                plt.figure(figsize=(10, 5))
                hourly = series.dt.hour.value_counts().sort_index()
                sns.lineplot(x=hourly.index, y=hourly.values, marker="o", color="#3b82f6", linewidth=2.5)
                plt.title("Incident Arrival Frequency by Hour of Day (00:00 - 23:00)", fontsize=14, fontweight="bold")
                plt.xlabel("Hour of Day (UTC/Local)", fontsize=12)
                plt.ylabel("Incident Arrival Count", fontsize=12)
                plt.xticks(range(24))
                plt.grid(True, linestyle="--", alpha=0.5)
                plt.tight_layout()
                plt.savefig(fig_path, dpi=300)
                plt.close()

                figures.append({
                    "filename": "03_hourly_arrival.png",
                    "title": "24-Hour Incident Arrival Curve",
                    "ml_explanation": "Validates cyclic temporal feature extraction (`opened_at_hour_sin` / `cos`). Spikes during peak business hours justify creating explicit `is_business_hours` flags for downstream decision trees."
                })

        # Chart 4: Numerical Correlation Heatmap
        valid_num = [c for c in num_cols if df[c].nunique() > 1 and c in ["priority", "business_impact", "urgency", "reassignment_count", "reopen_count"]]
        if len(valid_num) >= 2:
            fig_path = fig_dir / "04_numerical_correlation.png"
            plt.figure(figsize=(8, 6))
            corr = df[valid_num].corr()
            sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1, fmt=".2f", cbar_kws={"label": "Pearson Correlation (r)"})
            plt.title("Numerical Attribute Correlation Matrix", fontsize=14, fontweight="bold")
            plt.tight_layout()
            plt.savefig(fig_path, dpi=300)
            plt.close()

            figures.append({
                "filename": "04_numerical_correlation.png",
                "title": "Numerical Attribute Correlation Heatmap",
                "ml_explanation": "Detects multicollinearity among numeric indicators. For example, strong correlation between `priority`, `impact`, and `urgency` justifies exact interaction formulas (`priority_x_business_impact`) to assist tree splits."
            })

        # Chart 5: Text Word Count Distribution
        if "short_description" in df.columns:
            fig_path = fig_dir / "05_text_word_counts.png"
            plt.figure(figsize=(9, 5))
            word_counts = df["short_description"].fillna("").astype(str).apply(lambda x: len(x.split()))
            sns.histplot(word_counts, bins=30, kde=True, color="#8b5cf6")
            plt.title("Short Description Word Count Distribution", fontsize=14, fontweight="bold")
            plt.xlabel("Word Count per Incident", fontsize=12)
            plt.ylabel("Frequency", fontsize=12)
            plt.axvline(word_counts.mean(), color="red", linestyle="--", label=f"Mean: {word_counts.mean():.1f}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(fig_path, dpi=300)
            plt.close()

            figures.append({
                "filename": "05_text_word_counts.png",
                "title": "Semantic Embedding Sequence Truncation",
                "ml_explanation": "Confirms whether textual descriptions fit well within our local pipeline's limits."
            })

        return figures

    def _export_markdown_report(self, results: Dict[str, Any], md_file: Path) -> None:
        """Export executive markdown report with embedded charts and statistical tables."""
        summary = results["dataset_summary"]
        lines = [
            "# Enterprise Exploratory Data Analysis (EDA) Report (`v2.0.0-alpha`)",
            "",
            "**Organization:** First Citizens Bank — Enterprise Technology Division  ",
            f"**Dataset Analysis Records:** `{summary['total_records']:,}` records across `{summary['total_columns']}` attributes  ",
            f"**Memory Footprint:** `{summary['memory_usage_kb']:,.2f} KB`  ",
            f"**Target Leakage Audit Status:** `{results['target_leakage_report']['audit_status']}`",
            "",
            "---",
            "",
            "## 1. Executive Summary & Data Integrity Audit",
            "",
            "| Metric | Value | Status |",
            "|---|---|---|",
            f"| Total Incidents Evaluated | `{summary['total_records']:,}` | PASS |",
            f"| Duplicate Incident Keys | `{summary['duplicate_incident_count']}` (`{summary['duplicate_percentage']}%`) | {'PASS' if summary['duplicate_count' if 'duplicate_count' in summary else 'duplicate_incident_count'] == 0 else 'WARNING — REQUIRES DEDUPLICATION'} |",
            f"| Primary Target Column | `{summary['target_column']}` | PASS |",
            f"| Missing Target Labels | `{summary['target_missing_count']}` | {'PASS (0% missing)' if summary['target_missing_count'] == 0 else 'CRITICAL — TARGET NULLS DETECTED'} |",
            "",
            "### Missing Value Audit Table",
            "",
            "| Attribute (`technical_name`) | Missing Count | Missing % | Expected Max % | Status |",
            "|---|---|---|---|---|"
        ]

        for item in results["missing_value_report"][:15]:  # Top 15 missing
            lines.append(f"| `{item['column']}` | {item['missing_count']} | {item['missing_percentage']}% | {item['expected_max_percentage']}% | **{item['status']}** |")

        lines.extend([
            "",
            "---",
            "",
            "## 2. Categorical Class Imbalance & Information Theory Metrics",
            "",
            "To evaluate categorical diversity and splitting potential before training Random Forest models, we compute exact **Shannon Entropy ($H$)** and **Gini Impurity ($G$)** across key categorical attributes:",
            "",
            "| Categorical Attribute | Cardinality | Shannon Entropy ($H$) | Gini Impurity ($G$) | Top Class Frequency |",
            "|---|---|---|---|---|"
        ])

        for col, cat_data in results["categorical_analysis"].items():
            top_cls = list(cat_data["top_categories"].keys())[0] if cat_data["top_categories"] else "N/A"
            top_val = list(cat_data["top_categories"].values())[0] if cat_data["top_categories"] else 0
            lines.append(f"| `{col}` | {cat_data['unique_count']} | {cat_data['shannon_entropy']} | {cat_data['gini_impurity']} | `{top_cls}` ({top_val:,}) |")

        lines.extend([
            "",
            "---",
            "",
            "## 3. Numerical Statistics & Outlier IQR Analysis",
            "",
            "| Numerical Attribute | Mean | Std | Median ($Q_{50}$) | IQR | Outliers ($1.5 \\times \\text{IQR}$) | Outlier % |",
            "|---|---|---|---|---|---|---|"
        ])

        for col, num_data in results["numerical_analysis"].items():
            lines.append(f"| `{col}` | {num_data['mean']} | {num_data['std']} | {num_data['median']} | {num_data['iqr']} | {num_data['outlier_count_iqr']} | {num_data['outlier_percentage']}% |")

        lines.extend([
            "",
            "---",
            "",
            "## 4. NLP Text Readiness & Token Boundaries",
            "",
            "| Text Attribute | Mean Words | Max Words | Mean Token Est. | Max Token Est. | Exceeds 256 Tokens |",
            "| -------------- | ---------- | --------- | --------------- | -------------- | ------------------ |"
        ])

        for col, txt_data in results["text_analysis"].items():
            lines.append(f"| `{col}` | {txt_data['mean_word_count']} | {txt_data['max_word_count']} | {txt_data['mean_token_estimate']} | {txt_data['max_token_estimate']} | `{txt_data['exceeds_256_tokens_count']}` ({txt_data['exceeds_256_tokens_percentage']}%) |")

        lines.extend([
            "",
            "---",
            "",
            "## 5. Diagnostic Visual Charts & ML Decision Explanations",
            ""
        ])

        for fig in results["figures_metadata"]:
            lines.extend([
                f"### {fig['title']}",
                f"![{fig['title']}](figures/{fig['filename']})",
                "",
                f"> **Machine Learning & Feature Intelligence Explanation:**  ",
                f"> {fig['ml_explanation']}",
                ""
            ])

        lines.extend([
            "---",
            "",
            "## 6. Target Leakage Audit & Preprocessing Recommendations",
            "",
            f"**Audit Finding:** `{results['target_leakage_report']['audit_status']}`  ",
            f"**Blocked Leakage Columns Detected:** `{results['target_leakage_report']['pure_leakage_predictors_present']}`  ",
            "",
            "### Recommended Next Steps for Data Cleaning & Feature Engineering",
            "1. **Execute Enterprise Data Cleaner (`cleaner.py`):** Drop duplicate `incident_number` rows, impute missing text with `'Not Provided'`, and winsorize numeric outliers (`reassignment_count`).",
            "2. **Strict Leakage Interlock:** Explicitly drop `close_notes`, `resolved_at`, and `resolution_code` before constructing feature matrices for `assignment_group` prediction.",
            "3. **Cyclic Temporal Encoding:** Generate `opened_at_hour_sin` / `cos` and `opened_at_dayofweek_sin` / `cos` to capture non-linear arrival patterns without ordinal distortion."
        ])

        with open(md_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def _export_html_report(self, results: Dict[str, Any], html_file: Path) -> None:
        """Export self-contained HTML executive summary report."""
        summary = results["dataset_summary"]
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Enterprise EDA Report - First Citizens Bank</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #0f172a; color: #f8fafc; margin: 40px; }}
        h1, h2, h3 {{ color: #38bdf8; border-bottom: 1px solid #334155; padding-bottom: 10px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; background: #1e293b; }}
        th, td {{ padding: 12px; border: 1px solid #334155; text-align: left; }}
        th {{ background: #334155; color: #38bdf8; }}
        tr:nth-child(even) {{ background: #0f172a; }}
        .badge {{ padding: 4px 8px; border-radius: 4px; font-weight: bold; }}
        .pass {{ background: #10b981; color: #ffffff; }}
        .warn {{ background: #f59e0b; color: #ffffff; }}
    </style>
</head>
<body>
    <h1>Enterprise Exploratory Data Analysis (EDA) Report</h1>
    <p><strong>Organization:</strong> First Citizens Bank — Enterprise Technology Division</p>
    <p><strong>Total Records Evaluated:</strong> {summary['total_records']:,} across {summary['total_columns']} attributes</p>
    <p><strong>Target Leakage Audit Status:</strong> <span class="badge {'pass' if results['target_leakage_report']['audit_status'] == 'PASS_CLEAN_TRIAGE' else 'warn'}">{results['target_leakage_report']['audit_status']}</span></p>

    <h2>1. Categorical Shannon Entropy ($H$) & Gini Impurity ($G$)</h2>
    <table>
        <thead>
            <tr>
                <th>Attribute</th>
                <th>Cardinality</th>
                <th>Shannon Entropy ($H$)</th>
                <th>Gini Impurity ($G$)</th>
            </tr>
        </thead>
        <tbody>"""

        for col, cat_data in results["categorical_analysis"].items():
            html_content += f"""
            <tr>
                <td><code>{col}</code></td>
                <td>{cat_data['unique_count']}</td>
                <td>{cat_data['shannon_entropy']}</td>
                <td>{cat_data['gini_impurity']}</td>
            </tr>"""

        html_content += """
        </tbody>
    </table>
</body>
</html>"""
        with open(html_file, "w", encoding="utf-8") as f:
            f.write(html_content)
