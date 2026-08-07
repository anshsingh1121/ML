"""
Enterprise Feature Engineering Engine (`src/preprocessing/engineer.py`).

Generates production-ready temporal, interaction, text-statistical, linkage, and
historical frequency features from cleaned incident datasets.
Automatically registers all newly created attributes inside the Central Feature Registry
and records exact mathematical/logical transformation graphs in the Feature Lineage Tracker.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import math
import numpy as np
import pandas as pd

from src.data.feature_registry import FeatureDefinition, FeatureRegistry
from src.data.feature_lineage import FeatureLineageTracker
from src.utils.logger import get_logger

logger = get_logger(__name__)


# US Federal & Banking Holidays Month-Day pairs (configurable)
DEFAULT_BANKING_HOLIDAYS = {
    (1, 1),   # New Year's Day
    (1, 15),  # MLK Day (approximate exact date matching or general mid-Jan)
    (2, 19),  # Washington's Birthday (approx)
    (5, 27),  # Memorial Day (approx)
    (6, 19),  # Juneteenth National Independence Day
    (7, 4),   # Independence Day
    (9, 2),   # Labor Day (approx)
    (10, 14), # Columbus Day / Indigenous Peoples' Day (approx)
    (11, 11), # Veterans Day
    (11, 28), # Thanksgiving Day (approx)
    (12, 25)  # Christmas Day
}


class FeatureEngineeringEngine:
    """
    Automated feature generation and domain transformation engine.
    Produces high-signal tabular features and guarantees synchronized governance
    by updating FeatureRegistry and FeatureLineageTracker upon every transformation.
    """

    def __init__(
        self,
        registry: Optional[FeatureRegistry] = None,
        lineage: Optional[FeatureLineageTracker] = None,
        holidays: Optional[set] = None
    ) -> None:
        """Initialize engine with Registry and Lineage tracker singletons."""
        self.registry = registry or FeatureRegistry.get_instance()
        self.lineage = lineage or FeatureLineageTracker.get_instance()
        self.holidays = holidays or DEFAULT_BANKING_HOLIDAYS

    def engineer_features(
        self,
        df: pd.DataFrame,
        output_dir: str = "reports"
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Execute comprehensive feature engineering pipeline and sync governance registries.

        Args:
            df: Cleaned input pandas DataFrame.
            output_dir: Directory where `feature_engineering_report.md` & `.json` will be saved.

        Returns:
            Tuple[pd.DataFrame, Dict[str, Any]]: (Feature-engineered DataFrame, Engineering Report Dict)
        """
        logger.info(f"Initiating Enterprise Feature Engineering across {len(df):,} records...")
        eng_df = df.copy()
        report: Dict[str, Any] = {
            "initial_feature_count": len(eng_df.columns),
            "new_features_created": [],
            "importance_recommendations": []
        }

        # 1. Datetime & Cyclic Features
        eng_df = self._generate_datetime_features(eng_df, report)

        # 2. Configurable Holiday & Business Hours Indicators
        eng_df = self._generate_holiday_and_business_features(eng_df, report)

        # 3. Interaction & Multiplicative Features
        eng_df = self._generate_interaction_features(eng_df, report)

        # 4. Resolution Duration & Outcome Features
        eng_df = self._generate_resolution_features(eng_df, report)

        # 5. Text Statistics Features
        eng_df = self._generate_text_statistics(eng_df, report)

        # 6. Duplicate & Linkage Boolean Flags
        eng_df = self._generate_linkage_flags(eng_df, report)

        # 7. Historical Frequency Encodings
        eng_df = self._generate_frequency_encodings(eng_df, report)

        # Compile final report summary
        report["final_feature_count"] = len(eng_df.columns)
        report["total_new_features"] = len(report["new_features_created"])
        report["status"] = "CERTIFIED_ENGINEERED"

        # Generate importance recommendations
        report["importance_recommendations"] = self._compute_importance_recommendations(report["new_features_created"])

        # Export reports
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        json_file = out_path / "feature_engineering_report.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Exported feature engineering JSON report to: {json_file}")

        md_file = out_path / "feature_engineering_report.md"
        self._export_markdown_report(report, md_file)
        logger.info(f"Exported feature engineering Markdown report to: {md_file}")

        return eng_df, report

    def _register_new_feature(
        self,
        name: str,
        business_name: str,
        data_type: str,
        meaning: str,
        leakage: str,
        encoding: str,
        rf_usage: str,
        formula: str,
        parents: List[str],
        report: Dict[str, Any]
    ) -> None:
        """Helper to automatically sync FeatureRegistry and FeatureLineageTracker."""
        # 1. Sync Feature Registry
        feat_def = FeatureDefinition(
            business_name=business_name,
            technical_name=name,
            data_type=data_type,
            nullable=False,
            cardinality="medium" if data_type == "float" else "low",
            missing_percentage=0.0,
            business_meaning=meaning,
            ml_importance="high" if rf_usage == "predictor" else "medium",
            target_leakage_classification=leakage,
            encoding_strategy=encoding,
            imputation_strategy="zero",
            scaling_strategy="standard" if data_type in ("float", "integer") else "none",
            feature_engineering_rules=formula,
            random_forest_usage=rf_usage,
            embedding_usage="excluded",
            faiss_metadata_usage="excluded",
            dashboard_usage="kpi_filter" if data_type == "integer" else "detail_table",
            api_exposure="response_only",
            future_rag_usage="excluded",
            explainability_usage="shap_feature_label" if rf_usage == "predictor" else "excluded",
            required_or_optional="optional",
            deprecated_status=False
        )
        self.registry.register_feature(feat_def)

        # 2. Sync Feature Lineage Tracker
        self.lineage.add_edge(
            source=parents,
            target=name,
            transformation=f"FeatureEngineeringEngine -> {encoding}",
            formula=formula
        )

        report["new_features_created"].append({
            "feature_name": name,
            "data_type": data_type,
            "leakage_classification": leakage,
            "formula": formula,
            "parents": parents
        })

    def _generate_datetime_features(self, df: pd.DataFrame, report: Dict[str, Any]) -> pd.DataFrame:
        """Extract temporal components and cyclic encodings from opened_at."""
        if "opened_at" not in df.columns:
            return df

        dt = pd.to_datetime(df["opened_at"], errors="coerce")
        if dt.isna().all():
            return df

        # Hour of day (0-23)
        df["opened_at_hour"] = dt.dt.hour.fillna(12).astype(int)
        self._register_new_feature(
            "opened_at_hour", "Opened Hour (0-23)", "integer", "Hour of incident creation",
            "safe", "ordinal", "predictor", "opened_at.dt.hour", ["opened_at"], report
        )

        # Day of week (0=Mon, 6=Sun)
        df["opened_at_dayofweek"] = dt.dt.dayofweek.fillna(0).astype(int)
        self._register_new_feature(
            "opened_at_dayofweek", "Opened Day of Week (0-6)", "integer", "Day of week of creation",
            "safe", "ordinal", "predictor", "opened_at.dt.dayofweek", ["opened_at"], report
        )

        # Month (1-12)
        df["opened_at_month"] = dt.dt.month.fillna(1).astype(int)
        self._register_new_feature(
            "opened_at_month", "Opened Month (1-12)", "integer", "Month of creation",
            "safe", "ordinal", "predictor", "opened_at.dt.month", ["opened_at"], report
        )

        # Weekend indicator
        df["is_weekend"] = df["opened_at_dayofweek"].isin([5, 6]).astype(int)
        self._register_new_feature(
            "is_weekend", "Weekend Indicator", "integer", "1 if Saturday or Sunday else 0",
            "safe", "ordinal", "predictor", "1 if dayofweek in [5, 6] else 0", ["opened_at_dayofweek"], report
        )

        # Cyclic Sine/Cosine encodings
        df["opened_at_hour_sin"] = np.sin(2 * np.pi * df["opened_at_hour"] / 24.0)
        df["opened_at_hour_cos"] = np.cos(2 * np.pi * df["opened_at_hour"] / 24.0)
        self._register_new_feature(
            "opened_at_hour_sin", "Opened Hour Sine", "float", "Cyclic sine encoding of hour",
            "safe", "sine_cosine", "predictor", "sin(2 * pi * hour / 24)", ["opened_at_hour"], report
        )
        self._register_new_feature(
            "opened_at_hour_cos", "Opened Hour Cosine", "float", "Cyclic cosine encoding of hour",
            "safe", "sine_cosine", "predictor", "cos(2 * pi * hour / 24)", ["opened_at_hour"], report
        )

        df["opened_at_dayofweek_sin"] = np.sin(2 * np.pi * df["opened_at_dayofweek"] / 7.0)
        df["opened_at_dayofweek_cos"] = np.cos(2 * np.pi * df["opened_at_dayofweek"] / 7.0)
        self._register_new_feature(
            "opened_at_dayofweek_sin", "Opened Day of Week Sine", "float", "Cyclic sine encoding of weekday",
            "safe", "sine_cosine", "predictor", "sin(2 * pi * dayofweek / 7)", ["opened_at_dayofweek"], report
        )
        self._register_new_feature(
            "opened_at_dayofweek_cos", "Opened Day of Week Cosine", "float", "Cyclic cosine encoding of weekday",
            "safe", "sine_cosine", "predictor", "cos(2 * pi * dayofweek / 7)", ["opened_at_dayofweek"], report
        )
        return df

    def _generate_holiday_and_business_features(self, df: pd.DataFrame, report: Dict[str, Any]) -> pd.DataFrame:
        """Compute business hours flag and US Federal/Banking Holiday indicator."""
        if "opened_at" not in df.columns or "opened_at_hour" not in df.columns:
            return df

        dt = pd.to_datetime(df["opened_at"], errors="coerce")

        # Business hours flag (Mon-Fri 08:00 - 18:00)
        df["is_business_hours"] = ((df["opened_at_dayofweek"] < 5) & (df["opened_at_hour"] >= 8) & (df["opened_at_hour"] <= 18)).astype(int)
        self._register_new_feature(
            "is_business_hours", "Business Hours Indicator", "integer", "1 if Mon-Fri 08:00-18:00 else 0",
            "safe", "ordinal", "predictor", "1 if weekday < 5 and 8 <= hour <= 18 else 0", ["opened_at_dayofweek", "opened_at_hour"], report
        )

        # Holiday indicator
        month_day_series = list(zip(dt.dt.month.fillna(0).astype(int), dt.dt.day.fillna(0).astype(int)))
        df["is_holiday"] = [1 if md in self.holidays else 0 for md in month_day_series]
        self._register_new_feature(
            "is_holiday", "US Banking Holiday Indicator", "integer", "1 if New Year's, MLK, Memorial, July 4, Labor, Thanksgiving, or Christmas else 0",
            "safe", "ordinal", "predictor", "1 if (month, day) in US_BANKING_HOLIDAYS else 0", ["opened_at"], report
        )
        return df

    def _generate_interaction_features(self, df: pd.DataFrame, report: Dict[str, Any]) -> pd.DataFrame:
        """Create high-signal cross-interactions between priority, impact, urgency, and categories."""
        if "priority" in df.columns and "impact" in df.columns:
            df["priority_x_impact"] = df["priority"].astype(int) * df["impact"].astype(int)
            self._register_new_feature(
                "priority_x_impact", "Priority x Impact Interaction", "integer", "Multiplicative interaction score",
                "safe", "ordinal", "predictor", "priority * impact", ["priority", "impact"], report
            )

        if "priority" in df.columns and "urgency" in df.columns:
            df["priority_x_urgency"] = df["priority"].astype(int) * df["urgency"].astype(int)
            self._register_new_feature(
                "priority_x_urgency", "Priority x Urgency Interaction", "integer", "Multiplicative interaction score",
                "safe", "ordinal", "predictor", "priority * urgency", ["priority", "urgency"], report
            )

        if "category" in df.columns and "assignment_group" in df.columns:
            df["category_assignment_interaction"] = df["category"].astype(str) + "_" + df["assignment_group"].astype(str)
            self._register_new_feature(
                "category_assignment_interaction", "Category & Group Composite String", "string", "Composite string for specific historical group routing",
                "safe", "one_hot", "excluded", "category + '_' + assignment_group", ["category", "assignment_group"], report
            )
        return df

    def _generate_resolution_features(self, df: pd.DataFrame, report: Dict[str, Any]) -> pd.DataFrame:
        """Compute exact resolution duration in hours and strictly classify as BLOCKED target leakage."""
        if "opened_at" in df.columns and "resolved_at" in df.columns:
            op = pd.to_datetime(df["opened_at"], errors="coerce")
            res = pd.to_datetime(df["resolved_at"], errors="coerce")
            df["resolution_time_hours"] = ((res - op).dt.total_seconds() / 3600.0).round(4).fillna(0.0)
            
            # Ensure no negative durations exist after cleaning
            df.loc[df["resolution_time_hours"] < 0, "resolution_time_hours"] = 0.0

            self._register_new_feature(
                "resolution_time_hours", "Exact Resolution Time (Hours)", "float", "Total duration from opened to resolved",
                "blocked", "none", "target_resolution_time", "(resolved_at - opened_at) in hours", ["opened_at", "resolved_at"], report
            )
        return df

    def _generate_text_statistics(self, df: pd.DataFrame, report: Dict[str, Any]) -> pd.DataFrame:
        """Extract word count and character count metrics across description fields."""
        if "short_description" in df.columns:
            series = df["short_description"].fillna("").astype(str)
            df["short_description_word_count"] = series.apply(lambda x: len(x.split()))
            df["short_description_char_count"] = series.apply(len)
            self._register_new_feature(
                "short_description_word_count", "Short Description Word Count", "integer", "Number of words in short description",
                "safe", "ordinal", "predictor", "len(short_description.split())", ["short_description"], report
            )
            self._register_new_feature(
                "short_description_char_count", "Short Description Char Count", "integer", "Number of characters in short description",
                "safe", "ordinal", "predictor", "len(short_description)", ["short_description"], report
            )

        if "description" in df.columns:
            series = df["description"].fillna("").astype(str)
            df["description_word_count"] = series.apply(lambda x: len(x.split()))
            df["description_char_count"] = series.apply(len)
            self._register_new_feature(
                "description_word_count", "Description Word Count", "integer", "Number of words in full description",
                "safe", "ordinal", "predictor", "len(description.split())", ["description"], report
            )
            self._register_new_feature(
                "description_char_count", "Description Char Count", "integer", "Number of characters in full description",
                "safe", "ordinal", "predictor", "len(description)", ["description"], report
            )
        return df

    def _generate_linkage_flags(self, df: pd.DataFrame, report: Dict[str, Any]) -> pd.DataFrame:
        """Create explicit 0/1 boolean linkage indicators."""
        if "duplicate_incident" in df.columns:
            df["is_duplicate"] = df["duplicate_incident"].notna() & (df["duplicate_incident"] != "")
            df["is_duplicate"] = df["is_duplicate"].astype(int)
            self._register_new_feature(
                "is_duplicate", "Duplicate Incident Flag", "integer", "1 if duplicate incident reference linked else 0",
                "warning", "ordinal", "predictor", "1 if duplicate_incident not empty else 0", ["duplicate_incident"], report
            )

        if "parent_incident" in df.columns:
            df["has_parent_incident"] = df["parent_incident"].notna() & (df["parent_incident"] != "")
            df["has_parent_incident"] = df["has_parent_incident"].astype(int)
            self._register_new_feature(
                "has_parent_incident", "Parent Incident Link Flag", "integer", "1 if child of parent incident else 0",
                "safe", "ordinal", "predictor", "1 if parent_incident not empty else 0", ["parent_incident"], report
            )

        if "problem_record" in df.columns or "problem_flag" in df.columns:
            prob_series = df["problem_flag"] if "problem_flag" in df.columns else df["problem_record"].notna()
            df["has_problem_record"] = prob_series.astype(int)
            self._register_new_feature(
                "has_problem_record", "Problem Record Link Flag", "integer", "1 if linked to problem record else 0",
                "safe", "ordinal", "predictor", "1 if problem_record/flag true else 0", ["problem_flag" if "problem_flag" in df.columns else "problem_record"], report
            )

        if "change_request" in df.columns:
            df["has_change_request"] = df["change_request"].notna() & (df["change_request"] != "")
            df["has_change_request"] = df["has_change_request"].astype(int)
            self._register_new_feature(
                "has_change_request", "Change Request Link Flag", "integer", "1 if caused by change request else 0",
                "safe", "ordinal", "predictor", "1 if change_request not empty else 0", ["change_request"], report
            )

        if "knowledge_linked" in df.columns or "knowledge_base" in df.columns:
            kb_series = df["knowledge_linked"] if "knowledge_linked" in df.columns else df["knowledge_base"].notna()
            df["has_knowledge_link"] = kb_series.astype(int)
            self._register_new_feature(
                "has_knowledge_link", "Knowledge Base Link Flag", "integer", "1 if linked to KB article else 0",
                "safe", "ordinal", "predictor", "1 if knowledge_linked true else 0", ["knowledge_linked" if "knowledge_linked" in df.columns else "knowledge_base"], report
            )
        return df

    def _generate_frequency_encodings(self, df: pd.DataFrame, report: Dict[str, Any]) -> pd.DataFrame:
        """Encode high-cardinality categorical strings into normalized historical frequency probabilities."""
        freq_cols = ["assignment_group", "business_service", "vendor", "caller", "location"]
        for col in freq_cols:
            if col in df.columns and len(df) > 0:
                freq_map = df[col].value_counts(normalize=True).to_dict()
                freq_name = f"{col}_freq"
                df[freq_name] = df[col].map(freq_map).fillna(0.0).round(4)
                self._register_new_feature(
                    freq_name, f"{col.title()} Historical Frequency", "float", f"Historical frequency probability of {col}",
                    "safe", "frequency", "predictor", f"df[{col}].map(df[{col}].value_counts(normalize=True))", [col], report
                )
        return df

    def _compute_importance_recommendations(self, new_features: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Recommend specific features for downstream model inputs and explain rationale."""
        recs = []
        for f in new_features:
            name = f["feature_name"]
            leakage = f["leakage_classification"]
            if leakage == "blocked":
                recs.append({
                    "feature": name,
                    "recommendation": "EXCLUDE from initial triage assignment classification model.",
                    "rationale": "Classified as BLOCKED target leakage per Feature Registry. This outcome is only known after ticket resolution."
                })
            elif "sin" in name or "cos" in name or "hour" in name or "dayofweek" in name or "business_hours" in name:
                recs.append({
                    "feature": name,
                    "recommendation": "INCLUDE in Random Forest classifier.",
                    "rationale": "Captures cyclic arrival mechanics and shiftwork patterns essential for routing network/batch job failures."
                })
            elif "_x_" in name or "_freq" in name or "word_count" in name or "has_" in name:
                recs.append({
                    "feature": name,
                    "recommendation": "INCLUDE in Random Forest classifier.",
                    "rationale": "High-signal engineered indicator that directly separates complex technical tiers without high cardinality noise."
                })
        return recs

    def _export_markdown_report(self, report: Dict[str, Any], md_file: Path) -> None:
        """Export formal markdown feature engineering audit report."""
        lines = [
            "# Enterprise Feature Engineering & Registry Synchronization Report (`v2.0.0-alpha`)",
            "",
            "**Organization:** First Citizens Bank — Enterprise Technology Division  ",
            f"**Initial Feature Count:** `{report['initial_feature_count']}` attributes  ",
            f"**Final Feature Count:** `{report['final_feature_count']}` attributes  ",
            f"**Total Newly Engineered Features:** `{report['total_new_features']}`  ",
            f"**Certification Status:** `{report['status']}`",
            "",
            "---",
            "",
            "## 1. Newly Engineered Feature Catalog & Lineage Sync",
            "",
            "Every newly engineered feature was automatically registered inside `FeatureRegistry` and `FeatureLineageTracker` across all 22 governance dimensions:",
            "",
            "| Feature Name (`technical_name`) | Data Type | Leakage Tier | Exact Derivation Formula | Parent Dependency |",
            "|---|---|---|---|---|"
        ]

        for item in report["new_features_created"]:
            parents_str = ", ".join([f"`{p}`" for p in item["parents"]])
            lines.append(f"| `{item['feature_name']}` | `{item['data_type']}` | **{item['leakage_classification'].upper()}** | `{item['formula']}` | {parents_str} |")

        lines.extend([
            "",
            "---",
            "",
            "## 2. ML Feature Importance Recommendations",
            "",
            "| Feature | Recommended Action for Phase 3 (Random Forest) | Enterprise Rationale |",
            "|---|---|---|"
        ])

        for rec in report["importance_recommendations"][:15]:
            lines.append(f"| `{rec['feature']}` | **{rec['recommendation']}** | {rec['rationale']} |")

        lines.extend([
            "",
            "---",
            "",
            "## 3. Governance Certification Summary",
            f"All `{report['total_new_features']}` newly created columns have been synchronized across `FeatureRegistry.get_instance()` and `FeatureLineageTracker.get_instance()`. Downstream models (`Random Forest`, `FAISS`) can now query these features dynamically without hardcoding."
        ])

        with open(md_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
