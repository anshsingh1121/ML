"""
Enterprise Feature Registry — Single Source of Truth (`v1.5.0`).

Centralizes definitions, schemas, data types, leakage classifications, ML usages,
and transformation rules for every attribute (raw & derived) in the AI-Powered
Incident Intelligence Platform.

Downstream modules (Random Forest, SentenceTransformer, FAISS, Dashboard, API, RAG)
MUST consume this registry instead of relying on hardcoded feature lists.
"""

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(unsafe_hash=True)
class FeatureDefinition:
    """Formal specification for a single enterprise feature attribute across 22 governance dimensions."""
    business_name: str
    technical_name: str
    data_type: str
    nullable: bool
    cardinality: str
    missing_percentage: float
    business_meaning: str
    ml_importance: str
    target_leakage_classification: str
    encoding_strategy: str
    imputation_strategy: str
    scaling_strategy: str
    feature_engineering_rules: str
    random_forest_usage: str
    embedding_usage: str
    faiss_metadata_usage: str
    dashboard_usage: str
    api_exposure: str
    future_rag_usage: str
    explainability_usage: str
    required_or_optional: str
    deprecated_status: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert feature definition to dictionary."""
        return asdict(self)


class FeatureRegistry:
    """
    Singleton-style centralized Feature Registry managing all 38 raw ServiceNow columns
    and 11 derived engineering features.
    """

    _instance: Optional["FeatureRegistry"] = None
    _registry: Dict[str, FeatureDefinition] = {}

    def __init__(self, load_default: bool = True) -> None:
        """Initialize FeatureRegistry and populate with enterprise catalog if requested."""
        if load_default and not self._registry:
            self._populate_default_registry()

    @classmethod
    def get_instance(cls) -> "FeatureRegistry":
        """Get or create singleton FeatureRegistry instance."""
        if cls._instance is None:
            cls._instance = cls(load_default=True)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance and clear registry definitions (for unit testing)."""
        cls._instance = None
        cls._registry.clear()

    def register_feature(self, feature: FeatureDefinition) -> None:
        """Register or overwrite a feature definition."""
        self._registry[feature.technical_name] = feature
        logger.debug(f"Registered feature: '{feature.technical_name}' ({feature.business_name})")

    def get_feature(self, technical_name: str) -> Optional[FeatureDefinition]:
        """Retrieve a specific feature definition by technical name."""
        return self._registry.get(technical_name)

    def list_all_features(self) -> List[FeatureDefinition]:
        """List all registered feature definitions."""
        return list(self._registry.values())

    def get_features_by_usage(self, usage_field: str, target_value: str) -> List[FeatureDefinition]:
        """Filter features by a specific usage classification."""
        results = []
        for feat in self._registry.values():
            val = getattr(feat, usage_field, None)
            if val == target_value:
                results.append(feat)
        return results

    def get_features_by_leakage(self, classification: str) -> List[FeatureDefinition]:
        """Retrieve all features matching a leakage tier ('safe', 'warning', 'blocked')."""
        return self.get_features_by_usage("target_leakage_classification", classification.lower())

    def get_random_forest_predictors(self, target_type: str = "assignment_group") -> List[str]:
        """Retrieve safe predictor column names authorized for Random Forest training."""
        safe_feats = self.get_features_by_leakage("safe")
        results = []
        for feat in safe_feats:
            if feat.random_forest_usage == "predictor":
                results.append(feat.technical_name)
        return sorted(results)

    def get_embedding_features(self) -> List[str]:
        """Retrieve text column names authorized as primary/secondary embedding inputs."""
        results = []
        for feat in self._registry.values():
            if feat.embedding_usage in ["primary_semantic_input", "secondary_summary"]:
                results.append(feat.technical_name)
        return sorted(results)

    def get_faiss_metadata_features(self) -> List[str]:
        """Retrieve column names authorized as structural FAISS metadata filters/boosts."""
        results = []
        for feat in self._registry.values():
            if feat.faiss_metadata_usage in ["exact_match_filter", "structural_boost_tag"]:
                results.append(feat.technical_name)
        return sorted(results)

    def resolve_business_name(self, technical_name: str) -> str:
        """
        Resolve any raw or pipeline-transformed technical feature name to its exact enterprise business name.
        Handles one-hot expanded indicators (`category_Hardware`) and cyclic interaction terms (`priority_x_impact`).
        """
        import re
        # Strip any sklearn step prefix (e.g. freq__, onehot__, num__)
        clean_name = re.sub(r'^[a-zA-Z0-9]+__', '', technical_name)

        # 1. Exact match against FeatureRegistry
        feat_def = self.get_feature(clean_name)
        if feat_def is not None:
            return feat_def.business_name

        # 2. Check if it's a one-hot expanded attribute (match descending length base keys)
        sorted_keys = sorted(self._registry.keys(), key=len, reverse=True)
        for base_key in sorted_keys:
            if clean_name.startswith(f"{base_key}_"):
                base_feat = self.get_feature(base_key)
                if base_feat is not None:
                    value = clean_name[len(base_key) + 1:]
                    return f"{base_feat.business_name} = {value}"

        # 3. Handle dynamic interaction / cyclic terms cleanly if not explicitly registered
        if clean_name == "priority_x_impact":
            return "Priority x Impact Interaction Score"
        if clean_name == "priority_x_urgency":
            return "Priority x Urgency Interaction Score"
        if clean_name == "opened_at_hour_sin":
            return "Opened Hour Sine Cyclic"
        if clean_name == "opened_at_hour_cos":
            return "Opened Hour Cosine Cyclic"
        if clean_name == "opened_at_dayofweek_sin":
            return "Opened Day of Week Sine Cyclic"
        if clean_name == "opened_at_dayofweek_cos":
            return "Opened Day of Week Cosine Cyclic"

        # 4. Fallback: clean formatting without generic feature_x noise
        return clean_name.replace("_", " ").title()

    def export_json(self, output_path: Optional[str] = None) -> Path:
        """Export full feature registry to feature_registry.json."""
        out_file = Path(output_path or "reports/feature_registry.json")
        out_file.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "registry_version": "1.5.0",
            "last_updated": "2026-07-11T11:00:00Z",
            "total_features": len(self._registry),
            "features": {k: v.to_dict() for k, v in sorted(self._registry.items())}
        }

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        logger.info(f"Exported Feature Registry JSON to: {out_file}")
        return out_file

    def export_markdown(self, output_path: Optional[str] = None) -> Path:
        """Export full feature registry to feature_registry.md enterprise table."""
        out_file = Path(output_path or "reports/feature_registry.md")
        out_file.parent.mkdir(parents=True, exist_ok=True)

        lines = [
            "# Enterprise Feature Registry (`v1.5.0`)",
            "**Organization:** First Citizens Bank — Incident Intelligence Platform (IIP)  ",
            f"**Total Registered Attributes:** {len(self._registry)} (38 Raw ServiceNow Schema + 11 Derived ML Features)  ",
            "**Single Source of Truth Governance:** All downstream ML modules consume this registry contract.  \n",
            "---",
            "\n## Complete Attribute Registry Matrix\n",
            "| Technical Name | Business Name | Data Type | Nullable | Cardinality | Leakage Tier | RF Usage | Embedding Usage | FAISS Usage | RAG Usage |",
            "|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|"
        ]

        for k, feat in sorted(self._registry.items()):
            badge = {
                "safe": "🟢 Safe",
                "warning": "🟡 Warning",
                "blocked": "🔴 Blocked"
            }.get(feat.target_leakage_classification, feat.target_leakage_classification)

            lines.append(
                f"| `{feat.technical_name}` | {feat.business_name} | `{feat.data_type}` | "
                f"{'Yes' if feat.nullable else 'No'} | {feat.cardinality} | {badge} | "
                f"`{feat.random_forest_usage}` | `{feat.embedding_usage}` | `{feat.faiss_metadata_usage}` | `{feat.future_rag_usage}` |"
            )

        lines.extend([
            "\n---",
            "\n## Detailed Attribute Specifications\n"
        ])

        for k, feat in sorted(self._registry.items()):
            lines.extend([
                f"### `{feat.technical_name}` ({feat.business_name})",
                f"- **Data Type:** `{feat.data_type}` (`Nullable: {feat.nullable}`, `Expected Missing %: {feat.missing_percentage}%`)",
                f"- **Business Meaning:** {feat.business_meaning}",
                f"- **ML Importance:** `{feat.ml_importance}` | **Leakage Tier:** `{feat.target_leakage_classification}`",
                f"- **Preprocessing Strategy:** Encoding=`{feat.encoding_strategy}`, Imputation=`{feat.imputation_strategy}`, Scaling=`{feat.scaling_strategy}`",
                f"- **Feature Engineering Rules:** {feat.feature_engineering_rules}",
                f"- **Downstream Usage Contracts:** RF=`{feat.random_forest_usage}`, Embedding=`{feat.embedding_usage}`, FAISS=`{feat.faiss_metadata_usage}`, Dashboard=`{feat.dashboard_usage}`, API=`{feat.api_exposure}`, RAG=`{feat.future_rag_usage}`, Explainability=`{feat.explainability_usage}`",
                f"- **Status:** `{'Required' if feat.required_or_optional=='required' else 'Optional'}` (`Deprecated: {feat.deprecated_status}`)\n"
            ])

        with open(out_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        logger.info(f"Exported Feature Registry Markdown to: {out_file}")
        return out_file

    def _populate_default_registry(self) -> None:
        """Populate the registry with the complete 38 raw ServiceNow schema + 11 derived features."""
        # 1. incident_number
        self.register_feature(FeatureDefinition(
            business_name="Incident Number", technical_name="incident_number", data_type="string",
            nullable=False, cardinality="unique", missing_percentage=0.0,
            business_meaning="Unique system identifier assigned to ticket upon creation in ServiceNow.",
            ml_importance="none", target_leakage_classification="safe", encoding_strategy="none",
            imputation_strategy="none", scaling_strategy="none", feature_engineering_rules="Retain purely for audit ID and correlation; exclude from ML predictor arrays.",
            random_forest_usage="excluded", embedding_usage="excluded", faiss_metadata_usage="exact_match_filter",
            dashboard_usage="detail_table", api_exposure="required_payload", future_rag_usage="citation_reference",
            explainability_usage="excluded", required_or_optional="required"
        ))

        # 2. opened_at
        self.register_feature(FeatureDefinition(
            business_name="Opened Timestamp", technical_name="opened_at", data_type="datetime",
            nullable=False, cardinality="high", missing_percentage=0.0,
            business_meaning="UTC timestamp when the ticket was created and triage initiated.",
            ml_importance="high", target_leakage_classification="safe", encoding_strategy="none",
            imputation_strategy="none", scaling_strategy="none", feature_engineering_rules="Extract hour and day of week; apply continuous Sine/Cosine cyclic shift encoding.",
            random_forest_usage="excluded", embedding_usage="excluded", faiss_metadata_usage="excluded",
            dashboard_usage="kpi_filter", api_exposure="required_payload", future_rag_usage="excluded",
            explainability_usage="excluded", required_or_optional="required"
        ))

        # 3. resolved_at
        self.register_feature(FeatureDefinition(
            business_name="Resolved Timestamp", technical_name="resolved_at", data_type="datetime",
            nullable=True, cardinality="high", missing_percentage=2.0,
            business_meaning="UTC timestamp when support engineering restored operational service.",
            ml_importance="high", target_leakage_classification="blocked", encoding_strategy="none",
            imputation_strategy="none", scaling_strategy="none", feature_engineering_rules="Strictly exclude at triage time to prevent future timestamp leakage. Used only for MTTR outcome calculation.",
            random_forest_usage="excluded", embedding_usage="excluded", faiss_metadata_usage="excluded",
            dashboard_usage="chart_axis", api_exposure="response_only", future_rag_usage="excluded",
            explainability_usage="excluded", required_or_optional="optional"
        ))

        # 4. closed_at
        self.register_feature(FeatureDefinition(
            business_name="Closed Timestamp", technical_name="closed_at", data_type="datetime",
            nullable=True, cardinality="high", missing_percentage=5.0,
            business_meaning="UTC timestamp when ticket is administratively finalized after verification.",
            ml_importance="none", target_leakage_classification="blocked", encoding_strategy="none",
            imputation_strategy="none", scaling_strategy="none", feature_engineering_rules="Strictly exclude from all predictive models due to administrative delay variance.",
            random_forest_usage="excluded", embedding_usage="excluded", faiss_metadata_usage="excluded",
            dashboard_usage="detail_table", api_exposure="response_only", future_rag_usage="excluded",
            explainability_usage="excluded", required_or_optional="optional"
        ))

        # 5. priority
        self.register_feature(FeatureDefinition(
            business_name="Priority Level", technical_name="priority", data_type="integer",
            nullable=False, cardinality="5", missing_percentage=0.0,
            business_meaning="Ordinal severity rating (1=Critical to 5=Planning) governing SLA windows.",
            ml_importance="high", target_leakage_classification="safe", encoding_strategy="ordinal",
            imputation_strategy="mode", scaling_strategy="none", feature_engineering_rules="Retain as raw integer 1-5; directly weights decision tree splits and SLA calculation.",
            random_forest_usage="predictor", embedding_usage="excluded", faiss_metadata_usage="exact_match_filter",
            dashboard_usage="kpi_filter", api_exposure="required_payload", future_rag_usage="metadata_filter",
            explainability_usage="shap_feature_label", required_or_optional="required"
        ))

        # 6. impact
        self.register_feature(FeatureDefinition(
            business_name="Business Impact", technical_name="impact", data_type="integer",
            nullable=False, cardinality="3", missing_percentage=0.0,
            business_meaning="Scope of business disruption (1=High/Bank-wide to 3=Low/Single User).",
            ml_importance="medium", target_leakage_classification="safe", encoding_strategy="ordinal",
            imputation_strategy="mode", scaling_strategy="none", feature_engineering_rules="Retain as ordinal 1-3 predictor.",
            random_forest_usage="predictor", embedding_usage="excluded", faiss_metadata_usage="excluded",
            dashboard_usage="kpi_filter", api_exposure="required_payload", future_rag_usage="excluded",
            explainability_usage="shap_feature_label", required_or_optional="required"
        ))

        # 7. urgency
        self.register_feature(FeatureDefinition(
            business_name="Urgency Level", technical_name="urgency", data_type="integer",
            nullable=False, cardinality="3", missing_percentage=0.0,
            business_meaning="Time criticality of service restoration (1=High to 3=Low).",
            ml_importance="medium", target_leakage_classification="safe", encoding_strategy="ordinal",
            imputation_strategy="mode", scaling_strategy="none", feature_engineering_rules="Retain as ordinal 1-3 predictor.",
            random_forest_usage="predictor", embedding_usage="excluded", faiss_metadata_usage="excluded",
            dashboard_usage="kpi_filter", api_exposure="required_payload", future_rag_usage="excluded",
            explainability_usage="shap_feature_label", required_or_optional="required"
        ))

        # 8. severity
        self.register_feature(FeatureDefinition(
            business_name="System Severity", technical_name="severity", data_type="integer",
            nullable=False, cardinality="3", missing_percentage=0.0,
            business_meaning="Technical alarm intensity reported by monitoring tools.",
            ml_importance="medium", target_leakage_classification="safe", encoding_strategy="ordinal",
            imputation_strategy="mode", scaling_strategy="none", feature_engineering_rules="Retain as ordinal 1-3 predictor.",
            random_forest_usage="predictor", embedding_usage="excluded", faiss_metadata_usage="excluded",
            dashboard_usage="chart_axis", api_exposure="required_payload", future_rag_usage="excluded",
            explainability_usage="shap_feature_label", required_or_optional="required"
        ))

        # 9. state
        self.register_feature(FeatureDefinition(
            business_name="Lifecycle State", technical_name="state", data_type="integer",
            nullable=False, cardinality="8", missing_percentage=0.0,
            business_meaning="Current operational phase (1=New, 2=In Progress, 6=Resolved, 7=Closed, 8=Canceled).",
            ml_importance="medium", target_leakage_classification="safe", encoding_strategy="one_hot",
            imputation_strategy="mode", scaling_strategy="none", feature_engineering_rules="Use as status filter during training dataset preparation (exclude canceled/state=8).",
            random_forest_usage="excluded", embedding_usage="excluded", faiss_metadata_usage="exact_match_filter",
            dashboard_usage="kpi_filter", api_exposure="required_payload", future_rag_usage="metadata_filter",
            explainability_usage="excluded", required_or_optional="required"
        ))

        # 10. category
        self.register_feature(FeatureDefinition(
            business_name="IT Domain Category", technical_name="category", data_type="string",
            nullable=False, cardinality="8", missing_percentage=0.0,
            business_meaning="Top-level functional taxonomy classification (e.g., Core Banking, Security).",
            ml_importance="high", target_leakage_classification="safe", encoding_strategy="one_hot",
            imputation_strategy="constant_unknown", scaling_strategy="none", feature_engineering_rules="One-hot encode across 8 banking domains; primary driver for assignment routing.",
            random_forest_usage="predictor", embedding_usage="excluded", faiss_metadata_usage="structural_boost_tag",
            dashboard_usage="kpi_filter", api_exposure="required_payload", future_rag_usage="metadata_filter",
            explainability_usage="shap_feature_label", required_or_optional="required"
        ))

        # 11. subcategory
        self.register_feature(FeatureDefinition(
            business_name="Technical Subcategory", technical_name="subcategory", data_type="string",
            nullable=False, cardinality="40", missing_percentage=0.0,
            business_meaning="Granular software/hardware fault classification within Category.",
            ml_importance="high", target_leakage_classification="safe", encoding_strategy="frequency",
            imputation_strategy="constant_unknown", scaling_strategy="none", feature_engineering_rules="Apply smooth out-of-fold target/frequency encoding to manage high cardinality (~40 categories).",
            random_forest_usage="predictor", embedding_usage="excluded", faiss_metadata_usage="structural_boost_tag",
            dashboard_usage="chart_axis", api_exposure="required_payload", future_rag_usage="metadata_filter",
            explainability_usage="shap_feature_label", required_or_optional="required"
        ))

        # 12. assignment_group
        self.register_feature(FeatureDefinition(
            business_name="Assignment Group", technical_name="assignment_group", data_type="string",
            nullable=False, cardinality="18", missing_percentage=0.0,
            business_meaning="Designated L1/L2/L3 engineering support squad responsible for ticket resolution.",
            ml_importance="high", target_leakage_classification="safe", encoding_strategy="label",
            imputation_strategy="constant_unknown", scaling_strategy="none", feature_engineering_rules="Primary multi-class classification target (`y_assignment_group`). Apply balanced class weights.",
            random_forest_usage="target_assignment_group", embedding_usage="excluded", faiss_metadata_usage="exact_match_filter",
            dashboard_usage="kpi_filter", api_exposure="response_only", future_rag_usage="metadata_filter",
            explainability_usage="shap_feature_label", required_or_optional="required"
        ))

        # 13. assigned_to
        self.register_feature(FeatureDefinition(
            business_name="Assigned Engineer", technical_name="assigned_to", data_type="string",
            nullable=True, cardinality="high", missing_percentage=15.0,
            business_meaning="Individual engineer assigned inside the support squad.",
            ml_importance="none", target_leakage_classification="safe", encoding_strategy="none",
            imputation_strategy="constant_unknown", scaling_strategy="none", feature_engineering_rules="High cardinality person ID; exclude from all predictive modeling to prevent overfitting.",
            random_forest_usage="excluded", embedding_usage="excluded", faiss_metadata_usage="excluded",
            dashboard_usage="detail_table", api_exposure="optional_payload", future_rag_usage="excluded",
            explainability_usage="excluded", required_or_optional="optional"
        ))

        # 14. business_service
        self.register_feature(FeatureDefinition(
            business_name="Affected Business Service", technical_name="business_service", data_type="string",
            nullable=False, cardinality="15", missing_percentage=0.0,
            business_meaning="Business-facing banking service offering (e.g., SWIFT Payments, ATM Network).",
            ml_importance="high", target_leakage_classification="safe", encoding_strategy="frequency",
            imputation_strategy="constant_unknown", scaling_strategy="none", feature_engineering_rules="Apply frequency encoding; strong exact-match structural boost in hybrid similarity.",
            random_forest_usage="predictor", embedding_usage="excluded", faiss_metadata_usage="structural_boost_tag",
            dashboard_usage="chart_axis", api_exposure="required_payload", future_rag_usage="metadata_filter",
            explainability_usage="shap_feature_label", required_or_optional="required"
        ))

        # 15. cmdb_ci
        self.register_feature(FeatureDefinition(
            business_name="Configuration Item (CI)", technical_name="cmdb_ci", data_type="string",
            nullable=False, cardinality="high", missing_percentage=0.0,
            business_meaning="Exact CMDB infrastructure asset ID (server, database node, gateway).",
            ml_importance="medium", target_leakage_classification="safe", encoding_strategy="frequency",
            imputation_strategy="constant_unknown", scaling_strategy="none", feature_engineering_rules="Frequency encode for tabular models; provide exact match (+0.15 boost) inside FAISS hybrid retrieval.",
            random_forest_usage="predictor", embedding_usage="excluded", faiss_metadata_usage="structural_boost_tag",
            dashboard_usage="detail_table", api_exposure="required_payload", future_rag_usage="metadata_filter",
            explainability_usage="shap_feature_label", required_or_optional="required"
        ))

        # 16. vendor
        self.register_feature(FeatureDefinition(
            business_name="Third-Party Vendor", technical_name="vendor", data_type="string",
            nullable=True, cardinality="10", missing_percentage=40.0,
            business_meaning="Hardware/software vendor associated with the affected CI (e.g., IBM, Cisco).",
            ml_importance="low", target_leakage_classification="safe", encoding_strategy="frequency",
            imputation_strategy="constant_unknown", scaling_strategy="none", feature_engineering_rules="Impute missing as 'Internal/UNKNOWN'; frequency encode.",
            random_forest_usage="predictor", embedding_usage="excluded", faiss_metadata_usage="excluded",
            dashboard_usage="chart_axis", api_exposure="optional_payload", future_rag_usage="excluded",
            explainability_usage="shap_feature_label", required_or_optional="optional"
        ))

        # 17. caller
        self.register_feature(FeatureDefinition(
            business_name="Reporting Caller ID", technical_name="caller", data_type="string",
            nullable=False, cardinality="high", missing_percentage=0.0,
            business_meaning="Employee or automated monitoring daemon that initiated the ticket.",
            ml_importance="none", target_leakage_classification="safe", encoding_strategy="none",
            imputation_strategy="none", scaling_strategy="none", feature_engineering_rules="Exclude from ML modeling due to extreme cardinality and privacy governance.",
            random_forest_usage="excluded", embedding_usage="excluded", faiss_metadata_usage="excluded",
            dashboard_usage="detail_table", api_exposure="required_payload", future_rag_usage="excluded",
            explainability_usage="excluded", required_or_optional="required"
        ))

        # 18. short_description
        self.register_feature(FeatureDefinition(
            business_name="Incident Summary Title", technical_name="short_description", data_type="string",
            nullable=False, cardinality="high", missing_percentage=0.0,
            business_meaning="Primary summary headline describing the incident or monitoring alert.",
            ml_importance="high", target_leakage_classification="safe", encoding_strategy="none",
            imputation_strategy="constant_unknown", scaling_strategy="none", feature_engineering_rules="Normalize whitespace, strip HTML; primary text input for SentenceTransformer 384-D vector generation.",
            random_forest_usage="excluded", embedding_usage="primary_semantic_input", faiss_metadata_usage="excluded",
            dashboard_usage="detail_table", api_exposure="required_payload", future_rag_usage="knowledge_chunk_payload",
            explainability_usage="natural_language_context", required_or_optional="required"
        ))

        # 19. description
        self.register_feature(FeatureDefinition(
            business_name="Full Diagnostic Description", technical_name="description", data_type="string",
            nullable=False, cardinality="high", missing_percentage=0.0,
            business_meaning="Comprehensive diagnostic text, stack traces, and error logs reported.",
            ml_importance="high", target_leakage_classification="safe", encoding_strategy="none",
            imputation_strategy="constant_unknown", scaling_strategy="none", feature_engineering_rules="Truncate to 256 tokens; concatenate with short_description for secondary semantic embedding.",
            random_forest_usage="excluded", embedding_usage="secondary_summary", faiss_metadata_usage="excluded",
            dashboard_usage="detail_table", api_exposure="required_payload", future_rag_usage="knowledge_chunk_payload",
            explainability_usage="natural_language_context", required_or_optional="required"
        ))

        # 20. close_notes
        self.register_feature(FeatureDefinition(
            business_name="Resolution Close Notes", technical_name="close_notes", data_type="string",
            nullable=True, cardinality="high", missing_percentage=12.0,
            business_meaning="Engineer's technical summary of root cause and exact remediation steps taken.",
            ml_importance="high", target_leakage_classification="blocked", encoding_strategy="none",
            imputation_strategy="none", scaling_strategy="none", feature_engineering_rules="STRICTLY EXCLUDED at triage prediction time. Serves as the golden text payload inside RAG retrieval knowledge bases.",
            random_forest_usage="excluded", embedding_usage="excluded", faiss_metadata_usage="excluded",
            dashboard_usage="detail_table", api_exposure="response_only", future_rag_usage="knowledge_chunk_payload",
            explainability_usage="natural_language_context", required_or_optional="optional"
        ))

        # 21. resolution_code
        self.register_feature(FeatureDefinition(
            business_name="Resolution Code Taxonomy", technical_name="resolution_code", data_type="string",
            nullable=True, cardinality="6", missing_percentage=12.0,
            business_meaning="Standardized outcome classification (e.g., Solved Permanently, Workaround, User Error).",
            ml_importance="high", target_leakage_classification="blocked", encoding_strategy="none",
            imputation_strategy="none", scaling_strategy="none", feature_engineering_rules="Post-resolution outcome classification; exclude from all early-stage triage predictors.",
            random_forest_usage="excluded", embedding_usage="excluded", faiss_metadata_usage="excluded",
            dashboard_usage="chart_axis", api_exposure="response_only", future_rag_usage="metadata_filter",
            explainability_usage="excluded", required_or_optional="optional"
        ))

        # 22. resolution_time_hours
        self.register_feature(FeatureDefinition(
            business_name="Resolution Time (Hours)", technical_name="resolution_time_hours", data_type="float",
            nullable=True, cardinality="continuous", missing_percentage=2.0,
            business_meaning="Continuous elapsed clock time from opened_at to resolved_at.",
            ml_importance="high", target_leakage_classification="blocked", encoding_strategy="none",
            imputation_strategy="median", scaling_strategy="log1p", feature_engineering_rules="Primary regression target (`y_resolution_time`). Apply log1p transform (`np.log1p(y)`) to normalize right-skewed log-normal distribution.",
            random_forest_usage="target_resolution_time", embedding_usage="excluded", faiss_metadata_usage="excluded",
            dashboard_usage="chart_axis", api_exposure="response_only", future_rag_usage="excluded",
            explainability_usage="shap_feature_label", required_or_optional="optional"
        ))

        # 23. calendar_duration_hours
        self.register_feature(FeatureDefinition(
            business_name="Calendar Duration (Hours)", technical_name="calendar_duration_hours", data_type="float",
            nullable=True, cardinality="continuous", missing_percentage=2.0,
            business_meaning="Total elapsed calendar time from opened_at to administrative closed_at.",
            ml_importance="none", target_leakage_classification="blocked", encoding_strategy="none",
            imputation_strategy="none", scaling_strategy="none", feature_engineering_rules="Post-resolution outcome metric; exclude from triage prediction.",
            random_forest_usage="excluded", embedding_usage="excluded", faiss_metadata_usage="excluded",
            dashboard_usage="detail_table", api_exposure="response_only", future_rag_usage="excluded",
            explainability_usage="excluded", required_or_optional="optional"
        ))

        # 24. business_duration_hours
        self.register_feature(FeatureDefinition(
            business_name="Business Shift Duration (Hours)", technical_name="business_duration_hours", data_type="float",
            nullable=True, cardinality="continuous", missing_percentage=2.0,
            business_meaning="Elapsed engineering effort hours within standard operational shifts (8am-6pm).",
            ml_importance="none", target_leakage_classification="blocked", encoding_strategy="none",
            imputation_strategy="none", scaling_strategy="none", feature_engineering_rules="Post-resolution calculation; exclude from triage prediction.",
            random_forest_usage="excluded", embedding_usage="excluded", faiss_metadata_usage="excluded",
            dashboard_usage="chart_axis", api_exposure="response_only", future_rag_usage="excluded",
            explainability_usage="excluded", required_or_optional="optional"
        ))

        # 25. made_sla
        self.register_feature(FeatureDefinition(
            business_name="SLA Compliance Flag", technical_name="made_sla", data_type="boolean",
            nullable=False, cardinality="2", missing_percentage=0.0,
            business_meaning="Binary indicator of whether resolution met target window (True=Met, False=Breached).",
            ml_importance="high", target_leakage_classification="blocked", encoding_strategy="none",
            imputation_strategy="none", scaling_strategy="none", feature_engineering_rules="Post-resolution outcome flag; strictly banned from triage feature matrix.",
            random_forest_usage="excluded", embedding_usage="excluded", faiss_metadata_usage="excluded",
            dashboard_usage="kpi_filter", api_exposure="response_only", future_rag_usage="excluded",
            explainability_usage="excluded", required_or_optional="required"
        ))

        # 26. sla_status
        self.register_feature(FeatureDefinition(
            business_name="SLA Status Label", technical_name="sla_status", data_type="string",
            nullable=False, cardinality="2", missing_percentage=0.0,
            business_meaning="Categorical SLA status label ('Met' or 'Breached').",
            ml_importance="high", target_leakage_classification="blocked", encoding_strategy="none",
            imputation_strategy="none", scaling_strategy="none", feature_engineering_rules="Post-resolution outcome label; banned from triage predictors.",
            random_forest_usage="excluded", embedding_usage="excluded", faiss_metadata_usage="excluded",
            dashboard_usage="chart_axis", api_exposure="response_only", future_rag_usage="excluded",
            explainability_usage="excluded", required_or_optional="required"
        ))

        # 27. sla_due
        self.register_feature(FeatureDefinition(
            business_name="SLA Target Cutoff Timestamp", technical_name="sla_due", data_type="datetime",
            nullable=False, cardinality="high", missing_percentage=0.0,
            business_meaning="Exact UTC deadline by which ticket must be resolved to avoid SLA breach.",
            ml_importance="medium", target_leakage_classification="safe", encoding_strategy="none",
            imputation_strategy="none", scaling_strategy="none", feature_engineering_rules="Computed from opened_at + priority target window; usable for operational priority sorting.",
            random_forest_usage="excluded", embedding_usage="excluded", faiss_metadata_usage="excluded",
            dashboard_usage="detail_table", api_exposure="response_only", future_rag_usage="excluded",
            explainability_usage="excluded", required_or_optional="required"
        ))

        # 28. reassignment_count
        self.register_feature(FeatureDefinition(
            business_name="Reassignment Counter", technical_name="reassignment_count", data_type="integer",
            nullable=False, cardinality="low", missing_percentage=0.0,
            business_meaning="Number of times ticket ownership bounced across different assignment groups.",
            ml_importance="medium", target_leakage_classification="blocked", encoding_strategy="none",
            imputation_strategy="none", scaling_strategy="none", feature_engineering_rules="Operational outcome metric tracking routing efficiency; exclude from triage assignment prediction.",
            random_forest_usage="excluded", embedding_usage="excluded", faiss_metadata_usage="excluded",
            dashboard_usage="chart_axis", api_exposure="response_only", future_rag_usage="excluded",
            explainability_usage="excluded", required_or_optional="required"
        ))

        # 29. reopen_count
        self.register_feature(FeatureDefinition(
            business_name="Reopen Counter", technical_name="reopen_count", data_type="integer",
            nullable=False, cardinality="low", missing_percentage=0.0,
            business_meaning="Number of times ticket was reopened after initial resolution.",
            ml_importance="low", target_leakage_classification="blocked", encoding_strategy="none",
            imputation_strategy="none", scaling_strategy="none", feature_engineering_rules="Post-resolution outcome metric; exclude from triage prediction.",
            random_forest_usage="excluded", embedding_usage="excluded", faiss_metadata_usage="excluded",
            dashboard_usage="chart_axis", api_exposure="response_only", future_rag_usage="excluded",
            explainability_usage="excluded", required_or_optional="required"
        ))

        # 30. problem_flag
        self.register_feature(FeatureDefinition(
            business_name="Problem Investigation Flag", technical_name="problem_flag", data_type="boolean",
            nullable=False, cardinality="2", missing_percentage=0.0,
            business_meaning="Whether ticket linked or escalated to a formal Problem investigation (`PRB`).",
            ml_importance="medium", target_leakage_classification="warning", encoding_strategy="none",
            imputation_strategy="none", scaling_strategy="none", feature_engineering_rules="Usually linked post-triage; exercise caution if used as predictor.",
            random_forest_usage="excluded", embedding_usage="excluded", faiss_metadata_usage="excluded",
            dashboard_usage="kpi_filter", api_exposure="response_only", future_rag_usage="metadata_filter",
            explainability_usage="excluded", required_or_optional="required"
        ))

        # 31. problem_record
        self.register_feature(FeatureDefinition(
            business_name="Linked Problem ID", technical_name="problem_record", data_type="string",
            nullable=True, cardinality="medium", missing_percentage=90.0,
            business_meaning="Associated Problem investigation ID (`PRB001...`) when problem_flag is True.",
            ml_importance="low", target_leakage_classification="warning", encoding_strategy="none",
            imputation_strategy="constant_unknown", scaling_strategy="none", feature_engineering_rules="Derive binary indicator (`has_problem_record`).",
            random_forest_usage="excluded", embedding_usage="excluded", faiss_metadata_usage="excluded",
            dashboard_usage="detail_table", api_exposure="optional_payload", future_rag_usage="citation_reference",
            explainability_usage="excluded", required_or_optional="optional"
        ))

        # 32. change_request
        self.register_feature(FeatureDefinition(
            business_name="Linked Change Request ID", technical_name="change_request", data_type="string",
            nullable=True, cardinality="medium", missing_percentage=93.0,
            business_meaning="Associated Change Request ID (`CHG002...`) if caused by or resolved via deployment.",
            ml_importance="medium", target_leakage_classification="warning", encoding_strategy="none",
            imputation_strategy="constant_unknown", scaling_strategy="none", feature_engineering_rules="Derive binary indicator (`has_change_request`); strong signal for release/deployment engineering L3 squads.",
            random_forest_usage="excluded", embedding_usage="excluded", faiss_metadata_usage="excluded",
            dashboard_usage="detail_table", api_exposure="optional_payload", future_rag_usage="citation_reference",
            explainability_usage="excluded", required_or_optional="optional"
        ))

        # 33. knowledge_linked
        self.register_feature(FeatureDefinition(
            business_name="Knowledge Base Linked Flag", technical_name="knowledge_linked", data_type="boolean",
            nullable=False, cardinality="2", missing_percentage=0.0,
            business_meaning="Whether a Knowledge Base article was cited during incident resolution.",
            ml_importance="medium", target_leakage_classification="blocked", encoding_strategy="none",
            imputation_strategy="none", scaling_strategy="none", feature_engineering_rules="Post-resolution outcome indicator; banned from triage predictors.",
            random_forest_usage="excluded", embedding_usage="excluded", faiss_metadata_usage="excluded",
            dashboard_usage="kpi_filter", api_exposure="response_only", future_rag_usage="excluded",
            explainability_usage="excluded", required_or_optional="required"
        ))

        # 34. knowledge_base
        self.register_feature(FeatureDefinition(
            business_name="Linked Knowledge Article ID", technical_name="knowledge_base", data_type="string",
            nullable=True, cardinality="medium", missing_percentage=75.0,
            business_meaning="Specific KB article identifier (`KB0010042`) used for remediation.",
            ml_importance="medium", target_leakage_classification="blocked", encoding_strategy="none",
            imputation_strategy="constant_unknown", scaling_strategy="none", feature_engineering_rules="Direct citation reference inside RAG resolution retrieval responses.",
            random_forest_usage="excluded", embedding_usage="excluded", faiss_metadata_usage="excluded",
            dashboard_usage="detail_table", api_exposure="response_only", future_rag_usage="citation_reference",
            explainability_usage="excluded", required_or_optional="optional"
        ))

        # 35. contact_type
        self.register_feature(FeatureDefinition(
            business_name="Reporting Contact Type", technical_name="contact_type", data_type="string",
            nullable=False, cardinality="4", missing_percentage=0.0,
            business_meaning="Origin reporting channel (`Alert`, `Phone`, `Self-service`, `Email`).",
            ml_importance="low", target_leakage_classification="safe", encoding_strategy="one_hot",
            imputation_strategy="mode", scaling_strategy="none", feature_engineering_rules="One-hot encode into binary indicators (`contact_type_Alert`, etc.).",
            random_forest_usage="predictor", embedding_usage="excluded", faiss_metadata_usage="exact_match_filter",
            dashboard_usage="kpi_filter", api_exposure="required_payload", future_rag_usage="metadata_filter",
            explainability_usage="shap_feature_label", required_or_optional="required"
        ))

        # 36. location
        self.register_feature(FeatureDefinition(
            business_name="Origin Geographic Location", technical_name="location", data_type="string",
            nullable=False, cardinality="12", missing_percentage=0.0,
            business_meaning="Banking data center or branch facility where issue originated.",
            ml_importance="low", target_leakage_classification="safe", encoding_strategy="frequency",
            imputation_strategy="mode", scaling_strategy="none", feature_engineering_rules="Frequency encode to capture regional incident volume densities.",
            random_forest_usage="predictor", embedding_usage="excluded", faiss_metadata_usage="exact_match_filter",
            dashboard_usage="kpi_filter", api_exposure="required_payload", future_rag_usage="metadata_filter",
            explainability_usage="shap_feature_label", required_or_optional="required"
        ))

        # 37. duplicate_incident
        self.register_feature(FeatureDefinition(
            business_name="Master Duplicate Ticket ID", technical_name="duplicate_incident", data_type="string",
            nullable=True, cardinality="medium", missing_percentage=95.0,
            business_meaning="Parent ticket ID if this record is marked as a duplicate (`State=8`).",
            ml_importance="low", target_leakage_classification="warning", encoding_strategy="none",
            imputation_strategy="constant_unknown", scaling_strategy="none", feature_engineering_rules="Derive binary indicator (`is_duplicate`).",
            random_forest_usage="excluded", embedding_usage="excluded", faiss_metadata_usage="excluded",
            dashboard_usage="detail_table", api_exposure="optional_payload", future_rag_usage="excluded",
            explainability_usage="excluded", required_or_optional="optional"
        ))

        # 38. parent_incident
        self.register_feature(FeatureDefinition(
            business_name="Master Parent Incident ID", technical_name="parent_incident", data_type="string",
            nullable=True, cardinality="medium", missing_percentage=92.0,
            business_meaning="Master parent ticket linking related alert storms.",
            ml_importance="low", target_leakage_classification="warning", encoding_strategy="none",
            imputation_strategy="constant_unknown", scaling_strategy="none", feature_engineering_rules="Derive binary indicator (`has_parent_incident`).",
            random_forest_usage="excluded", embedding_usage="excluded", faiss_metadata_usage="excluded",
            dashboard_usage="detail_table", api_exposure="optional_payload", future_rag_usage="excluded",
            explainability_usage="excluded", required_or_optional="optional"
        ))

        # =====================================================================
        # Derived Engineering Features (11 attributes)
        # =====================================================================

        # 39. opened_at_hour
        self.register_feature(FeatureDefinition(
            business_name="Opened Hour of Day", technical_name="opened_at_hour", data_type="integer",
            nullable=False, cardinality="24", missing_percentage=0.0,
            business_meaning="Integer hour (0-23) extracted from opened_at.",
            ml_importance="medium", target_leakage_classification="safe", encoding_strategy="sine_cosine",
            imputation_strategy="none", scaling_strategy="none", feature_engineering_rules="Derived from opened_at. Intermediate step for cyclic shift encoding.",
            random_forest_usage="predictor", embedding_usage="excluded", faiss_metadata_usage="excluded",
            dashboard_usage="chart_axis", api_exposure="internal_only", future_rag_usage="excluded",
            explainability_usage="shap_feature_label", required_or_optional="required"
        ))

        # 40. opened_at_hour_sin
        self.register_feature(FeatureDefinition(
            business_name="Opened Hour Sine Cyclic", technical_name="opened_at_hour_sin", data_type="float",
            nullable=False, cardinality="continuous", missing_percentage=0.0,
            business_meaning="Sine cyclic component of opened_at_hour (`sin(2*pi*hour/24)`).",
            ml_importance="medium", target_leakage_classification="safe", encoding_strategy="none",
            imputation_strategy="none", scaling_strategy="none", feature_engineering_rules="Preserves 23:00 to 00:00 continuous shift proximity for tree models and clustering.",
            random_forest_usage="predictor", embedding_usage="excluded", faiss_metadata_usage="excluded",
            dashboard_usage="excluded", api_exposure="internal_only", future_rag_usage="excluded",
            explainability_usage="shap_feature_label", required_or_optional="required"
        ))

        # 41. opened_at_hour_cos
        self.register_feature(FeatureDefinition(
            business_name="Opened Hour Cosine Cyclic", technical_name="opened_at_hour_cos", data_type="float",
            nullable=False, cardinality="continuous", missing_percentage=0.0,
            business_meaning="Cosine cyclic component of opened_at_hour (`cos(2*pi*hour/24)`).",
            ml_importance="medium", target_leakage_classification="safe", encoding_strategy="none",
            imputation_strategy="none", scaling_strategy="none", feature_engineering_rules="Preserves continuous shift proximity in tandem with sine component.",
            random_forest_usage="predictor", embedding_usage="excluded", faiss_metadata_usage="excluded",
            dashboard_usage="excluded", api_exposure="internal_only", future_rag_usage="excluded",
            explainability_usage="shap_feature_label", required_or_optional="required"
        ))

        # 42. opened_at_dayofweek
        self.register_feature(FeatureDefinition(
            business_name="Opened Day of Week", technical_name="opened_at_dayofweek", data_type="integer",
            nullable=False, cardinality="7", missing_percentage=0.0,
            business_meaning="Integer day of week (0=Monday to 6=Sunday) extracted from opened_at.",
            ml_importance="medium", target_leakage_classification="safe", encoding_strategy="sine_cosine",
            imputation_strategy="none", scaling_strategy="none", feature_engineering_rules="Derived from opened_at. Distinguishes weekend vs weekday incident patterns.",
            random_forest_usage="predictor", embedding_usage="excluded", faiss_metadata_usage="excluded",
            dashboard_usage="chart_axis", api_exposure="internal_only", future_rag_usage="excluded",
            explainability_usage="shap_feature_label", required_or_optional="required"
        ))

        # 43. opened_at_dayofweek_sin
        self.register_feature(FeatureDefinition(
            business_name="Opened Day of Week Sine Cyclic", technical_name="opened_at_dayofweek_sin", data_type="float",
            nullable=False, cardinality="continuous", missing_percentage=0.0,
            business_meaning="Sine cyclic component of opened_at_dayofweek (`sin(2*pi*day/7)`).",
            ml_importance="medium", target_leakage_classification="safe", encoding_strategy="none",
            imputation_strategy="none", scaling_strategy="none", feature_engineering_rules="Preserves Sunday-Monday cyclic boundary.",
            random_forest_usage="predictor", embedding_usage="excluded", faiss_metadata_usage="excluded",
            dashboard_usage="excluded", api_exposure="internal_only", future_rag_usage="excluded",
            explainability_usage="shap_feature_label", required_or_optional="required"
        ))

        # 44. opened_at_dayofweek_cos
        self.register_feature(FeatureDefinition(
            business_name="Opened Day of Week Cosine Cyclic", technical_name="opened_at_dayofweek_cos", data_type="float",
            nullable=False, cardinality="continuous", missing_percentage=0.0,
            business_meaning="Cosine cyclic component of opened_at_dayofweek (`cos(2*pi*day/7)`).",
            ml_importance="medium", target_leakage_classification="safe", encoding_strategy="none",
            imputation_strategy="none", scaling_strategy="none", feature_engineering_rules="Preserves Sunday-Monday cyclic boundary.",
            random_forest_usage="predictor", embedding_usage="excluded", faiss_metadata_usage="excluded",
            dashboard_usage="excluded", api_exposure="internal_only", future_rag_usage="excluded",
            explainability_usage="shap_feature_label", required_or_optional="required"
        ))

        # 45. is_business_hours
        self.register_feature(FeatureDefinition(
            business_name="Business Hours Indicator", technical_name="is_business_hours", data_type="integer",
            nullable=False, cardinality="2", missing_percentage=0.0,
            business_meaning="Binary indicator (`1` if opened Mon-Fri 8am-6pm else `0`).",
            ml_importance="medium", target_leakage_classification="safe", encoding_strategy="none",
            imputation_strategy="none", scaling_strategy="none", feature_engineering_rules="Derived from opened_at; strong predictor of initial L1 triage latency.",
            random_forest_usage="predictor", embedding_usage="excluded", faiss_metadata_usage="exact_match_filter",
            dashboard_usage="kpi_filter", api_exposure="internal_only", future_rag_usage="excluded",
            explainability_usage="shap_feature_label", required_or_optional="required"
        ))

        # 46. has_parent_incident
        self.register_feature(FeatureDefinition(
            business_name="Has Parent Incident Flag", technical_name="has_parent_incident", data_type="integer",
            nullable=False, cardinality="2", missing_percentage=0.0,
            business_meaning="Binary indicator (`1` if parent_incident is not empty else `0`).",
            ml_importance="low", target_leakage_classification="safe", encoding_strategy="none",
            imputation_strategy="none", scaling_strategy="none", feature_engineering_rules="Derived from parent_incident.",
            random_forest_usage="predictor", embedding_usage="excluded", faiss_metadata_usage="excluded",
            dashboard_usage="chart_axis", api_exposure="internal_only", future_rag_usage="excluded",
            explainability_usage="shap_feature_label", required_or_optional="required"
        ))

        # 47. has_change_request
        self.register_feature(FeatureDefinition(
            business_name="Has Change Request Flag", technical_name="has_change_request", data_type="integer",
            nullable=False, cardinality="2", missing_percentage=0.0,
            business_meaning="Binary indicator (`1` if change_request is not empty else `0`).",
            ml_importance="medium", target_leakage_classification="safe", encoding_strategy="none",
            imputation_strategy="none", scaling_strategy="none", feature_engineering_rules="Derived from change_request; strong signal for routing to release/change engineering L3 squads.",
            random_forest_usage="predictor", embedding_usage="excluded", faiss_metadata_usage="excluded",
            dashboard_usage="chart_axis", api_exposure="internal_only", future_rag_usage="excluded",
            explainability_usage="shap_feature_label", required_or_optional="required"
        ))

        # 48. has_problem_record
        self.register_feature(FeatureDefinition(
            business_name="Has Problem Record Flag", technical_name="has_problem_record", data_type="integer",
            nullable=False, cardinality="2", missing_percentage=0.0,
            business_meaning="Binary indicator (`1` if problem_record is not empty else `0`).",
            ml_importance="low", target_leakage_classification="safe", encoding_strategy="none",
            imputation_strategy="none", scaling_strategy="none", feature_engineering_rules="Derived from problem_record.",
            random_forest_usage="predictor", embedding_usage="excluded", faiss_metadata_usage="excluded",
            dashboard_usage="chart_axis", api_exposure="internal_only", future_rag_usage="excluded",
            explainability_usage="shap_feature_label", required_or_optional="required"
        ))

        # 49. is_duplicate
        self.register_feature(FeatureDefinition(
            business_name="Is Duplicate Ticket Flag", technical_name="is_duplicate", data_type="integer",
            nullable=False, cardinality="2", missing_percentage=0.0,
            business_meaning="Binary indicator (`1` if duplicate_incident is not empty else `0`).",
            ml_importance="low", target_leakage_classification="safe", encoding_strategy="none",
            imputation_strategy="none", scaling_strategy="none", feature_engineering_rules="Derived from duplicate_incident.",
            random_forest_usage="predictor", embedding_usage="excluded", faiss_metadata_usage="excluded",
            dashboard_usage="chart_axis", api_exposure="internal_only", future_rag_usage="excluded",
            explainability_usage="shap_feature_label", required_or_optional="required"
        ))

        # 50. priority_x_impact
        self.register_feature(FeatureDefinition(
            business_name="Priority x Impact Interaction Score", technical_name="priority_x_impact", data_type="float",
            nullable=False, cardinality="continuous", missing_percentage=0.0,
            business_meaning="Non-linear interaction term multiplying Priority by Impact.",
            ml_importance="high", target_leakage_classification="safe", encoding_strategy="none",
            imputation_strategy="median", scaling_strategy="none", feature_engineering_rules="Derived via EnterpriseFeatureExtractor.",
            random_forest_usage="predictor", embedding_usage="excluded", faiss_metadata_usage="excluded",
            dashboard_usage="chart_axis", api_exposure="internal_only", future_rag_usage="excluded",
            explainability_usage="shap_feature_label", required_or_optional="required"
        ))

        # 51. priority_x_urgency
        self.register_feature(FeatureDefinition(
            business_name="Priority x Urgency Interaction Score", technical_name="priority_x_urgency", data_type="float",
            nullable=False, cardinality="continuous", missing_percentage=0.0,
            business_meaning="Non-linear interaction term multiplying Priority by Urgency.",
            ml_importance="high", target_leakage_classification="safe", encoding_strategy="none",
            imputation_strategy="median", scaling_strategy="none", feature_engineering_rules="Derived via EnterpriseFeatureExtractor.",
            random_forest_usage="predictor", embedding_usage="excluded", faiss_metadata_usage="excluded",
            dashboard_usage="chart_axis", api_exposure="internal_only", future_rag_usage="excluded",
            explainability_usage="shap_feature_label", required_or_optional="required"
        ))
