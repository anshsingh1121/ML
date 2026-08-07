"""
Enterprise Pipeline Contracts (`v1.5.0`).

Provides standardized API adapters that downstream modules call to obtain authorized
feature sets directly from the central Feature Registry, preventing hardcoded columns
and verifying dataframe schema/leakage compliance before any model processing.
"""

from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

from src.data.feature_registry import FeatureRegistry
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PipelineContractValidator:
    """
    Standard adapter interface governing feature access across 6 downstream modules:
    Random Forest, Embeddings, FAISS, Dashboard, ServiceNow API, and Future RAG.
    """

    def __init__(self, registry: Optional[FeatureRegistry] = None) -> None:
        """Initialize validator with FeatureRegistry singleton."""
        self.registry = registry or FeatureRegistry.get_instance()

    def get_random_forest_features(self, target: str = "assignment_group") -> List[str]:
        """Retrieve safe predictor column names authorized for Random Forest training."""
        return self.registry.get_random_forest_predictors(target)

    def get_embedding_text_features(self) -> List[str]:
        """Retrieve text column names authorized for neural embedding tokenization."""
        return self.registry.get_embedding_features()

    def get_faiss_metadata_features(self) -> List[str]:
        """Retrieve column names authorized for structural exact matching and FAISS filtering."""
        return self.registry.get_faiss_metadata_features()

    def get_dashboard_kpi_features(self) -> List[str]:
        """Retrieve column names authorized for dashboard filters and interactive chart axes."""
        feats = self.registry.get_features_by_usage("dashboard_usage", "kpi_filter")
        feats.extend(self.registry.get_features_by_usage("dashboard_usage", "chart_axis"))
        return sorted({f.technical_name for f in feats})

    def get_api_request_schema(self) -> Dict[str, Any]:
        """Retrieve JSON schema of required and optional payload fields for REST API ingestion."""
        required = [f.technical_name for f in self.registry.list_all_features() if f.api_exposure == "required_payload"]
        optional = [f.technical_name for f in self.registry.list_all_features() if f.api_exposure == "optional_payload"]
        return {
            "schema_version": "1.5.0",
            "required_payload_fields": sorted(required),
            "optional_payload_fields": sorted(optional)
        }

    def get_rag_knowledge_features(self) -> List[str]:
        """Retrieve column names authorized as knowledge chunks or citations in RAG retrieval."""
        chunks = self.registry.get_features_by_usage("future_rag_usage", "knowledge_chunk_payload")
        citations = self.registry.get_features_by_usage("future_rag_usage", "citation_reference")
        return sorted({f.technical_name for f in chunks + citations})

    def validate_dataframe_compliance(self, df: pd.DataFrame, expected_usage: str = "triage_prediction") -> Tuple[bool, List[str]]:
        """
        Verify that a candidate dataframe conforms to registry boundaries and contains no Blocked features.

        Args:
            df: Candidate pandas DataFrame.
            expected_usage: 'triage_prediction' (strict exclusion of post-resolution fields) or 'post_resolution_analytics'.

        Returns:
            Tuple[bool, List[str]]: (is_compliant, list_of_violations)
        """
        violations = []

        # 1. Check if required triage attributes exist
        if expected_usage == "triage_prediction":
            derived_features = {
                "opened_at_hour", "opened_at_hour_sin", "opened_at_hour_cos",
                "opened_at_dayofweek", "opened_at_dayofweek_sin", "opened_at_dayofweek_cos",
                "is_business_hours", "has_parent_incident", "has_change_request",
                "has_problem_record", "is_duplicate", "priority_x_impact", "priority_x_urgency"
            }
            safe_feats = self.registry.get_features_by_leakage("safe")
            required_triage = [
                f.technical_name for f in safe_feats
                if f.required_or_optional == "required" and f.technical_name not in derived_features
            ]
            missing_req = [c for c in required_triage if c not in df.columns]
            if missing_req:
                violations.append(f"Missing required triage features: {missing_req}")

            # 2. Check if any blocked leakage attributes are present in predictor columns
            # Note: During training prep, target variables may be present, but post-resolution leakage predictors must not be.
            blocked_feats = [f.technical_name for f in self.registry.get_features_by_leakage("blocked")]
            # Allow target labels if explicit target, but flag leakage predictors like close_notes or resolution_code
            strict_leakage = [b for b in blocked_feats if b in df.columns and b not in ["assignment_group", "resolution_time_hours"]]
            if strict_leakage:
                violations.append(f"Target Leakage Violation: Dataframe contains blocked post-resolution features: {strict_leakage}")

        is_compliant = len(violations) == 0
        if not is_compliant:
            logger.warning(f"Dataframe compliance check FAILED ({len(violations)} violations encountered).")
        else:
            logger.debug("Dataframe compliance check passed cleanly against Feature Registry.")
        return is_compliant, violations
