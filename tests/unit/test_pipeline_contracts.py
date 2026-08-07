"""Unit tests for PipelineContractValidator (`src/data/pipeline_contracts.py`)."""

from pathlib import Path
import pandas as pd
import pytest

from src.data.pipeline_contracts import PipelineContractValidator


def test_pipeline_contract_feature_queries() -> None:
    """Verify clean retrieval of contract feature lists across RF, Embeddings, FAISS, Dashboard, API, and RAG."""
    validator = PipelineContractValidator()

    rf_feats = validator.get_random_forest_features()
    assert "priority" in rf_feats
    assert "category" in rf_feats
    assert "close_notes" not in rf_feats

    embed_feats = validator.get_embedding_text_features()
    assert "short_description" in embed_feats
    assert "description" in embed_feats

    faiss_feats = validator.get_faiss_metadata_features()
    assert "cmdb_ci" in faiss_feats

    kpi_feats = validator.get_dashboard_kpi_features()
    assert "category" in kpi_feats

    api_schema = validator.get_api_request_schema()
    assert "required_payload_fields" in api_schema
    assert "short_description" in api_schema["required_payload_fields"]

    rag_feats = validator.get_rag_knowledge_features()
    assert "close_notes" in rag_feats
    assert "knowledge_base" in rag_feats


def test_dataframe_compliance_validation_clean_vs_leaky() -> None:
    """Verify dataframe schema compliance check and Target Leakage blocking."""
    validator = PipelineContractValidator()

    clean_df = pd.DataFrame({
        "incident_number": ["INC001"],
        "opened_at": ["2026-07-01"],
        "priority": [2],
        "impact": [2],
        "urgency": [2],
        "severity": [2],
        "state": [1],
        "category": ["Core Banking"],
        "subcategory": ["Database"],
        "assignment_group": ["Database Support"],
        "business_service": ["SWIFT"],
        "cmdb_ci": ["CI001"],
        "caller": ["Caller01"],
        "short_description": ["DB Slow"],
        "description": ["Slow queries"],
        "sla_due": ["2026-07-02"],
        "problem_flag": [False],
        "contact_type": ["Alert"],
        "location": ["DC-East"]
    })

    compliant, violations = validator.validate_dataframe_compliance(clean_df, expected_usage="triage_prediction")
    assert compliant is True
    assert len(violations) == 0

    # Inject blocked leakage predictor column into candidate triage predictors df
    leaky_df = clean_df.copy()
    leaky_df["close_notes"] = ["Fixed index"]
    leaky_df["resolution_code"] = ["Solved Permanently"]

    compliant, violations = validator.validate_dataframe_compliance(leaky_df, expected_usage="triage_prediction")
    assert compliant is False
    assert any("Target Leakage Violation" in v for v in violations)
