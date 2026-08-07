"""
Unit tests for Phase 5 Enterprise Hybrid Incident Intelligence Engine (`v2.0.0-alpha`).
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.ml.hybrid.confidence_engine import HybridConfidenceEngine
from src.ml.hybrid.decision_engine import HybridDecisionEngine
from src.ml.hybrid.reasoning_engine import HybridReasoningEngine
from src.ml.hybrid.recommendation_engine import HybridRecommendationEngine
from src.utils.config_manager import ConfigManager


@pytest.fixture
def clean_config():
    """Ensure clean ConfigManager instance initialized with hybrid config."""
    ConfigManager.reset()
    cfg = ConfigManager.get_instance()
    yield cfg
    ConfigManager.reset()


@pytest.fixture
def sample_rf_prediction():
    return {
        "assignment_group": "Platform-Support-L2",
        "confidence_score": 0.85,
        "resolution_time_hours": 3.0,
        "priority": "P2 - High"
    }


@pytest.fixture
def sample_semantic_matches():
    return [
        {
            "incident_number": "INC0010001",
            "similarity_score": 0.9200,
            "assignment_group": "Platform-Support-L2",
            "resolution_time_hours": 2.5,
            "reassignment_count": 0,
            "short_description": "Server crash during ATM sync"
        },
        {
            "incident_number": "INC0010002",
            "similarity_score": 0.8800,
            "assignment_group": "Platform-Support-L2",
            "resolution_time_hours": 3.5,
            "reassignment_count": 0,
            "short_description": "Database connection timeout ATM"
        },
        {
            "incident_number": "INC0010003",
            "similarity_score": 0.8400,
            "assignment_group": "Network-Ops",
            "resolution_time_hours": 4.0,
            "reassignment_count": 1,
            "short_description": "Network packet drop ATM gateway"
        },
        {
            "incident_number": "INC0010004",
            "similarity_score": 0.8100,
            "assignment_group": "Platform-Support-L2",
            "resolution_time_hours": 2.0,
            "reassignment_count": 0,
            "short_description": "Platform middleware freeze"
        },
        {
            "incident_number": "INC0010005",
            "similarity_score": 0.7500,
            "assignment_group": "Platform-Support-L2",
            "resolution_time_hours": 3.0,
            "reassignment_count": 0,
            "short_description": "ATM transaction rollback"
        }
    ]


def test_confidence_engine_configuration_driven(clean_config):
    """Test HybridConfidenceEngine reads values cleanly from ConfigManager without hardcoding."""
    engine = HybridConfidenceEngine(clean_config)
    assert engine.rf_weight == 0.60
    assert engine.semantic_weight == 0.40
    assert engine.agreement_bonus == 0.10
    assert engine.disagreement_penalty == 0.05
    assert engine.very_high_thresh == 0.88

    # Test agreement calculation
    # base = 0.80 * 0.6 + 0.80 * 0.4 = 0.80. agreement bonus = +0.10 -> 0.90
    conf, tier = engine.calculate_confidence(rf_confidence=0.80, sem_confidence=0.80, agreement=True, top_k_matches=5)
    assert conf == 0.90
    assert tier == "Very High"

    # Test disagreement calculation
    # base = 0.60 * 0.6 + 0.50 * 0.4 = 0.56. disagreement penalty = -0.05 -> 0.51
    conf_dis, tier_dis = engine.calculate_confidence(rf_confidence=0.60, sem_confidence=0.50, agreement=False, top_k_matches=5)
    assert conf_dis == 0.51
    assert tier_dis == "Low"


def test_decision_engine_fusing_agreement(clean_config, sample_rf_prediction, sample_semantic_matches):
    """Test HybridDecisionEngine when RF and Semantic consensus agree."""
    dec_engine = HybridDecisionEngine(config_manager=clean_config)
    decision = dec_engine.fuse_recommendation(sample_rf_prediction, sample_semantic_matches)

    assert decision["recommended_assignment_group"] == "Platform-Support-L2"
    assert decision["agreement"] is True
    assert decision["decision_reason_code"] == "AGREEMENT"
    assert decision["semantic_consensus_count"] == 4  # 4 of 5 belong to Platform-Support-L2
    assert decision["semantic_consensus_pct"] == 80.0

    # Check MTTR blending: RF=3.0, Semantic Mean of (2.5, 3.5, 4.0, 2.0, 3.0)=3.0 -> blended=3.0
    assert decision["estimated_resolution_time_hours"] == 3.0
    assert decision["historical_success_rate"] == 80.0  # 4 matches had 0 reassignments and matched target group


def test_decision_engine_fusing_disagreement_rf_dominant(clean_config, sample_semantic_matches):
    """Test HybridDecisionEngine when RF disagrees with Semantic consensus but RF confidence >= 0.70."""
    rf_pred = {
        "assignment_group": "Security-Operations",
        "confidence_score": 0.78,  # >= 0.70 threshold
        "resolution_time_hours": 5.0
    }
    dec_engine = HybridDecisionEngine(config_manager=clean_config)
    decision = dec_engine.fuse_recommendation(rf_pred, sample_semantic_matches)

    assert decision["recommended_assignment_group"] == "Security-Operations"
    assert decision["agreement"] is False
    assert decision["decision_reason_code"] == "RF_DOMINANT"


def test_decision_engine_fusing_disagreement_semantic_dominant(clean_config, sample_semantic_matches):
    """Test HybridDecisionEngine when RF disagrees and RF confidence < 0.70."""
    rf_pred = {
        "assignment_group": "Security-Operations",
        "confidence_score": 0.62,  # < 0.70 threshold
        "resolution_time_hours": 5.0
    }
    dec_engine = HybridDecisionEngine(config_manager=clean_config)
    decision = dec_engine.fuse_recommendation(rf_pred, sample_semantic_matches)

    assert decision["recommended_assignment_group"] == "Platform-Support-L2"  # Mode of semantic matches
    assert decision["agreement"] is False
    assert decision["decision_reason_code"] == "SEMANTIC_DOMINANT"


def test_reasoning_engine_formatting(clean_config, sample_rf_prediction, sample_semantic_matches):
    """Test HybridReasoningEngine generates clean natural language summaries and Historical Evidence tables."""
    dec_engine = HybridDecisionEngine(config_manager=clean_config)
    decision = dec_engine.fuse_recommendation(sample_rf_prediction, sample_semantic_matches)
    reasoning = HybridReasoningEngine.generate_reasoning(decision, sample_semantic_matches)

    assert "executive_summary" in reasoning
    assert "Platform-Support-L2" in reasoning["executive_summary"]
    assert len(reasoning["bullet_breakdown"]) >= 4
    assert len(reasoning["historical_evidence"]) == 5

    row0 = reasoning["historical_evidence"][0]
    assert row0["rank"] == 1
    assert row0["incident_number"] == "INC0010001"
    assert row0["historical_assignment_group"] == "Platform-Support-L2"
    assert row0["historical_resolution_time"] == "2.50h"


def test_recommendation_engine_orchestration(clean_config, tmp_path, sample_semantic_matches):
    """Test HybridRecommendationEngine end-to-end flow with mocked RF/Semantic engines."""
    # Mock Semantic Engine
    mock_sem = MagicMock()
    mock_sem.find_similar_incidents.return_value = sample_semantic_matches

    # Mock RF Classifier Pipeline
    mock_clf = MagicMock()
    mock_clf.predict.return_value = ["Platform-Support-L2"]
    mock_clf.predict_proba.return_value = [[0.05, 0.90, 0.05]]

    # Mock RF Regressor Pipeline
    mock_reg = MagicMock()
    mock_reg.predict.return_value = [3.2]  # Raw numeric MTTR

    rec_engine = HybridRecommendationEngine(
        models_dir=tmp_path / "models",
        reports_dir=tmp_path / "reports",
        rf_classifier_pipeline=mock_clf,
        rf_regressor_pipeline=mock_reg,
        semantic_engine=mock_sem,
        config_manager=clean_config
    )

    # Execute recommendation from free text
    res = rec_engine.recommend(input_payload="ATM cash deposit failing constantly", top_k=5, export_reports=True)

    assert res["recommended_assignment_group"] == "Platform-Support-L2"
    assert res["confidence_tier"] in ("Very High", "High")
    assert res["agreement"] is True
    assert "historical_evidence" in res
    assert len(res["historical_evidence"]) == 5

    # Verify reports written
    assert (tmp_path / "reports" / "hybrid_prediction.json").exists()
    assert (tmp_path / "reports" / "hybrid_prediction.md").exists()
    assert (tmp_path / "reports" / "hybrid_prediction.csv").exists()


def test_recommendation_engine_file_lock_resilience(clean_config, tmp_path, sample_semantic_matches):
    """Test Windows file-lock fallback writing to `_latest` when primary files raise PermissionError."""
    mock_sem = MagicMock()
    mock_sem.find_similar_incidents.return_value = sample_semantic_matches

    rec_engine = HybridRecommendationEngine(
        models_dir=tmp_path / "models",
        reports_dir=tmp_path / "reports",
        semantic_engine=mock_sem,
        config_manager=clean_config
    )

    rec = rec_engine.recommend("ATM cash jam", export_reports=False)

    # Force PermissionError when exporting
    original_open = open
    def locked_open(file, mode="r", *args, **kwargs):
        path_str = str(file)
        if "hybrid_prediction." in path_str and not "latest" in path_str and "w" in mode:
            raise PermissionError("File locked by Excel")
        return original_open(file, mode, *args, **kwargs)

    with patch("builtins.open", side_effect=locked_open):
        rec_engine.export_reports(rec)

    # Check `_latest` fallback files exist
    assert (tmp_path / "reports" / "hybrid_prediction_latest.json").exists()
    assert (tmp_path / "reports" / "hybrid_prediction_latest.md").exists()
    assert (tmp_path / "reports" / "hybrid_prediction_latest.csv").exists()


def test_sync_features_and_edge_cases(clean_config, tmp_path, sample_semantic_matches):
    """Test _sync_features_for_model branches and missing model fallbacks in HybridRecommendationEngine."""
    mock_sem = MagicMock()
    mock_sem.find_similar_incidents.return_value = sample_semantic_matches

    # Create dummy model with feature_names_in_
    dummy_model = MagicMock()
    dummy_model.feature_names_in_ = np.array(["short_description", "category", "reassignment_count", "custom_num_col"])
    dummy_model.predict.return_value = ["Network-Ops"]
    dummy_model.predict_proba.return_value = [[0.1, 0.8, 0.1]]

    rec_engine = HybridRecommendationEngine(
        models_dir=tmp_path / "models",
        reports_dir=tmp_path / "reports",
        rf_classifier_pipeline=dummy_model,
        rf_regressor_pipeline=None,
        semantic_engine=mock_sem,
        config_manager=clean_config
    )

    import pandas as pd
    df_raw = pd.DataFrame([{"short_description": "test"}])
    df_synced = rec_engine._sync_features_for_model(df_raw, dummy_model)
    assert "category" in df_synced.columns
    assert df_synced["category"].iloc[0] == "UNKNOWN"
    assert "custom_num_col" in df_synced.columns
    assert df_synced["custom_num_col"].iloc[0] == 0

    # Test recommend when regressor is None (falls back to defaults/mean)
    res = rec_engine.recommend("System freeze", top_k=3, export_reports=False)
    assert res["recommended_assignment_group"] == "Network-Ops"
    assert "confidence_score" in res

    # Test when both classifier and regressor are present and log1p transformed (< 6.0)
    dummy_reg = MagicMock()
    dummy_reg.feature_names_in_ = np.array(["short_description", "category"])
    dummy_reg.predict.return_value = [1.609438]  # np.expm1(1.609438) ~ 4.0 hours
    rec_engine.rf_regressor = dummy_reg
    rf_pred = rec_engine._predict_rf({"short_description": "Network lag", "category": "Network"})
    assert rf_pred["resolution_time_hours"] > 3.9 and rf_pred["resolution_time_hours"] < 4.1

    # Test export_reports edge branches
    rec_engine.export_reports(res)
    assert (tmp_path / "reports" / "hybrid_prediction.json").exists()

    # Test when rf_classifier_pipeline and regressor are None (pure semantic fallback branch and load exception handling)
    from src.ml.model_registry import ModelRegistry
    ModelRegistry._instance = None
    (tmp_path / "models").mkdir(parents=True, exist_ok=True)
    (tmp_path / "models" / "random_forest_assignment_group.pkl").touch()
    (tmp_path / "models" / "random_forest_resolution_time_hours.pkl").touch()
    rec_no_models = HybridRecommendationEngine(
        models_dir=tmp_path / "models",
        reports_dir=tmp_path / "reports",
        rf_classifier_pipeline=None,
        rf_regressor_pipeline=None,
        semantic_engine=mock_sem,
        config_manager=clean_config
    )
    res_fallback = rec_no_models.recommend("Emergency power outage", top_k=2, export_reports=False)
    assert "recommended_assignment_group" in res_fallback

    # Test get_model_path coverage directly on ModelRegistry
    assert rec_no_models.model_reg.get_model_path("random_forest_assignment_group") is not None
    assert rec_no_models.model_reg.get_model_path("non_existent_model") is None
    ModelRegistry._instance = None
