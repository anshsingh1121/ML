"""Unit tests for FeatureRegistry (`src/data/feature_registry.py`)."""

from pathlib import Path
import json
import pytest

from src.data.feature_registry import FeatureDefinition, FeatureRegistry


@pytest.fixture(autouse=True)
def reset_feature_registry() -> None:
    """Reset singleton instance before each test to guarantee isolated verification."""
    FeatureRegistry.reset_instance()


def test_feature_registry_singleton_and_defaults() -> None:
    """Verify FeatureRegistry singleton initialization and default 49 attributes."""
    reg1 = FeatureRegistry.get_instance()
    reg2 = FeatureRegistry.get_instance()
    assert reg1 is reg2
    assert len(reg1.list_all_features()) == 51

    feat = reg1.get_feature("assignment_group")
    assert feat is not None
    assert feat.business_name == "Assignment Group"
    assert feat.target_leakage_classification == "safe"
    assert feat.catboost_usage == "target_assignment_group"


def test_feature_registry_queries() -> None:
    """Verify querying features by leakage, RF usage, embedding usage, and FAISS usage."""
    reg = FeatureRegistry.get_instance()

    safe_feats = reg.get_features_by_leakage("safe")
    blocked_feats = reg.get_features_by_leakage("blocked")
    assert len(safe_feats) > 0
    assert len(blocked_feats) > 0
    assert "close_notes" in [f.technical_name for f in blocked_feats]
    assert "category" in [f.technical_name for f in safe_feats]

    rf_preds = reg.get_catboost_predictors()
    assert "priority" in rf_preds
    assert "category" in rf_preds
    assert "close_notes" not in rf_preds

    embed_feats = reg.get_embedding_features()
    assert "short_description" in embed_feats
    assert "description" in embed_feats

    faiss_feats = reg.get_faiss_metadata_features()
    assert "cmdb_ci" in faiss_feats
    assert "business_service" in faiss_feats


def test_feature_registry_export_json_and_md(tmp_path: Path) -> None:
    """Verify JSON and Markdown export functionality."""
    reg = FeatureRegistry.get_instance()
    json_path = tmp_path / "reports" / "feature_registry.json"
    md_path = tmp_path / "reports" / "feature_registry.md"

    out_j = reg.export_json(str(json_path))
    out_m = reg.export_markdown(str(md_path))

    assert out_j.exists()
    assert out_m.exists()

    with open(out_j, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["total_features"] == 51
    assert "assignment_group" in data["features"]
