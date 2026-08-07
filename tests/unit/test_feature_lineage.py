"""Unit tests for FeatureLineageTracker (`src/data/feature_lineage.py`)."""

from pathlib import Path
import json
import pytest

from src.data.feature_lineage import LineageEdge, FeatureLineageTracker


def test_feature_lineage_tracker_defaults() -> None:
    """Verify default lineage derivations and formula retrieval."""
    tracker = FeatureLineageTracker()
    assert len(tracker.edges) >= 11

    edge = tracker.get_lineage("opened_at_hour_sin")
    assert edge is not None
    assert edge.source_features == ["opened_at_hour"]
    assert "sin" in edge.formula


def test_ancestry_chain() -> None:
    """Verify recursive trace of source features from derived features."""
    tracker = FeatureLineageTracker()
    chain = tracker.get_ancestry_chain("opened_at_hour_sin")
    # opened_at_hour_sin -> opened_at_hour -> opened_at
    assert "opened_at_hour_sin" in chain
    assert "opened_at_hour" in chain
    assert "opened_at" in chain


def test_lineage_export_json_and_md(temp_workspace: Path) -> None:
    """Verify JSON and Markdown export functionality for lineage graph."""
    tracker = FeatureLineageTracker()
    json_path = temp_workspace / "reports" / "feature_lineage.json"
    md_path = temp_workspace / "reports" / "feature_lineage.md"

    out_j = tracker.export_json(str(json_path))
    out_m = tracker.export_markdown(str(md_path))

    assert out_j.exists()
    assert out_m.exists()

    with open(out_j, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["total_derived_features"] == len(tracker.edges)
    assert "opened_at_hour_sin" in data["lineage_graph"]
