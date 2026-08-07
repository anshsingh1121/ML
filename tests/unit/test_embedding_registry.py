"""Unit tests for EmbeddingRegistry (`src/ml/embedding_registry.py`)."""

from pathlib import Path
import json
import pytest

from src.ml.embedding_registry import EmbeddingIndexMetadata, EmbeddingRegistry


def test_embedding_registry_registration_and_export(temp_workspace: Path) -> None:
    """Verify FAISS vector index registration and markdown export."""
    idx_dir = temp_workspace / "indexes"
    reg = EmbeddingRegistry(base_dir=str(idx_dir))

    dummy_faiss = idx_dir / "incidents_v1.faiss"
    with open(dummy_faiss, "wb") as f:
        f.write(b"MOCK_FAISS_INDEX_BINARY_DATA_384_VECTORS")

    meta = reg.register_index(
        index_name="incidents_semantic_index",
        embedding_model="all-MiniLM-L6-v2",
        version="v1.0",
        dimension=384,
        chunk_strategy="truncate_256_tokens_strip_html",
        dataset_version="v1",
        vector_count=10000,
        distance_metric="L2_Euclidean",
        faiss_index_version="1.7.4",
        index_file_path=str(dummy_faiss)
    )

    assert meta.dimension == 384
    assert meta.vector_count == 10000
    assert reg.get_index_metadata("incidents_semantic_index") is not None

    md_path = reg.export_markdown()
    assert md_path.exists()
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()
    assert "all-MiniLM-L6-v2" in content
    assert "384" in content
