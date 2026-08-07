"""
Unit & Verification Tests for Enterprise Semantic Similarity Engine (`v1.5.0` - Phase 4).

Covers:
- `SemanticEmbeddingGenerator`: Document string construction, numpy matrix normalization, disk separation.
- `FAISSVectorIndex`: FlatIP/FlatL2/IVFFlat creation, incremental vector ingestion, Top-K search, and persistence.
- `SemanticSimilarityEngine`: End-to-end indexing, incident lookup, free-text similarity query, and report exports.
"""

from pathlib import Path
from typing import Any, Dict, List
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.ml.semantic.embedding_generator import SemanticEmbeddingGenerator
from src.ml.semantic.faiss_index import FAISSVectorIndex
from src.ml.semantic.similarity_engine import SemanticSimilarityEngine


class TestSemanticEmbeddingGenerator:
    """Verification suite for neural text formatting and embedding generation."""

    def test_construct_semantic_text_dict(self):
        record = {
            "short_description": "Database login failure",
            "description": "User cannot connect to production Oracle instance via JDBC.",
            "category": "Database",
            "subcategory": "Oracle",
            "business_service": "Retail Online Banking",
            "cmdb_ci": "db-prod-01",
            "priority": "P1 - High"
        }
        text = SemanticEmbeddingGenerator.construct_semantic_text(record)
        assert "[Category: Database | Subcategory: Oracle]" in text
        assert "[Service: Retail Online Banking | CI: db-prod-01]" in text
        assert "[Priority: P1 - High]" in text
        assert "Database login failure. User cannot connect to production Oracle instance via JDBC." in text

    def test_construct_semantic_text_overlap(self):
        record = {
            "short_description": "Network outage on Switch A",
            "description": "Network outage on Switch A during maintenance.",
            "category": "Network",
            "subcategory": "Switch",
            "business_service": "Core Routing",
            "cmdb_ci": "sw-prod-01",
            "priority": "P2 - High"
        }
        text = SemanticEmbeddingGenerator.construct_semantic_text(record)
        # Verify it doesn't duplicate `Network outage on Switch A. Network outage on Switch A during maintenance.`
        assert text.endswith("Network outage on Switch A during maintenance.")

    def test_construct_semantic_text_missing_vals(self):
        record = {"short_description": None, "category": np.nan, "priority": ""}
        text = SemanticEmbeddingGenerator.construct_semantic_text(record)
        assert "[Category: General | Subcategory: General]" in text
        assert "No short description." in text

    def test_embed_dataframe_and_storage(self, tmp_path):

        df = pd.DataFrame([
            {"incident_number": "INC001", "short_description": "A", "description": "B", "category": "DB"},
            {"incident_number": "INC002", "short_description": "C", "description": "D", "category": "Net"}
        ])

        embedder = SemanticEmbeddingGenerator(cache_dir=tmp_path)
        embeddings, meta_df = embedder.embed_dataframe(df, batch_size=2)

        assert embeddings.shape[0] == 2
        assert len(meta_df) == 2
        assert meta_df["incident_number"].tolist() == ["INC001", "INC002"]

        npy_path, meta_path = embedder.save_embeddings(embeddings, meta_df, output_dir=tmp_path, prefix="test")
        assert npy_path.exists()
        assert meta_path.exists()

        loaded_emb, loaded_meta = SemanticEmbeddingGenerator.load_embeddings(input_dir=tmp_path, prefix="test")
        assert loaded_emb.shape == embeddings.shape
        assert len(loaded_meta) == 2


class TestFAISSVectorIndex:
    """Verification suite for FAISS vector indexing, incremental addition, and search."""

    def test_create_and_search_flat_ip(self, tmp_path):
        dim = 8
        faiss_idx = FAISSVectorIndex(dimension=dim, index_type="FlatIP", index_dir=tmp_path, index_name="unit_test")
        vectors = np.eye(dim, dtype=np.float32)
        meta_df = pd.DataFrame([{"incident_number": f"INC{i:03d}", "assignment_group": "DB_Group"} for i in range(dim)])

        total = faiss_idx.add_embeddings(vectors, meta_df)
        assert total == dim
        assert faiss_idx.index.ntotal == dim

        # Query exact match with first vector
        query = vectors[0:1]
        results = faiss_idx.search(query, top_k=3)
        assert len(results) == 3
        assert results[0]["incident_number"] == "INC000"
        assert results[0]["rank"] == 1
        assert pytest.approx(results[0]["similarity_score"], 1e-5) == 1.0

    def test_create_ivf_flat(self, tmp_path):
        dim = 16
        vectors = np.random.randn(50, dim).astype(np.float32)
        meta_df = pd.DataFrame([{"incident_number": f"INC{i}"} for i in range(50)])

        faiss_idx = FAISSVectorIndex(dimension=dim, index_type="IVFFlat", nlist=5, index_dir=tmp_path)
        faiss_idx.add_embeddings(vectors, meta_df)
        assert faiss_idx.index.is_trained
        assert faiss_idx.index.ntotal == 50

    def test_save_and_load_index(self, tmp_path):
        dim = 4
        faiss_idx = FAISSVectorIndex(dimension=dim, index_type="FlatL2", index_dir=tmp_path, index_name="save_test", version="v1")
        vectors = np.ones((5, dim), dtype=np.float32)
        meta = pd.DataFrame([{"incident_number": f"I{i}"} for i in range(5)])
        faiss_idx.add_embeddings(vectors, meta)

        idx_file, meta_file = faiss_idx.save_index()
        assert idx_file.exists()
        assert meta_file.exists()

        loaded_controller = FAISSVectorIndex(dimension=dim, index_dir=tmp_path, index_name="save_test", version="v1")
        loaded_controller.load_index()
        assert loaded_controller.index.ntotal == 5
        assert len(loaded_controller.metadata_df) == 5


class TestSemanticSimilarityEngine:
    """Verification suite for end-to-end semantic similarity retrieval and formal reporting."""

    def test_build_and_query_engine(self, tmp_path):
        embedder = SemanticEmbeddingGenerator(cache_dir=tmp_path / "embed")
        faiss_idx = FAISSVectorIndex(dimension=4, index_type="FlatIP", index_dir=tmp_path / "idx")
        engine = SemanticSimilarityEngine(embedding_generator=embedder, faiss_index=faiss_idx, reports_dir=tmp_path / "reports")

        df = pd.DataFrame([
            {"incident_number": "INC001", "short_description": "ATM cash jam", "assignment_group": "ATM_Ops", "priority": "P1"},
            {"incident_number": "INC002", "short_description": "Database timeout", "assignment_group": "DB_Team", "priority": "P2"},
            {"incident_number": "INC003", "short_description": "Network latency", "assignment_group": "Net_Team", "priority": "P3"},
            {"incident_number": "INC004", "short_description": "Login password reset", "assignment_group": "ServiceDesk", "priority": "P4"}
        ])

        count = engine.build_index_from_dataframe(df, index_name="test_sim_index")
        assert count == 4

        # Query by incident number
        res_inc = engine.find_similar_incidents("INC001", top_k=2, export_reports=True)
        assert len(res_inc) == 2
        assert res_inc[0]["incident_number"] == "INC001"
        assert (tmp_path / "reports" / "similarity_results.csv").exists()
        assert (tmp_path / "reports" / "similarity_results.md").exists()

        # Query by free text
        res_txt = engine.find_similar_incidents("ATM withdrawal failing", top_k=2, export_reports=True)
        assert len(res_txt) == 2
        assert "similarity_score" in res_txt[0]
