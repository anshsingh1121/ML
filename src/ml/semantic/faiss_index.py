"""
Enterprise FAISS Vector Index Manager (`v1.5.0` - Phase 4).

Provides production-grade vector similarity indexing, exact inner product (Cosine) and
inverted file (IVFFlat) approximate search, incremental updates, index persistence,
and cryptographic registration via `EmbeddingRegistry`.

Governance Mandate:
- Offline local execution using `faiss-cpu`.
- Cryptographic SHA256 integrity validation upon index registration.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import faiss
import numpy as np
import pandas as pd
from src.utils import robust_read_csv

from src.ml.embedding_registry import EmbeddingRegistry
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FAISSVectorIndex:
    """
    Enterprise FAISS index controller managing in-memory vector indexing, incremental
    incident additions, top-K nearest neighbor search, and metadata mapping.
    """

    DEFAULT_INDEX_DIR = "indexes"
    DEFAULT_DIMENSION = 384
    DEFAULT_INDEX_TYPE = "FlatIP"  # FlatIP (Cosine for normalized vectors), FlatL2, IVFFlat

    def __init__(
        self,
        dimension: int = DEFAULT_DIMENSION,
        index_type: str = DEFAULT_INDEX_TYPE,
        nlist: int = 100,
        nprobe: int = 10,
        index_dir: Optional[Union[str, Path]] = None,
        index_name: str = "incident_semantic_index",
        version: str = "latest"
    ) -> None:
        """
        Initialize FAISS index controller.

        Args:
            dimension: Vector embedding dimension (typically from TF-IDF + SVD).
            index_type: FAISS index structure (`FlatIP`, `FlatL2`, `IVFFlat`).
            nlist: Number of Voronoi cells/centroids for `IVFFlat`.
            nprobe: Number of centroids to visit during `IVFFlat` search.
            index_dir: Base directory for storing `.index` files and metadata.
            index_name: Unique name for registration and file persistence.
            version: Index version tag (`v1.5.0`, `latest`, etc.).
        """
        self.dimension = dimension
        self.index_type = index_type
        self.nlist = nlist
        self.nprobe = nprobe
        self.index_dir = Path(index_dir or self.DEFAULT_INDEX_DIR)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.index_name = index_name
        self.version = version

        self.index: Optional[faiss.Index] = None
        self.metadata_df: pd.DataFrame = pd.DataFrame()
        self.registry = EmbeddingRegistry.get_instance(base_dir=str(self.index_dir))

    def create_index(self, initial_embeddings: Optional[np.ndarray] = None, add_to_index: bool = True) -> faiss.Index:
        """
        Build and initialize a new FAISS vector index (`FlatIP`, `FlatL2`, or `IVFFlat`).
        If initial_embeddings is provided (`IVFFlat` requirement), train the index immediately.
        """
        logger.info(f"Initializing FAISS index '{self.index_name}' (type={self.index_type}, dim={self.dimension})...")

        if self.index_type.upper() == "FLATIP":
            # Inner Product (Cosine similarity when vectors are L2-normalized)
            self.index = faiss.IndexFlatIP(self.dimension)
        elif self.index_type.upper() == "FLATL2":
            # Euclidean L2 distance
            self.index = faiss.IndexFlatL2(self.dimension)
        elif self.index_type.upper() == "IVFFLAT":
            quantizer = faiss.IndexFlatIP(self.dimension)
            # Adjust nlist safely if initial dataset is small
            num_vecs = len(initial_embeddings) if initial_embeddings is not None else 0
            safe_nlist = min(self.nlist, max(1, int(num_vecs / 39))) if num_vecs > 0 else self.nlist
            if safe_nlist < 1:
                safe_nlist = 1

            ivf_index = faiss.IndexIVFFlat(quantizer, self.dimension, safe_nlist, faiss.METRIC_INNER_PRODUCT)
            ivf_index.nprobe = min(self.nprobe, safe_nlist)

            if initial_embeddings is not None and len(initial_embeddings) > 0:
                logger.info(f"Training IVFFlat quantizer across {len(initial_embeddings):,} vectors (nlist={safe_nlist})...")
                ivf_index.train(initial_embeddings.astype(np.float32))

            self.index = ivf_index
        else:
            raise ValueError(f"Unsupported FAISS index type: {self.index_type}. Must be FlatIP, FlatL2, or IVFFlat.")

        if add_to_index and initial_embeddings is not None and len(initial_embeddings) > 0:
            self.add_embeddings(initial_embeddings, None)

        return self.index

    def add_embeddings(
        self,
        embeddings: np.ndarray,
        new_metadata_df: Optional[pd.DataFrame] = None
    ) -> int:
        """
        Incrementally add vectors and aligned metadata to the active FAISS index.

        Returns:
            Total vector count in the index after addition.
        """
        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Embedding dimension mismatch: expected {self.dimension}, got {embeddings.shape[1]}")

        vectors = embeddings.astype(np.float32)

        # Initialize index if not created yet
        if self.index is None:
            self.create_index(initial_embeddings=vectors if self.index_type.upper() == "IVFFLAT" else None, add_to_index=False)

        # If index is IVFFlat and not trained, train it now
        if not self.index.is_trained:
            logger.info(f"Index not trained; training IVFFlat quantizer on {len(vectors):,} vectors...")
            self.index.train(vectors)

        # Add vectors to FAISS
        self.index.add(vectors)

        # Update metadata table
        if new_metadata_df is not None:
            if len(new_metadata_df) != len(vectors):
                raise ValueError(f"Metadata row count ({len(new_metadata_df)}) must match vector count ({len(vectors)})")
            if self.metadata_df.empty:
                self.metadata_df = new_metadata_df.copy().reset_index(drop=True)
            else:
                self.metadata_df = pd.concat([self.metadata_df, new_metadata_df.copy()], ignore_index=True)

        total_count = self.index.ntotal
        logger.info(f"Added {len(vectors):,} vectors. Total indexed vectors: {total_count:,}")
        return total_count

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search the FAISS vector index for the Top-K most semantically similar records.

        Args:
            query_vector: Dense query vector of shape `(1, D)` or `(D,)`.
            top_k: Number of nearest neighbors to retrieve.

        Returns:
            List of structured match dictionaries containing similarity score and all metadata fields.
        """
        if self.index is None or self.index.ntotal == 0:
            raise RuntimeError("Cannot search: FAISS index is uninitialized or contains 0 vectors.")

        q_vec = query_vector.astype(np.float32)
        if q_vec.ndim == 1:
            q_vec = q_vec.reshape(1, -1)
        if q_vec.shape[1] != self.dimension:
            raise ValueError(f"Query vector dimension {q_vec.shape[1]} does not match index dimension {self.dimension}")

        # Ensure nprobe is set for IVFFlat during query execution
        if hasattr(self.index, "nprobe"):
            self.index.nprobe = min(self.nprobe, getattr(self.index, "nlist", self.nprobe))

        scores, indices = self.index.search(q_vec, top_k)

        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if idx < 0 or idx >= len(self.metadata_df):
                continue

            meta_row = self.metadata_df.iloc[idx].to_dict()
            meta_row["similarity_score"] = float(score)
            meta_row["rank"] = rank
            results.append(meta_row)

        return results

    def save_index(
        self,
        index_name: Optional[str] = None,
        version: Optional[str] = None,
        dataset_version: str = "v1.5.0",
        embedding_model: str = "tfidf-svd-384"
    ) -> Tuple[Path, Path]:
        """
        Persist the FAISS binary `.index` file and structured `.csv` metadata table to disk,
        and register the index with cryptographic SHA256 checksums inside `EmbeddingRegistry`.
        """
        if self.index is None:
            raise RuntimeError("Cannot save uninitialized FAISS index.")

        name = index_name or self.index_name
        ver = version or self.version

        index_file = self.index_dir / f"{name}_{ver}.index"
        meta_file = self.index_dir / f"{name}_{ver}_metadata.csv"

        faiss.write_index(self.index, str(index_file))
        if not self.metadata_df.empty:
            self.metadata_df.to_csv(meta_file, index=False, encoding="utf-8")

        dist_metric = "Cosine_InnerProduct" if "IP" in self.index_type.upper() else "L2_Euclidean"

        # Register in EmbeddingRegistry
        self.registry.register_index(
            index_name=name,
            embedding_model=embedding_model,
            version=ver,
            dimension=self.dimension,
            chunk_strategy="MultiAttribute_CompositeString",
            dataset_version=dataset_version,
            vector_count=self.index.ntotal,
            distance_metric=dist_metric,
            faiss_index_version=f"FAISS-{faiss.__version__}",
            index_file_path=str(index_file),
            status="Active"
        )
        self.registry.export_markdown()

        logger.info(f"Successfully saved FAISS index to {index_file} ({self.index.ntotal:,} vectors) and registered in EmbeddingRegistry.")
        return index_file, meta_file

    def load_index(
        self,
        index_name: Optional[str] = None,
        version: Optional[str] = None
    ) -> faiss.Index:
        """
        Load an existing FAISS binary `.index` and corresponding `.csv` metadata from disk.
        """
        name = index_name or self.index_name
        ver = version or self.version

        index_file = self.index_dir / f"{name}_{ver}.index"
        meta_file = self.index_dir / f"{name}_{ver}_metadata.csv"

        if not index_file.exists():
            # Check if standard name without version suffix exists
            index_file_alt = self.index_dir / f"{name}.index"
            meta_file_alt = self.index_dir / f"{name}_metadata.csv"
            if index_file_alt.exists():
                index_file, meta_file = index_file_alt, meta_file_alt
            else:
                raise FileNotFoundError(f"FAISS index file not found at {index_file}")

        self.index = faiss.read_index(str(index_file))
        self.dimension = self.index.d
        if meta_file.exists():
            self.metadata_df = robust_read_csv(meta_file)
        else:
            logger.warning(f"Metadata file {meta_file} not found; searching will return vector indices only.")

        logger.info(f"Loaded FAISS index '{name}:{ver}' from {index_file}. Total vectors: {self.index.ntotal:,}")
        return self.index
