"""
Embedding & FAISS Index Registry (`v1.5.0`).

Manages vector generation specifications, neural model boundaries, chunking strategies,
distance metrics, vector counts, index file paths, and SHA256 checksums.
"""

import hashlib
import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from src.utils import robust_open

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EmbeddingIndexMetadata:
    """Formal specification for a registered FAISS vector index."""
    index_name: str
    embedding_model: str
    version: str
    dimension: int
    chunk_strategy: str
    dataset_version: str
    vector_count: int
    distance_metric: str  # 'L2_Euclidean', 'Cosine_InnerProduct'
    faiss_index_version: str
    index_file_path: str
    sha256_checksum: str
    status: str  # 'Active', 'Archived'

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return asdict(self)


class EmbeddingRegistry:
    """
    Singleton-style registry tracking vector indexes and embedding model configurations.
    """

    _instance: Optional["EmbeddingRegistry"] = None

    def __init__(self, base_dir: Optional[str] = None) -> None:
        """Initialize EmbeddingRegistry at base_dir or default indexes/ directory."""
        self.base_dir = Path(base_dir or "indexes")
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.base_dir / "embedding_registry.json"
        self.indexes: Dict[str, EmbeddingIndexMetadata] = {}
        self._load_registry()

    @classmethod
    def get_instance(cls, base_dir: Optional[str] = None) -> "EmbeddingRegistry":
        """Get or create singleton EmbeddingRegistry instance."""
        if cls._instance is None:
            cls._instance = cls(base_dir=base_dir)
        return cls._instance

    @staticmethod
    def compute_sha256(file_path: Path) -> str:
        """Compute SHA256 cryptographic hash of index file on disk."""
        if not file_path.exists():
            return "UNMATERIALIZED_INDEX_SHA256"
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(65536), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def register_index(
        self,
        index_name: str,
        embedding_model: str,
        version: str,
        dimension: int,
        chunk_strategy: str,
        dataset_version: str,
        vector_count: int,
        distance_metric: str,
        faiss_index_version: str,
        index_file_path: str,
        status: str = "Active"
    ) -> EmbeddingIndexMetadata:
        """Register a FAISS vector index."""
        fp = Path(index_file_path)
        sha256 = self.compute_sha256(fp)

        key = f"{index_name}:{version}"
        meta = EmbeddingIndexMetadata(
            index_name=index_name,
            embedding_model=embedding_model,
            version=version,
            dimension=dimension,
            chunk_strategy=chunk_strategy,
            dataset_version=dataset_version,
            vector_count=vector_count,
            distance_metric=distance_metric,
            faiss_index_version=faiss_index_version,
            index_file_path=str(fp),
            sha256_checksum=sha256,
            status=status
        )

        self.indexes[key] = meta
        self._save_registry()
        logger.info(f"Registered embedding index {key} ({vector_count:,} vectors, Dim: {dimension})")
        return meta

    def get_index_metadata(self, index_name: str, version: str = "latest") -> Optional[EmbeddingIndexMetadata]:
        """Retrieve index metadata by name and version."""
        if version != "latest":
            return self.indexes.get(f"{index_name}:{version}")
        candidates = [m for m in self.indexes.values() if m.index_name == index_name and m.status == "Active"]
        if not candidates:
            candidates = [m for m in self.indexes.values() if m.index_name == index_name]
        if not candidates:
            return None
        return candidates[-1]

    def export_markdown(self, output_path: Optional[str] = None) -> Path:
        """Export index registry to embedding_registry.md."""
        out_file = Path(output_path or self.base_dir / "embedding_registry.md")
        out_file.parent.mkdir(parents=True, exist_ok=True)

        lines = [
            "# Embedding & FAISS Vector Index Registry (`v1.5.0`)",
            "**Organization:** First Citizens Bank — Incident Intelligence Platform (IIP)  ",
            f"**Total Registered Vector Indexes:** {len(self.indexes)}  ",
            "**Governance Mandate:** All vector searches must conform to registered dimensions (`384-D`) and distance metrics.  \n",
            "---",
            "\n## Vector Index Catalog Matrix\n",
            "| Index Key | Embedding Model | Dimension | Vector Count | Distance Metric | Dataset Version | Status |",
            "|---|---|:---:|:---:|---|:---:|:---:|"
        ]

        for k, meta in sorted(self.indexes.items()):
            lines.append(
                f"| `{k}` | `{meta.embedding_model}` | `{meta.dimension}` | "
                f"{meta.vector_count:,} | `{meta.distance_metric}` | `{meta.dataset_version}` | **{meta.status}** |"
            )

        lines.extend([
            "\n---",
            "\n## Detailed Index Specifications\n"
        ])

        for k, meta in sorted(self.indexes.items()):
            lines.extend([
                f"### `{k}` (`{meta.embedding_model}`)",
                f"- **Index Path:** `{meta.index_file_path}` (`SHA256: {meta.sha256_checksum[:16]}...`)",
                f"- **Chunking & Tokenization Strategy:** `{meta.chunk_strategy}`",
                f"- **FAISS Index Version:** `{meta.faiss_index_version}` | **Vectors Indexed:** `{meta.vector_count:,}`\n"
            ])

        with open(out_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        logger.info(f"Exported Embedding Registry Markdown to: {out_file}")
        return out_file

    def _load_registry(self) -> None:
        """Load from JSON disk if present."""
        if self.registry_file.exists():
            try:
                with robust_open(self.registry_file, "r") as f:
                    data = json.load(f)
                for k, v in data.get("indexes", {}).items():
                    self.indexes[k] = EmbeddingIndexMetadata(**v)
            except Exception as e:
                logger.error(f"Failed to load existing embedding registry JSON: {e}")

    def _save_registry(self) -> None:
        """Save dictionary to JSON disk."""
        payload = {
            "registry_version": "1.5.0",
            "last_updated": datetime.now().isoformat(),
            "total_indexes": len(self.indexes),
            "indexes": {k: v.to_dict() for k, v in sorted(self.indexes.items())}
        }
        with open(self.registry_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
