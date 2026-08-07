"""
Enterprise Semantic Embedding Generator (`v1.5.0` - Phase 4).

Generates dense neural representations (`384-D`) of ServiceNow incidents using a local
Scikit-Learn TF-IDF + TruncatedSVD pipeline. Ensures exact separation of
dense embedding matrices (`.npy`) and structured incident metadata (`.csv`).

Governance Mandate:
- Zero cloud data egress (`device='cpu'` by default, local disk cache).
- Standardized multi-field semantic composition:
  `[Category | Subcategory] [Service | CI] [Priority] Short Description. Description`
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import normalize

from src.utils.logger import get_logger
from src.utils import robust_read_csv

logger = get_logger(__name__)


class SemanticEmbeddingGenerator:
    """
    Enterprise embedding engine responsible for transforming raw ServiceNow incident records
    and natural language query strings into L2-normalized dense vector representations (`384-D`).
    """

    DEFAULT_MODEL_NAME = "tfidf-svd-384"
    DEFAULT_CACHE_DIR = "models/embeddings"

    def __init__(
        self,
        model_name: Optional[str] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        device: str = "cpu",
        normalize_output: bool = True,
        n_components: int = 384
    ) -> None:
        """
        Initialize the embedding generator.

        Args:
            model_name: Identifier for the local model.
            cache_dir: Local storage directory for pre-trained weights and embeddings.
            device: Compute device (kept for compatibility).
            normalize_output: If True, apply L2 normalization to output embeddings (enables Cosine/IP distance).
            n_components: Dimension of the output embeddings.
        """
        self.model_name = model_name or self.DEFAULT_MODEL_NAME
        self.cache_dir = Path(cache_dir or self.DEFAULT_CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.normalize_output = normalize_output
        self.n_components = n_components
        
        self.model_path = self.cache_dir / f"{self.model_name}.pkl"
        self._pipeline: Optional[Pipeline] = None

    @property
    def model(self) -> Pipeline:
        """Lazy-load and return the Scikit-learn Pipeline."""
        if self._pipeline is None:
            if self.model_path.exists():
                logger.info(f"Loading local embedding pipeline '{self.model_name}' from {self.model_path}...")
                try:
                    self._pipeline = joblib.load(self.model_path)
                    logger.info(f"Successfully loaded embedding pipeline. Dimension: {self.n_components}")
                except Exception as e:
                    logger.error(f"Failed to load embedding pipeline '{self.model_name}': {e}")
                    raise RuntimeError(f"Embedding model initialization failure: {e}") from e
            else:
                logger.warning(f"Pipeline {self.model_path} does not exist. It will be trained upon calling embed_dataframe.")
        return self._pipeline

    def get_embedding_dimension(self) -> int:
        """Return exact vector dimension (`384`)."""
        return self.n_components

    @staticmethod
    def construct_semantic_text(record: Union[pd.Series, Dict[str, Any]]) -> str:
        """
        Construct a structured, domain-rich semantic document string by intelligently
        combining critical IT incident attributes.
        """
        if isinstance(record, pd.Series):
            record = record.to_dict()

        def _clean_val(val: Any, default: str = "Unknown") -> str:
            if val is None or pd.isna(val):
                return default
            val_str = str(val).strip()
            return val_str if val_str and val_str.lower() != "nan" else default

        short_desc = _clean_val(record.get("short_description"), "No short description")
        desc = _clean_val(record.get("description"), "No detailed description")
        category = _clean_val(record.get("category"), "General")
        subcategory = _clean_val(record.get("subcategory"), "General")
        service = _clean_val(record.get("business_service"), "Enterprise Service")
        ci = _clean_val(record.get("cmdb_ci"), "General CI")
        priority = _clean_val(record.get("priority"), "P3 - Moderate")

        # Strip redundant punctuation if description duplicates short description
        if desc.lower().startswith(short_desc.lower()) and len(desc) > len(short_desc):
            desc_part = desc
        elif desc.lower() == short_desc.lower():
            desc_part = short_desc
        else:
            desc_part = f"{short_desc}. {desc}"

        semantic_text = (
            f"[Category: {category} | Subcategory: {subcategory}] "
            f"[Service: {service} | CI: {ci}] "
            f"[Priority: {priority}] {desc_part}"
        )
        return semantic_text.strip()

    def embed_dataframe(
        self,
        df: pd.DataFrame,
        batch_size: int = 64,
        show_progress_bar: bool = True
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Generate embeddings for an entire historical or production incident DataFrame.
        """
        if df.empty:
            raise ValueError("Cannot generate embeddings from an empty DataFrame.")

        logger.info(f"Constructing semantic text documents across {len(df):,} incident records...")
        semantic_texts = [self.construct_semantic_text(row) for _, row in df.iterrows()]

        # Only train pipeline if it's not already trained
        try:
            _ = self.model  # Trigger lazy load if it exists
        except RuntimeError:
            pass

        if self._pipeline is None:
            logger.info(f"Training TF-IDF + TruncatedSVD pipeline across {len(semantic_texts):,} documents...")
            tfidf = TfidfVectorizer(max_features=25000, stop_words="english")
            tfidf_matrix = tfidf.fit_transform(semantic_texts)
            actual_features = tfidf_matrix.shape[1]
            
            # Adaptive components
            max_possible = min(len(semantic_texts) - 1, actual_features - 1, self.n_components)
            max_possible = max(1, max_possible)
            
            self._pipeline = Pipeline([
                ("tfidf", tfidf),
                ("svd", TruncatedSVD(n_components=max_possible, random_state=42))
            ])
            embeddings = self._pipeline.fit_transform(semantic_texts)
        else:
            logger.info(f"Using pre-trained embedding pipeline across {len(semantic_texts):,} documents...")
            embeddings = self._pipeline.transform(semantic_texts)

        if self.normalize_output:
            embeddings = normalize(embeddings, norm='l2', axis=1)

        embeddings = embeddings.astype(np.float32)

        joblib.dump(self._pipeline, self.model_path)
        logger.info(f"Saved local embedding pipeline to {self.model_path}")

        # Build separate metadata table aligned index-for-index with embeddings
        meta_cols = [
            "incident_number", "assignment_group", "priority",
            "business_service", "short_description", "description",
            "resolution_time_hours", "category", "subcategory", "cmdb_ci"
        ]
        meta_dict = {}
        for col in meta_cols:
            if col in df.columns:
                meta_dict[col] = df[col].values
            else:
                meta_dict[col] = ["Unknown"] * len(df)

        meta_dict["semantic_text"] = semantic_texts
        metadata_df = pd.DataFrame(meta_dict)

        logger.info(f"Embedding complete. Matrix shape: {embeddings.shape}, Metadata records: {len(metadata_df):,}")
        return embeddings, metadata_df

    def embed_text(self, text_or_texts: Union[str, List[str]]) -> np.ndarray:
        """
        Encode raw natural language text (e.g., query string or free text) into L2-normalized vectors.
        """
        if self._pipeline is None:
            self.model  # forces load

        if self._pipeline is None:
            raise RuntimeError("Embedding pipeline not fitted yet. Run embed_dataframe first.")

        if isinstance(text_or_texts, str):
            texts = [text_or_texts]
        else:
            texts = text_or_texts

        embeddings = self._pipeline.transform(texts)
        if self.normalize_output:
            embeddings = normalize(embeddings, norm='l2', axis=1)

        return embeddings.astype(np.float32)

    def save_embeddings(
        self,
        embeddings: np.ndarray,
        metadata_df: pd.DataFrame,
        output_dir: Optional[Union[str, Path]] = None,
        prefix: str = "incident"
    ) -> Tuple[Path, Path]:
        """
        Store dense embeddings (`.npy`) and structured metadata (`.csv` / `.parquet`)
        separately on local disk to enforce clean storage isolation.
        """
        out_dir = Path(output_dir or self.cache_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        npy_path = out_dir / f"{prefix}_embeddings.npy"
        meta_path = out_dir / f"{prefix}_metadata.csv"

        np.save(npy_path, embeddings)
        metadata_df.to_csv(meta_path, index=False, encoding="utf-8")

        # Attempt to save Parquet if available for high-speed indexing
        try:
            parquet_path = out_dir / f"{prefix}_metadata.parquet"
            metadata_df.to_parquet(parquet_path, index=False)
            logger.debug(f"Saved parquet metadata to {parquet_path}")
        except Exception:
            pass

        logger.info(f"Saved separated embedding artifacts to:\n  - Vector Matrix: {npy_path} ({embeddings.shape})\n  - Metadata Table: {meta_path} ({len(metadata_df):,} records)")
        return npy_path, meta_path

    @staticmethod
    def load_embeddings(
        input_dir: Union[str, Path] = "models/embeddings",
        prefix: str = "incident"
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Load separated dense embeddings and structured metadata from disk.
        """
        in_dir = Path(input_dir)
        npy_path = in_dir / f"{prefix}_embeddings.npy"
        meta_path = in_dir / f"{prefix}_metadata.csv"

        if not npy_path.exists() or not meta_path.exists():
            raise FileNotFoundError(f"Required embedding files missing in {in_dir}: {npy_path.name} / {meta_path.name}")

        embeddings = np.load(npy_path)
        metadata_df = robust_read_csv(meta_path)

        logger.info(f"Loaded embeddings matrix {embeddings.shape} and {len(metadata_df):,} metadata rows from {in_dir}")
        return embeddings, metadata_df
