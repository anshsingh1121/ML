"""
Enterprise Semantic Similarity Engine (`v1.5.0` - Phase 4).

Orchestrates zero-cloud neural text embedding (`SemanticEmbeddingGenerator`) and
high-performance vector retrieval (`FAISSVectorIndex`) to identify historical incident
precedents, predict possible assignment routing, and estimate resolution times based
on semantic distance.

Governance Mandate:
- Offline local execution (zero cloud API dependencies).
- Standardized reporting (`reports/similarity_results.csv` & `.md`).
- Resilience against Windows file locks via automatic fallback (`_latest`).
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from src.ml.semantic.embedding_generator import SemanticEmbeddingGenerator
from src.ml.semantic.faiss_index import FAISSVectorIndex
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SemanticSimilarityEngine:
    """
    High-level similarity retrieval controller connecting neural embeddings,
    FAISS vector search, and standardized audit reporting.
    """

    def __init__(
        self,
        embedding_generator: Optional[SemanticEmbeddingGenerator] = None,
        faiss_index: Optional[FAISSVectorIndex] = None,
        reports_dir: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Initialize the Semantic Similarity Engine.

        Args:
            embedding_generator: Controller for `TF-IDF + SVD` vector creation.
            faiss_index: Controller for FAISS similarity index.
            reports_dir: Output directory for similarity results and audit tables.
        """
        self.embedding_generator = embedding_generator or SemanticEmbeddingGenerator()
        self.faiss_index = faiss_index or FAISSVectorIndex()
        self.reports_dir = Path(reports_dir or "reports")
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def build_index_from_dataframe(
        self,
        df: pd.DataFrame,
        index_name: str = "incident_semantic_index",
        version: str = "latest",
        batch_size: int = 64
    ) -> int:
        """
        End-to-end pipeline: embed all historical records in df, initialize and populate FAISS,
        save binary artifacts to disk, and register in EmbeddingRegistry.
        """
        logger.info(f"Starting end-to-end semantic index build across {len(df):,} records...")

        try:
            embeddings, metadata_df = self.embedding_generator.load_embeddings(prefix=index_name)
            logger.info(f"Loaded {len(metadata_df):,} pre-computed embeddings from disk. Skipping redundant generation.")
        except FileNotFoundError:
            embeddings, metadata_df = self.embedding_generator.embed_dataframe(
                df, batch_size=batch_size, show_progress_bar=True
            )
            # Save separated embedding matrix and metadata to disk
            self.embedding_generator.save_embeddings(embeddings, metadata_df, prefix=index_name)

        # 2. Build and populate FAISS index
        self.faiss_index.index_name = index_name
        self.faiss_index.version = version
        self.faiss_index.dimension = embeddings.shape[1]
        self.faiss_index.create_index(initial_embeddings=embeddings)
        self.faiss_index.metadata_df = metadata_df

        # 3. Save index to disk and register checksums
        self.faiss_index.save_index(
            index_name=index_name,
            version=version,
            embedding_model=self.embedding_generator.model_name
        )

        logger.info(f"Successfully built and registered semantic index '{index_name}:{version}' ({len(metadata_df):,} vectors).")
        return len(metadata_df)

    def find_similar_incidents(
        self,
        query: Union[str, Dict[str, Any], pd.Series],
        top_k: int = 10,
        export_reports: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Retrieve Top-K semantically similar incidents from the active FAISS index.

        Args:
            query: Can be:
                   - An Incident Number string (`INC0000012`) -> retrieves precomputed or matches text
                   - A free natural language string (`"ATM cash withdrawal failing"`)
                   - A dictionary or pd.Series representing a new incoming incident record.
            top_k: Number of nearest historical precedents to return.
            export_reports: If True, generate `reports/similarity_results.csv` and `.md`.

        Returns:
            List of structured match dictionaries containing exact required fields:
            `incident_number`, `similarity_score`, `assignment_group`, `priority`,
            `business_service`, `short_description`, `resolution_time`.
        """
        if self.faiss_index.index is None or self.faiss_index.index.ntotal == 0:
            # Attempt to auto-load default index if not loaded in memory
            try:
                self.faiss_index.load_index()
            except Exception as e:
                raise RuntimeError(
                    "FAISS index is uninitialized and could not be loaded from disk. "
                    "Please run `python main.py index` to build the semantic index first."
                ) from e

        query_text = ""
        query_label = "Free Text / Custom Query"
        query_vector: Optional[np.ndarray] = None

        if isinstance(query, str):
            query_str = query.strip()
            # Check if query string matches an existing incident_number inside metadata_df
            if (
                not self.faiss_index.metadata_df.empty
                and "incident_number" in self.faiss_index.metadata_df.columns
                and query_str in self.faiss_index.metadata_df["incident_number"].values
            ):
                query_label = f"Incident Number: {query_str}"
                row_idx = self.faiss_index.metadata_df.index[
                    self.faiss_index.metadata_df["incident_number"] == query_str
                ][0]
                query_text = str(self.faiss_index.metadata_df.iloc[row_idx].get("semantic_text", query_str))
                logger.info(f"Resolved query '{query_str}' to existing historical record. Generating query vector...")
                query_vector = self.embedding_generator.embed_text(query_text)
            else:
                query_label = f"Text Query: '{query_str[:50]}...'"
                query_text = query_str
                query_vector = self.embedding_generator.embed_text(query_text)

        elif isinstance(query, (dict, pd.Series)):
            if isinstance(query, pd.Series):
                query = query.to_dict()
            query_label = f"Incident Object: {query.get('incident_number', 'New Incident')}"
            query_text = SemanticEmbeddingGenerator.construct_semantic_text(query)
            query_vector = self.embedding_generator.embed_text(query_text)
        else:
            raise ValueError("Query must be a string (`INC...` / free text), dictionary, or pandas Series.")

        logger.info(f"Executing FAISS Top-{top_k} similarity search for {query_label}...")
        raw_results = self.faiss_index.search(query_vector, top_k=top_k)

        # Standardize strictly to required output fields
        formatted_results = []
        for match in raw_results:
            res_time = match.get("resolution_time_hours", match.get("resolution_time", 0.0))
            try:
                res_time_float = round(float(res_time), 2)
            except (ValueError, TypeError):
                res_time_float = 0.0

            formatted_match = {
                "rank": match.get("rank", 0),
                "incident_number": str(match.get("incident_number", "UNKNOWN")),
                "similarity_score": round(float(match.get("similarity_score", 0.0)), 6),
                "assignment_group": str(match.get("assignment_group", "Unassigned")),
                "priority": str(match.get("priority", "P3 - Moderate")),
                "business_service": str(match.get("business_service", "General Service")),
                "short_description": str(match.get("short_description", "No short description")),
                "resolution_time": res_time_float
            }
            formatted_results.append(formatted_match)

        if export_reports and formatted_results:
            self._export_similarity_reports(formatted_results, query_label, query_text)

        return formatted_results

    def _export_similarity_reports(
        self,
        results: List[Dict[str, Any]],
        query_label: str,
        query_text: str
    ) -> None:
        """
        Export standardized CSV and Markdown reports (`reports/similarity_results.csv` & `.md`)
        with complete PermissionError resilience.
        """
        df_res = pd.DataFrame(results)

        # 1. Export CSV
        csv_path = self.reports_dir / "similarity_results.csv"
        try:
            df_res.to_csv(csv_path, index=False, encoding="utf-8")
        except PermissionError:
            csv_path = self.reports_dir / "similarity_results_latest.csv"
            df_res.to_csv(csv_path, index=False, encoding="utf-8")
            logger.warning(f"Primary similarity CSV locked by another process; exported to {csv_path}")

        # 2. Export Markdown
        md_path = self.reports_dir / "similarity_results.md"
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Calculate consensus statistics
        avg_sim = df_res["similarity_score"].mean() if not df_res.empty else 0.0
        top_group = df_res["assignment_group"].mode()[0] if not df_res.empty and "assignment_group" in df_res.columns else "N/A"
        avg_time = df_res["resolution"].mean() if "resolution" in df_res.columns else (
            df_res["resolution_time"].mean() if "resolution_time" in df_res.columns else 0.0
        )

        lines = [
            "# Enterprise Semantic Similarity Retrieval Report (`v1.5.0`)",
            "**Organization:** First Citizens Bank — Incident Intelligence Platform (IIP)  ",
            f"**Execution Timestamp:** `{now_str}`  ",
            f"**Query Source:** `{query_label}`  ",
            f"**Constructed Semantic Query Text:** `{query_text}`  \n",
            "---",
            "\n## Top-K Semantic Similarity Precedents\n",
            "| Rank | Incident Number | Similarity Score | Assignment Group | Priority | Business Service | Resolution Time (Hrs) | Short Description |",
            "|:---:|---|:---:|---|---|---|:---:|---|"
        ]

        for r in results:
            badge = "🔥 Top Precedent" if r.get("rank", 1) <= 2 else f"#{r.get('rank', '-')}"
            lines.append(
                f"| **{badge}** | `{r['incident_number']}` | **`{r['similarity_score']:.4f}`** | "
                f"`{r['assignment_group']}` | `{r['priority']}` | {r['business_service']} | "
                f"`{r['resolution_time']:.2f}` | {r['short_description'][:75]}..." if len(str(r['short_description'])) > 75 else f"| **{badge}** | `{r['incident_number']}` | **`{r['similarity_score']:.4f}`** | `{r['assignment_group']}` | `{r['priority']}` | {r['business_service']} | `{r['resolution_time']:.2f}` | {r['short_description']} |"
            )

        lines.extend([
            "\n---",
            "\n## Semantic Consensus Intelligence\n",
            f"- **Top Consensus Assignment Group:** `{top_group}`",
            f"- **Mean Top-K Similarity Score:** `{avg_sim:.4f}`",
            f"- **Expected Historical Resolution Time:** `{avg_time:.2f} hours`\n",
            "> [!TIP]",
            "> These retrieved semantic precedents directly provide historical context and candidate assignment routes for incoming ServiceNow incidents without requiring manual keyword searches across archived tickets."
        ])

        try:
            with open(md_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
        except PermissionError:
            md_path = self.reports_dir / "similarity_results_latest.md"
            with open(md_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            logger.warning(f"Primary similarity Markdown locked by another process; exported to {md_path}")

        logger.info(f"Exported Top-{len(results)} similarity retrieval reports to {csv_path} and {md_path}")
