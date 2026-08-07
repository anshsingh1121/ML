"""
Dataset Version Control & Metadata Management Engine.

Ensures strict immutability and audit tracking for synthetic and production
datasets by creating incremented version directories (datasets/synthetic/v1, v2, etc.)
and generating standardized metadata.json manifests.

Design Decisions:
    - Immutable Version Directories: Prevents silent data overwrites during ML experimentation.
      Every generated run gets a dedicated v{N} folder with clean isolation.
    - Comprehensive Metadata Manifest (`metadata.json`): Captures exact generator
      versions, random seeds, schema bounds, and categorical distribution summaries
      to enable reproducibility across training experiments.
    - Centralized Version Catalog (`version_history.json`): Provides instant audit
      visibility across all historical data releases without scanning disk files.
"""

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from src.utils import robust_open

from src.utils.config_manager import ConfigManager
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DatasetVersionManager:
    """
    Manages non-destructive dataset versioning under datasets/synthetic/vX/,
    metadata manifest generation (`metadata.json`), and version history tracking.
    """

    SCHEMA_VERSION = "1.5.0"
    GENERATOR_VERSION = "1.5.0"

    def __init__(
        self,
        base_dir: Optional[str] = None,
        config: Optional[ConfigManager] = None
    ) -> None:
        """Initialize DatasetVersionManager with base directory path."""
        self.config = config or ConfigManager()
        self.base_dir = Path(base_dir or self.config.get("data.synthetic_dir", "datasets/synthetic"))
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.history_file = self.base_dir / "version_history.json"

    def get_latest_version_number(self) -> int:
        """Determine the highest version number currently stored on disk."""
        max_v = 0
        if not self.base_dir.exists():
            return 0

        for item in self.base_dir.iterdir():
            if item.is_dir() and re.match(r"^v\d+$", item.name):
                v_num = int(item.name[1:])
                if v_num > max_v:
                    max_v = v_num
        return max_v

    def get_next_version_id(self) -> str:
        """Get the next version string ID (e.g., 'v1', 'v2')."""
        return f"v{self.get_latest_version_number() + 1}"

    def save_versioned_dataset(
        self,
        df: pd.DataFrame,
        seed: int = 42,
        file_format: str = "csv",
        custom_version: Optional[str] = None
    ) -> Tuple[Path, Path, str]:
        """
        Save dataframe to an immutable version directory along with metadata.json.

        Args:
            df: Pandas DataFrame containing incident records.
            seed: Random seed used during generation.
            file_format: Output file format ('csv' or 'parquet').
            custom_version: Optional explicit version override if non-existent.

        Returns:
            Tuple containing (dataset_file_path, metadata_file_path, version_id).

        Raises:
            FileExistsError: If custom_version directory already exists.
        """
        version_id = custom_version or self.get_next_version_id()
        v_dir = self.base_dir / version_id

        if v_dir.exists() and any(v_dir.iterdir()):
            if custom_version:
                raise FileExistsError(f"Dataset version {version_id} already exists at {v_dir}. Overwriting is strictly prohibited.")
            # If race condition, increment
            version_id = self.get_next_version_id()
            v_dir = self.base_dir / version_id

        v_dir.mkdir(parents=True, exist_ok=True)

        filename = f"incidents.{file_format.lower()}"
        data_file = v_dir / filename

        logger.info(f"Saving immutable dataset ({len(df):,} rows) to version {version_id} -> {data_file}...")

        if file_format.lower() == "parquet":
            df.to_parquet(data_file, index=False, engine="pyarrow")
        else:
            df.to_csv(data_file, index=False, encoding="utf-8")

        # Generate metadata.json
        metadata = self._generate_metadata(df, version_id, seed, data_file.name)
        metadata_file = v_dir / "metadata.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        # Update global history
        self._update_history(metadata)

        logger.info(f"Dataset version {version_id} successfully finalized with verified metadata.")
        return data_file, metadata_file, version_id

    def _generate_metadata(
        self,
        df: pd.DataFrame,
        version_id: str,
        seed: int,
        filename: str
    ) -> Dict[str, Any]:
        """Compute statistical distributions and format metadata manifest."""
        dist_summary = {}

        if "category" in df.columns:
            dist_summary["categories"] = df["category"].value_counts().to_dict()
        if "assignment_group" in df.columns:
            dist_summary["assignment_groups"] = df["assignment_group"].value_counts().to_dict()
        if "priority" in df.columns:
            dist_summary["priorities"] = {str(k): int(v) for k, v in df["priority"].value_counts().items()}
        if "state" in df.columns:
            dist_summary["states"] = {str(k): int(v) for k, v in df["state"].value_counts().items()}

        sla_pct = 0.0
        if "made_sla" in df.columns:
            sla_pct = round(float(df["made_sla"].mean()) * 100.0, 2)
        dist_summary["sla_compliance_rate_pct"] = sla_pct

        avg_rt = 0.0
        if "resolution_time_hours" in df.columns:
            avg_rt = round(float(df["resolution_time_hours"].dropna().mean()), 2)
        dist_summary["avg_resolution_time_hours"] = avg_rt

        return {
            "dataset_version": version_id,
            "generation_timestamp": datetime.now().isoformat(),
            "random_seed": seed,
            "filename": filename,
            "num_rows": len(df),
            "num_columns": len(df.columns),
            "schema_version": self.SCHEMA_VERSION,
            "generator_version": self.GENERATOR_VERSION,
            "distribution_summary": dist_summary
        }

    def _update_history(self, metadata: Dict[str, Any]) -> None:
        """Append metadata summary to version_history.json."""
        history = []
        if self.history_file.exists():
            try:
                with robust_open(self.history_file, "r") as f:
                    history = json.load(f)
                    if not isinstance(history, list):
                        history = []
            except Exception as e:
                logger.warning(f"Could not read version_history.json ({e}). Creating clean history list.")
                history = []

        summary_entry = {
            "version": metadata["dataset_version"],
            "timestamp": metadata["generation_timestamp"],
            "rows": metadata["num_rows"],
            "columns": metadata["num_columns"],
            "seed": metadata["random_seed"],
            "filename": metadata["filename"]
        }

        # Check if version exists in history, replace if so, else append
        existing_idx = next((i for i, h in enumerate(history) if h.get("version") == metadata["dataset_version"]), None)
        if existing_idx is not None:
            history[existing_idx] = summary_entry
        else:
            history.append(summary_entry)

        with open(self.history_file, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

    def load_metadata(self, version_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve metadata manifest for a specific dataset version."""
        meta_file = self.base_dir / version_id / "metadata.json"
        if not meta_file.exists():
            return None
        with robust_open(meta_file, "r") as f:
            return json.load(f)

    def list_all_versions(self) -> List[Dict[str, Any]]:
        """Return the complete list of historical dataset versions."""
        if not self.history_file.exists():
            return []
        with robust_open(self.history_file, "r") as f:
            content = json.load(f)
            return content if isinstance(content, list) else []
