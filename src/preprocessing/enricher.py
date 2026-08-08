"""
Module for Data Enrichment from external sources (e.g. CMDB, Shift Schedules).
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple
from src.utils.logger import get_logger

logger = get_logger(__name__)

class EnterpriseDataEnricher:
    """Safely merges external context data into the incident dataframe if files exist."""
    
    def __init__(self, raw_data_dir: str = "data/raw"):
        self.raw_data_dir = Path(raw_data_dir)
        
    def enrich_dataset(self, df: pd.DataFrame, report: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Look for optional external datasets (cmdb.csv, shift_schedules.csv) and merge them.
        """
        enriched_df = df.copy()
        initial_cols = len(enriched_df.columns)
        
        # Check for CMDB Enrichment
        cmdb_path = self.raw_data_dir / "cmdb.csv"
        if cmdb_path.exists():
            try:
                cmdb_df = pd.read_csv(cmdb_path)
                if "category" in cmdb_df.columns:
                    enriched_df = enriched_df.merge(cmdb_df, on="category", how="left")
                    logger.info(f"Enriched dataset with CMDB context from {cmdb_path.name}")
                    report["transformations"].append({
                        "step": "External Data Enrichment",
                        "action": "Merged CMDB Data on 'category'"
                    })
            except Exception as e:
                logger.warning(f"Failed to merge CMDB data: {e}")
                
        # Check for Shift Schedules
        shift_path = self.raw_data_dir / "shift_schedules.csv"
        if shift_path.exists():
            try:
                shift_df = pd.read_csv(shift_path)
                if "assignment_group" in shift_df.columns:
                    enriched_df = enriched_df.merge(shift_df, on="assignment_group", how="left")
                    logger.info(f"Enriched dataset with Shift Schedule context from {shift_path.name}")
                    report["transformations"].append({
                        "step": "External Data Enrichment",
                        "action": "Merged Shift Schedule Data on 'assignment_group'"
                    })
            except Exception as e:
                logger.warning(f"Failed to merge Shift data: {e}")
                
        added_cols = len(enriched_df.columns) - initial_cols
        if added_cols > 0:
            logger.info(f"Data Enrichment complete. Added {added_cols} external features.")
            
        return enriched_df, report
