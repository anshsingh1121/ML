"""
ServiceNow API Fetcher (`src/data/servicenow_api.py`).

Responsible for paginated extraction of incident records from the ServiceNow Table API.
Handles huge datasets by pulling data in chunks and writing directly to disk or a DataFrame.
"""

import os
from pathlib import Path
from typing import Optional, List, Dict, Any
import pandas as pd
import httpx
from dotenv import load_dotenv
from src.utils.logger import get_logger

logger = get_logger(__name__)

class ServiceNowFetcher:
    """
    Client for interacting with the ServiceNow Table API to fetch incidents.
    """

    def __init__(self, env_path: Optional[str] = None):
        """
        Initialize the fetcher and load credentials from .env.
        
        Args:
            env_path: Optional path to the .env file. Defaults to '.env' in the project root.
        """
        if env_path:
            load_dotenv(dotenv_path=env_path)
        else:
            load_dotenv()
            
        self.url = os.getenv("SERVICENOW_URL")
        self.user = os.getenv("SERVICENOW_USER")
        self.password = os.getenv("SERVICENOW_PASS")
        
        # We need these three to make a connection
        self.is_configured = bool(self.url and self.user and self.password)
        
        # Ensure URL doesn't have a trailing slash
        if self.url and self.url.endswith('/'):
            self.url = self.url[:-1]

    def fetch_incidents(
        self, 
        output_path: str = "data/raw/incidents.csv", 
        limit: int = 100000, 
        batch_size: int = 10000
    ) -> bool:
        """
        Fetch incidents from ServiceNow and save to CSV.
        
        Args:
            output_path: Path to save the extracted CSV data.
            limit: Maximum total records to pull (use a very high number for huge datasets).
            batch_size: Number of records to pull per API request.
            
        Returns:
            bool: True if successful and data was written, False otherwise.
        """
        if not self.is_configured:
            logger.error("ServiceNow credentials (SERVICENOW_URL, SERVICENOW_USER, SERVICENOW_PASS) are missing or incomplete in .env.")
            print("\n[ERROR] ServiceNow credentials missing in .env. Please configure them first.")
            return False

        endpoint = f"{self.url}/api/now/table/incident"
        
        # The fields the pipeline requires
        fields = [
            "number", # Mapped to incident_number
            "opened_at",
            "priority",
            "category",
            "assignment_group.name", # Dot-walking to get the group name
            "short_description",
            "description"
        ]
        
        headers = {"Accept": "application/json"}
        auth = (self.user, self.password)
        
        print(f"\n---> [API] Connecting to ServiceNow: {self.url}")
        print(f"---> [API] Target: {limit} incidents in batches of {batch_size}")
        
        all_records = []
        offset = 0
        
        try:
            with httpx.Client(auth=auth, headers=headers, timeout=60.0) as client:
                while offset < limit:
                    current_batch = min(batch_size, limit - offset)
                    params = {
                        "sysparm_limit": current_batch,
                        "sysparm_offset": offset,
                        "sysparm_fields": ",".join(fields),
                        "sysparm_display_value": "true", # To get human-readable values for choices and references
                        "sysparm_exclude_reference_link": "true"
                    }
                    
                    response = client.get(endpoint, params=params)
                    
                    if response.status_code == 401:
                        logger.error("ServiceNow API Authentication Failed. Check SERVICENOW_USER and SERVICENOW_PASS.")
                        print("\n[ERROR] Authentication failed! Please verify credentials in .env")
                        return False
                    
                    response.raise_for_status()
                    data = response.json().get("result", [])
                    
                    if not data:
                        break # No more data available
                        
                    all_records.extend(data)
                    print(f"  -> Fetched {len(data)} records (Total: {len(all_records)})")
                    
                    offset += current_batch
                    
                    if len(data) < current_batch:
                        break # Reached the end of the table
                        
        except httpx.RequestError as e:
            logger.error(f"Network error while connecting to ServiceNow: {e}")
            print(f"\n[ERROR] Network error: {e}")
            return False
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error from ServiceNow: {e.response.status_code} - {e.response.text}")
            print(f"\n[ERROR] HTTP {e.response.status_code}: {e.response.text}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during API extraction: {e}")
            print(f"\n[ERROR] Unexpected error: {e}")
            return False

        if not all_records:
            print("\n[INFO] No records found.")
            return False
            
        # Map fields to match the pipeline's expected schema
        df = pd.DataFrame(all_records)
        
        # Rename 'number' to 'incident_number' and 'assignment_group.name' to 'assignment_group'
        rename_map = {
            "number": "incident_number",
            "assignment_group.name": "assignment_group"
        }
        df = df.rename(columns=rename_map)
        
        # Ensure all required columns are present (fill with empty if completely missing from response)
        required_cols = ["incident_number", "opened_at", "priority", "category", "assignment_group", "short_description", "description"]
        for col in required_cols:
            if col not in df.columns:
                df[col] = ""
                
        # Reorder to standard schema
        df = df[required_cols]
        
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        
        print(f"\n[SUCCESS] Extracted {len(df)} records from ServiceNow.")
        print(f"[SUCCESS] Dataset saved to: {out_path}")
        
        return True
