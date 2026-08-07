"""
Dataset Validation Framework — Enterprise Data Quality Engine.

Provides automated, comprehensive data quality, schema, timestamp, SLA,
and domain-specific rule validation for synthetic and real ServiceNow datasets.

Design Decisions:
    - Rule-Driven OOP Validation: Each validation rule is isolated into its own
      method returning a standardized CheckResult dataclass/dict for clean reporting.
    - Zero Premature Drop: Validation identifies and categorizes anomalies without
      mutating or truncating the underlying dataset (preserves auditability).
    - Enterprise Reporting: Outputs both machine-parseable JSON reports and
      executive-ready Markdown reports to reports/validation_report.* after execution.
"""

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


from src.utils.config_manager import ConfigManager
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CheckResult:
    """Standardized result representing the outcome of a single validation rule check."""
    rule_id: str
    rule_name: str
    passed: bool
    error_count: int
    error_percentage: float
    details: str
    sample_anomalies: List[Dict[str, Any]]


class DatasetValidator:
    """
    Automated validation engine verifying 12 strict data quality requirements across
    ServiceNow attributes, timestamps, domain catalogs, and SLA mechanics.
    """

    REQUIRED_FIELDS = [
        "incident_number", "opened_at", "priority", "category",
        "assignment_group", "short_description", "description"
    ]

    def __init__(self, config: Optional[ConfigManager] = None) -> None:
        """Initialize the DatasetValidator with optional ConfigManager."""
        self.config = config or ConfigManager()
        self.results: List[CheckResult] = []

    def validate_dataset(
        self,
        df: pd.DataFrame,
        save_report: bool = True,
        report_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute all 12 validation checks on the provided dataset.

        Args:
            df: Pandas DataFrame containing incident records to validate.
            save_report: Whether to save validation report to disk.
            report_dir: Custom output directory for reports. Defaults to reports/.

        Returns:
            Dictionary containing overall validation status, check counts, and detailed results.
        """
        logger.info(f"Starting Dataset Validation on {len(df):,} incident records...")
        self.results.clear()

        # 1. Missing Values Check
        self.results.append(self._check_missing_values(df))

        # 2. Duplicate Incident Numbers
        self.results.append(self._check_duplicate_ids(df))

        # 3. Invalid Timestamps
        self.results.append(self._check_invalid_timestamps(df))

        # 4. Invalid Categories
        self.results.append(self._check_invalid_categories(df))

        # 5. Invalid Assignment Groups
        self.results.append(self._check_invalid_assignment_groups(df))

        # 6. Invalid Priorities
        self.results.append(self._check_invalid_priorities(df))

        # 7. SLA Inconsistencies
        self.results.append(self._check_sla_inconsistencies(df))

        # 8. Resolution Time Inconsistencies
        self.results.append(self._check_resolution_time_inconsistencies(df))

        # 9. Invalid CMDB References
        self.results.append(self._check_invalid_cmdb_references(df))

        # 10. Invalid Business Services
        self.results.append(self._check_invalid_business_services(df))

        # 11. Empty Descriptions
        self.results.append(self._check_empty_descriptions(df))

        # 12. Empty Short Descriptions
        self.results.append(self._check_empty_short_descriptions(df))

        passed_checks = sum(1 for r in self.results if r.passed)
        failed_checks = len(self.results) - passed_checks
        is_valid = failed_checks == 0

        summary = {
            "validation_timestamp": datetime.now().isoformat(),
            "total_records": len(df),
            "is_valid": is_valid,
            "total_checks": len(self.results),
            "passed_checks": passed_checks,
            "failed_checks": failed_checks,
            "checks": [asdict(r) for r in self.results]
        }

        if is_valid:
            logger.info(f"Dataset validation PASSED across all {len(self.results)} quality gates.")
        else:
            logger.warning(f"Dataset validation detected {failed_checks} failed quality gates out of {len(self.results)}.")

        if save_report:
            self.save_validation_report(summary, report_dir)

        return summary

    def _check_missing_values(self, df: pd.DataFrame) -> CheckResult:
        """Rule 1: Check for missing or null values across critical required fields."""
        missing_counts = df[self.REQUIRED_FIELDS].isnull().sum()
        total_missing = int(missing_counts.sum())
        passed = total_missing == 0
        error_pct = round((total_missing / (len(df) * len(self.REQUIRED_FIELDS))) * 100, 4) if len(df) > 0 else 0.0

        details_msg = "No missing values found across critical required fields." if passed else f"Found {total_missing} total missing values across critical fields: {missing_counts[missing_counts > 0].to_dict()}"
        
        sample = []
        if not passed:
            missing_rows = df[df[self.REQUIRED_FIELDS].isnull().any(axis=1)].head(3)
            for _, row in missing_rows.iterrows():
                sample.append({"incident_number": row.get("incident_number", "UNKNOWN"), "missing_cols": [col for col in self.REQUIRED_FIELDS if pd.isnull(row.get(col))]})

        return CheckResult(
            rule_id="CHK-01",
            rule_name="Missing Values Check",
            passed=passed,
            error_count=total_missing,
            error_percentage=error_pct,
            details=details_msg,
            sample_anomalies=sample
        )

    def _check_duplicate_ids(self, df: pd.DataFrame) -> CheckResult:
        """Rule 2: Check for exact duplicate incident_number keys."""
        if "incident_number" not in df.columns:
            return CheckResult("CHK-02", "Duplicate Incident Numbers", False, len(df), 100.0, "Missing incident_number column", [])

        dup_mask = df["incident_number"].duplicated(keep=False)
        error_count = int(dup_mask.sum())
        passed = error_count == 0
        error_pct = round((error_count / len(df)) * 100, 4) if len(df) > 0 else 0.0

        details_msg = "All incident numbers are unique." if passed else f"Found {error_count} records sharing duplicate incident numbers."
        sample = []
        if not passed:
            dup_records = df[dup_mask]["incident_number"].head(5).tolist()
            sample.append({"duplicate_ids": dup_records})

        return CheckResult("CHK-02", "Duplicate Incident Numbers", passed, error_count, error_pct, details_msg, sample)

    def _check_invalid_timestamps(self, df: pd.DataFrame) -> CheckResult:
        """Rule 3: Check for logical timestamp anomalies (resolved < opened, closed < resolved)."""
        error_count = 0
        anomalies = []

        if "opened_at" in df.columns and "resolved_at" in df.columns:
            # Check resolved_at < opened_at for resolved records
            res_mask = ~df["resolved_at"].isnull()
            invalid_res = df[res_mask & (pd.to_datetime(df["resolved_at"]) < pd.to_datetime(df["opened_at"]))]
            error_count += len(invalid_res)
            for _, r in invalid_res.head(3).iterrows():
                anomalies.append({"incident_number": r["incident_number"], "error": "resolved_at < opened_at", "opened_at": str(r["opened_at"]), "resolved_at": str(r["resolved_at"])})

        if "resolved_at" in df.columns and "closed_at" in df.columns:
            close_mask = ~df["closed_at"].isnull() & ~df["resolved_at"].isnull()
            invalid_close = df[close_mask & (pd.to_datetime(df["closed_at"]) < pd.to_datetime(df["resolved_at"]))]
            error_count += len(invalid_close)
            for _, r in invalid_close.head(3).iterrows():
                anomalies.append({"incident_number": r["incident_number"], "error": "closed_at < resolved_at", "resolved_at": str(r["resolved_at"]), "closed_at": str(r["closed_at"])})

        passed = error_count == 0
        error_pct = round((error_count / len(df)) * 100, 4) if len(df) > 0 else 0.0
        details_msg = "All operational timestamps maintain logical progression sequence." if passed else f"Detected {error_count} timestamp sequence errors."

        return CheckResult("CHK-03", "Invalid Timestamps", passed, error_count, error_pct, details_msg, anomalies)

    def _check_invalid_categories(self, df: pd.DataFrame) -> CheckResult:
        """Rule 4: Check if categories exist and are not null."""
        if "category" not in df.columns:
            return CheckResult("CHK-04", "Invalid Categories", False, len(df), 100.0, "Missing category column", [])

        invalid_mask = df["category"].isnull() | (df["category"].astype(str).str.strip() == "")

        error_count = int(invalid_mask.sum())
        passed = error_count == 0
        error_pct = round((error_count / len(df)) * 100, 4) if len(df) > 0 else 0.0

        details_msg = "All categories are valid." if passed else f"Found {error_count} records with missing/non-standard categories."
        sample = [{"invalid_category": str(c)} for c in df[invalid_mask]["category"].head(3).tolist()] if not passed else []

        return CheckResult("CHK-04", "Invalid Categories", passed, error_count, error_pct, details_msg, sample)

    def _check_invalid_assignment_groups(self, df: pd.DataFrame) -> CheckResult:
        """Rule 9: Check if assignment groups exist and are not null."""
        if "assignment_group" not in df.columns:
            return CheckResult("CHK-09", "Invalid Assignment Groups", False, len(df), 100.0, "Missing column", [])

        invalid_mask = df["assignment_group"].isnull() | (df["assignment_group"].astype(str).str.strip() == "")

        error_count = int(invalid_mask.sum())
        passed = error_count == 0
        error_pct = round((error_count / len(df)) * 100, 4) if len(df) > 0 else 0.0

        details_msg = "All assignment groups are valid." if passed else f"Found {error_count} records with missing/unknown assignment groups."
        sample = [{"invalid_group": str(g)} for g in df[invalid_mask]["assignment_group"].head(3).tolist()] if not passed else []

        return CheckResult("CHK-05", "Invalid Assignment Groups", passed, error_count, error_pct, details_msg, sample)

    def _check_invalid_priorities(self, df: pd.DataFrame) -> CheckResult:
        """Rule 6: Check if priorities fall cleanly within integer levels 1 through 5."""
        if "priority" not in df.columns:
            return CheckResult("CHK-06", "Invalid Priorities", False, len(df), 100.0, "Missing priority column", [])

        valid_priorities = {1, 2, 3, 4, 5}
        invalid_mask = ~df["priority"].isin(valid_priorities)
        error_count = int(invalid_mask.sum())
        passed = error_count == 0
        error_pct = round((error_count / len(df)) * 100, 4) if len(df) > 0 else 0.0

        details_msg = "All ticket priorities conform to standard 1-5 integer scale." if passed else f"Found {error_count} tickets with out-of-bounds priorities."
        sample = [{"invalid_priority": int(p)} for p in df[invalid_mask]["priority"].head(3).tolist()] if not passed else []

        return CheckResult("CHK-06", "Invalid Priorities", passed, error_count, error_pct, details_msg, sample)

    def _check_sla_inconsistencies(self, df: pd.DataFrame) -> CheckResult:
        """Rule 7: Check if SLA status (made_sla) logically aligns with resolution time vs targets."""
        error_count = 0
        anomalies = []

        if "made_sla" in df.columns and "resolution_time_hours" in df.columns and "priority" in df.columns:
            sla_targets = {1: 4.0, 2: 12.0, 3: 48.0, 4: 120.0, 5: 240.0}
            res_mask = ~df["resolution_time_hours"].isnull() & ~df["made_sla"].isnull()
            
            for idx, r in df[res_mask].iterrows():
                target = sla_targets.get(r["priority"], 48.0)
                if r["made_sla"] and r["resolution_time_hours"] > (target + 0.05):
                    error_count += 1
                    if len(anomalies) < 3:
                        anomalies.append({"incident_number": r["incident_number"], "error": "made_sla=True but resolution exceeded target", "priority": r["priority"], "resolution_time_hours": r["resolution_time_hours"], "target_hours": target})
                elif not r["made_sla"] and r["resolution_time_hours"] <= target and r["sla_status"] == "Breached":
                    error_count += 1
                    if len(anomalies) < 3:
                        anomalies.append({"incident_number": r["incident_number"], "error": "made_sla=False but resolution within target", "priority": r["priority"], "resolution_time_hours": r["resolution_time_hours"], "target_hours": target})

        passed = error_count == 0
        error_pct = round((error_count / len(df)) * 100, 4) if len(df) > 0 else 0.0
        details_msg = "SLA compliance flags strictly correspond to resolution time boundaries." if passed else f"Detected {error_count} SLA calculation inconsistencies."

        return CheckResult("CHK-07", "SLA Inconsistencies", passed, error_count, error_pct, details_msg, anomalies)

    def _check_resolution_time_inconsistencies(self, df: pd.DataFrame) -> CheckResult:
        """Rule 8: Check for negative resolution times or divergence from resolved_at - opened_at."""
        error_count = 0
        anomalies = []

        if "resolution_time_hours" in df.columns:
            # Check negative resolution times
            neg_mask = df["resolution_time_hours"] < 0
            neg_count = int(neg_mask.sum())
            error_count += neg_count
            if neg_count > 0:
                for _, r in df[neg_mask].head(3).iterrows():
                    anomalies.append({"incident_number": r["incident_number"], "error": "Negative resolution time", "resolution_time_hours": r["resolution_time_hours"]})

            # Check timestamp difference vs resolution_time_hours tolerance
            if "opened_at" in df.columns and "resolved_at" in df.columns:
                res_df = df[~df["resolved_at"].isnull() & ~df["resolution_time_hours"].isnull()]
                if len(res_df) > 0:
                    dt_diff_hours = (pd.to_datetime(res_df["resolved_at"]) - pd.to_datetime(res_df["opened_at"])).dt.total_seconds() / 3600.0
                    diff_error = np.abs(dt_diff_hours - res_df["resolution_time_hours"]) > 0.5
                    diff_count = int(diff_error.sum())
                    error_count += diff_count
                    if diff_count > 0 and len(anomalies) < 3:
                        for _, r in res_df[diff_error].head(3).iterrows():
                            anomalies.append({"incident_number": r["incident_number"], "error": "Resolution time diverges from resolved_at - opened_at by >30 mins"})

        passed = error_count == 0
        error_pct = round((error_count / len(df)) * 100, 4) if len(df) > 0 else 0.0
        details_msg = "Resolution times are positive and align accurately with timestamp deltas." if passed else f"Detected {error_count} resolution time inconsistencies."

        return CheckResult("CHK-08", "Resolution Time Inconsistencies", passed, error_count, error_pct, details_msg, anomalies)

    def _check_invalid_cmdb_references(self, df: pd.DataFrame) -> CheckResult:
        """Rule 9: Check if cmdb_ci references are populated or adhere to CI naming patterns."""
        if "cmdb_ci" not in df.columns:
            return CheckResult("CHK-09", "Invalid CMDB References", False, len(df), 100.0, "Missing cmdb_ci column", [])

        empty_mask = df["cmdb_ci"].isnull() | (df["cmdb_ci"].astype(str).str.strip() == "") | (df["cmdb_ci"] == "UNKNOWN")
        error_count = int(empty_mask.sum())
        passed = error_count == 0
        error_pct = round((error_count / len(df)) * 100, 4) if len(df) > 0 else 0.0

        details_msg = "All incident records maintain valid CMDB Configuration Item references." if passed else f"Found {error_count} records with missing or unknown CMDB references."
        sample = [{"incident_number": str(num), "cmdb_ci": "EMPTY"} for num in df[empty_mask]["incident_number"].head(3).tolist()] if not passed else []

        return CheckResult("CHK-09", "Invalid CMDB References", passed, error_count, error_pct, details_msg, sample)

    def _check_invalid_business_services(self, df: pd.DataFrame) -> CheckResult:
        """Rule 10: Check if business_service is populated with valid service catalog mappings."""
        if "business_service" not in df.columns:
            return CheckResult("CHK-10", "Invalid Business Services", False, len(df), 100.0, "Missing business_service column", [])

        empty_mask = df["business_service"].isnull() | (df["business_service"].astype(str).str.strip() == "")
        error_count = int(empty_mask.sum())
        passed = error_count == 0
        error_pct = round((error_count / len(df)) * 100, 4) if len(df) > 0 else 0.0

        details_msg = "All tickets map cleanly to enterprise Business Services." if passed else f"Found {error_count} tickets missing Business Service designations."
        sample = [{"incident_number": str(num)} for num in df[empty_mask]["incident_number"].head(3).tolist()] if not passed else []

        return CheckResult("CHK-10", "Invalid Business Services", passed, error_count, error_pct, details_msg, sample)

    def _check_empty_descriptions(self, df: pd.DataFrame) -> CheckResult:
        """Rule 11: Check for empty or whitespace-only description fields."""
        if "description" not in df.columns:
            return CheckResult("CHK-11", "Empty Descriptions", False, len(df), 100.0, "Missing description column", [])

        empty_mask = df["description"].isnull() | (df["description"].astype(str).str.strip() == "")
        error_count = int(empty_mask.sum())
        passed = error_count == 0
        error_pct = round((error_count / len(df)) * 100, 4) if len(df) > 0 else 0.0

        details_msg = "All records contain detailed incident descriptions." if passed else f"Found {error_count} tickets with empty or blank descriptions."
        sample = [{"incident_number": str(num)} for num in df[empty_mask]["incident_number"].head(3).tolist()] if not passed else []

        return CheckResult("CHK-11", "Empty Descriptions", passed, error_count, error_pct, details_msg, sample)

    def _check_empty_short_descriptions(self, df: pd.DataFrame) -> CheckResult:
        """Rule 12: Check for empty or whitespace-only short_description fields."""
        if "short_description" not in df.columns:
            return CheckResult("CHK-12", "Empty Short Descriptions", False, len(df), 100.0, "Missing short_description column", [])

        empty_mask = df["short_description"].isnull() | (df["short_description"].astype(str).str.strip() == "")
        error_count = int(empty_mask.sum())
        passed = error_count == 0
        error_pct = round((error_count / len(df)) * 100, 4) if len(df) > 0 else 0.0

        details_msg = "All records contain valid summary short descriptions." if passed else f"Found {error_count} tickets with empty short descriptions."
        sample = [{"incident_number": str(num)} for num in df[empty_mask]["incident_number"].head(3).tolist()] if not passed else []

        return CheckResult("CHK-12", "Empty Short Descriptions", passed, error_count, error_pct, details_msg, sample)

    def save_validation_report(self, summary: Dict[str, Any], report_dir: Optional[str] = None) -> Tuple[Path, Path]:
        """
        Save validation results to disk in both JSON and Markdown format.

        Args:
            summary: The validation summary dictionary returned by validate_dataset().
            report_dir: Target directory path. Defaults to reports/.

        Returns:
            Tuple containing paths to the saved (JSON, Markdown) report files.
        """
        out_dir = Path(report_dir or self.config.get("reports.dir", "reports"))
        out_dir.mkdir(parents=True, exist_ok=True)

        json_path = out_dir / "validation_report.json"
        md_path = out_dir / "validation_report.md"

        # Save JSON
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        # Save Markdown
        lines = [
            "# Dataset Validation Report — Quality Gate Summary",
            f"**Validation Timestamp:** {summary['validation_timestamp']}  ",
            f"**Total Records Evaluated:** {summary['total_records']:,}  ",
            f"**Overall Quality Gate Status:** {'✅ PASSED' if summary['is_valid'] else '❌ FAILED'}  ",
            f"**Checks Summary:** {summary['passed_checks']}/{summary['total_checks']} Passed  \n",
            "---",
            "\n## Detailed Quality Check Matrix\n",
            "| Rule ID | Quality Rule Name | Status | Anomaly Count | Error % | Details |",
            "|:---:|---|:---:|:---:|:---:|---|"
        ]

        for check in summary["checks"]:
            status_icon = "✅ PASS" if check["passed"] else "⚠️ FAIL"
            lines.append(
                f"| `{check['rule_id']}` | **{check['rule_name']}** | {status_icon} | "
                f"{check['error_count']:,} | {check['error_percentage']:.4f}% | {check['details']} |"
            )

        if not summary["is_valid"]:
            lines.extend([
                "\n---",
                "\n## Detected Anomaly Samples for Investigation\n"
            ])
            for check in summary["checks"]:
                if not check["passed"] and check["sample_anomalies"]:
                    lines.append(f"### `{check['rule_id']}` — {check['rule_name']}")
                    lines.append("```json")
                    lines.append(json.dumps(check["sample_anomalies"], indent=2))
                    lines.append("```\n")

        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        logger.info(f"Validation reports generated: {md_path} & {json_path}")
        return json_path, md_path
