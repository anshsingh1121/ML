# Section-Wise Code Explanation
## First 3 Invoked Scripts — Line-by-Line Breakdown

---

# SCRIPT 1: `main.py` (154 lines)
## Role: Entry Point — The Front Door

---

### Section 1: Module Docstring (Lines 1-8)
```python
"""
Enterprise Incident Intelligence Platform (IIP) — First Citizens Bank (v2.0.0-alpha).
Root executable entry point (main.py).
"""
```
**Purpose:** Describes what this file is. Python ignores this at runtime — it's purely for developers reading the code.

---

### Section 2: Imports (Lines 10-18)
```python
import argparse                          # Built-in library for parsing CLI arguments
import sys                               # Built-in library for system-level operations
from pathlib import Path                 # Built-in library for cross-platform file paths

sys.path.insert(0, str(Path(__file__).resolve().parent))   # LINE 15

from src.cli.main_cli import EnterpriseCLI    # The central controller class
from src.utils.logger import get_logger       # Logging factory
```

**Line 15 is critical:** `Path(__file__).resolve().parent` gets the directory where `main.py` lives (the project root). `sys.path.insert(0, ...)` adds it to Python's module search path so that `from src.cli.main_cli import ...` works regardless of where you run the command from.

**Without this line:** Running `python main.py` from a different directory would throw `ModuleNotFoundError: No module named 'src'`.

---

### Section 3: Logger Initialization (Line 20)
```python
logger = get_logger(__name__)
```
**What it does:** Creates a logger named `"__main__"`. This logger writes to `logs/app.log` (rotating, max 10MB) and the console. Every module in the project starts with this exact same line.

---

### Section 4: `main()` Function — The Argument Parser (Lines 23-148)

#### 4a: Parser Setup (Lines 25-44)
```python
def main() -> int:
    parser = argparse.ArgumentParser(
        description="First Citizens Bank — AI-Powered Incident Intelligence Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                                  # Launch interactive menu
  python main.py train --target assignment_group   # Train classifier
  ...
"""
    )
    subparsers = parser.add_subparsers(dest="command", help="Operational Subcommands")
```
**What it does:** Creates the argument parser. `RawDescriptionHelpFormatter` preserves the formatting of the help text. `subparsers` lets each command (`validate`, `train`, `embed`, etc.) have its own set of arguments.

#### 4b: Subcommand Definitions (Lines 46-143)
Each block follows the same pattern:
```python
# Example: the "validate" command
val_parser = subparsers.add_parser("validate", help="Run dataset validation checks")
val_parser.add_argument("--input", type=str, default="data/raw/incidents.csv", help="Input CSV path")
```
**What it does:** Registers the `validate` subcommand. When user types `python main.py validate --input mydata.csv`, argparse stores `args.command = "validate"` and `args.input = "mydata.csv"`.

**All 19 subcommands registered here:**
| Command | Default Input | What It Triggers |
|---|---|---|
| `menu` | — | Interactive terminal menu |
| `status` | — | Health check |
| `validate` | `data/raw/incidents.csv` | 12-rule quality check |
| `readiness` | `data/raw/incidents.csv` | ML feasibility audit |
| `eda` | `data/raw/incidents.csv` | EDA charts + stats |
| `clean` | `data/raw/incidents.csv` | 8-step data cleaning |
| `engineer` | `data/processed/cleaned_incidents.csv` | Feature generation |
| `split` | `data/processed/engineered_incidents.csv` | Train/val/test split |
| `pipeline` | `data/raw/incidents.csv` | 5-stage data pipeline |
| `train` | `data/processed/train.csv` | CatBoost training + HPO |
| `evaluate` | `data/processed/test.csv` | Model metrics + charts |
| `explain` | `data/processed/test.csv` | SHAP attribution |
| `models` | — | Model registry audit |
| `predict` | (required) | Inference + SHAP |
| `embed` | `data/processed/train.csv` | TF-IDF+SVD vectors |
| `index` | `data/processed/train.csv` | FAISS index build |
| `similar` | — | Top-K precedent search |
| `recommend` | — | Hybrid recommendation |
| `full-pipeline` | `data/raw/incidents.csv` | All 12 stages |
| `clean-workspace` | — | Delete generated files |

#### 4c: Dispatch (Lines 145-148)
```python
args = parser.parse_args()       # Parse whatever the user typed
cli = EnterpriseCLI()            # Create the central controller
return cli.run_command(args)     # Hand off to main_cli.py
```
**This is where control leaves main.py and enters main_cli.py.**

---

### Section 5: Script Execution Block (Lines 151-152)
```python
if __name__ == "__main__":
    sys.exit(main())
```
**What it does:** Only runs when you execute `python main.py` directly. `sys.exit(main())` ensures the process exit code matches the return value (0 = success, 1 = error). This matters for CI/CD pipelines and shell scripts.

---
---

# SCRIPT 2: `src/cli/main_cli.py` (849 lines)
## Role: Central Controller — The Brain

---

### Section 1: Imports (Lines 1-39)

#### 1a: Standard Library Imports (Lines 9-17)
```python
import argparse          # For type hint on args.Namespace
import json              # For reading JSON payloads
from pathlib import Path # For file path operations
import shutil            # For deleting directories (clean-workspace)
import subprocess        # Not used directly here (used in run_dashboard.py)
import sys               # For sys.stdin.isatty() check in interactive menu
import time              # For timing the full pipeline execution
from typing import ...   # Type hints
import pandas as pd      # DataFrame operations
```

#### 1b: Project Internal Imports (Lines 18-38)
```python
from src.utils import robust_read_csv, robust_open    # Multi-encoding file readers
from src.data.validation import DatasetValidator       # 12-rule quality engine
from src.data.readiness import MLReadinessEvaluator as MLReadinessChecker  # ML diagnostic
from src.preprocessing.eda import EnterpriseEDAEngine  # Automated EDA
from src.preprocessing.cleaner import EnterpriseDataCleaner     # 8-step cleaner
from src.preprocessing.enricher import EnterpriseDataEnricher   # CMDB/shift merge
from src.preprocessing.engineer import FeatureEngineeringEngine  # Feature creator
from src.preprocessing.text_preprocessor import TextPreprocessor # NLP normalizer
from src.preprocessing.splitter import DatasetSplitter           # Train/val/test split
from src.data.feature_registry import FeatureRegistry            # Feature governance
from src.ml.catboost.trainer import EnterpriseCatBoostTrainer    # Model training
from src.ml.catboost.evaluator import ModelEvaluator             # Model evaluation
from src.ml.explainability.shap_explainer import SHAPIntelligenceExplainer  # SHAP
from src.ml.model_registry import ModelRegistry         # Model artifact tracking
from src.ml.semantic.embedding_generator import SemanticEmbeddingGenerator  # Vectors
from src.ml.semantic.faiss_index import FAISSVectorIndex  # FAISS index
from src.ml.semantic.similarity_engine import SemanticSimilarityEngine  # Search
from src.utils.logger import get_logger
```
**This is why every file in your list is mandatory** — they're all imported here at the module level. If any file is missing, Python crashes with `ImportError` before a single line of `EnterpriseCLI` code runs.

---

### Section 2: `__init__()` — Boot Sequence (Lines 48-60)
```python
def __init__(self) -> None:
    # Create all runtime directories
    for required_dir in [
        "data/raw", "data/processed",
        "models", "models/embeddings", "indexes",
        "reports", "reports/figures", "reports/daily", "reports/weekly", "reports/monthly",
        "logs"
    ]:
        Path(required_dir).mkdir(parents=True, exist_ok=True)

    self.registry = FeatureRegistry.get_instance()     # Load 49-feature governance catalog
    self.model_reg = ModelRegistry.get_instance()       # Load model artifact tracker
    from src.utils.config_manager import ConfigManager
    self.config = ConfigManager()                       # Load all YAML configs + .env
```
**What happens:**
1. **Creates 10 directories** — `mkdir(parents=True, exist_ok=True)` creates them if missing, does nothing if they already exist.
2. **Loads FeatureRegistry singleton** — Registers 38 raw + 11 engineered features with 22 governance dimensions each.
3. **Loads ModelRegistry singleton** — Reads `models/model_registry.json` to track trained models.
4. **Loads ConfigManager** — Reads `config/config.yaml`, `model_config.yaml`, `servicenow.yaml`, and `.env`.

**Note:** `ConfigManager` is imported inside the method (line 59), not at the top of the file. This is a deliberate pattern to avoid circular import issues.

---

### Section 3: `_check_and_self_heal()` (Lines 62-75)
```python
def _check_and_self_heal(self) -> None:
    raw_path = Path("data/raw/incidents.csv")
    clf_path = Path("models/catboost_assignment_group.pkl")
    idx_path = Path("indexes/incident_semantic_index_latest.index")
    proc_path = Path("data/processed/master_engineered_incidents.csv")

    if not clf_path.exists() or not idx_path.exists() or not proc_path.exists() or not raw_path.exists():
        self.cmd_full_pipeline(input_path=str(raw_path))
```
**What it does:** Before running inference commands (`recommend`, `predict`, `evaluate`, `explain`), it checks if the 4 core artifacts exist. If ANY is missing, it automatically runs the entire 12-stage pipeline to regenerate them. This is the "self-healing" mechanism.

---

### Section 4: `run_command()` — The Dispatcher (Lines 77-132)
```python
def run_command(self, args: argparse.Namespace) -> int:
    command = getattr(args, "command", "menu")
    if command is None or command == "menu":
        return self.run_interactive_menu()

    # Self-heal before inference
    if command in ["recommend", "predict", "evaluate", "explain"]:
        self._check_and_self_heal()

    try:
        if command == "validate":
            return self.cmd_validate(args.input)
        elif command == "train":
            return self.cmd_train(args.target, args.compare_baselines, ...)
        elif command == "full-pipeline":
            return self.cmd_full_pipeline(input_path=...)
        # ... 16 more elif branches ...
    except Exception as e:
        logger.error(f"Command execution failed ({command}): {e}", exc_info=True)
        return 1
```
**What it does:** A giant if/elif chain that maps CLI command strings to their respective methods. Every method returns `0` for success, `1` for error. The entire block is wrapped in try/except so any unhandled crash is logged and returns exit code 1.

---

### Section 5: Individual Stage Methods (Lines 137-484)
Each `cmd_*` method follows the same pattern:
```python
def cmd_validate(self, input_path: str) -> int:
    print(f"\n---> [1/1] Running Enterprise Dataset Validation on: {input_path}...")
    df = robust_read_csv(input_path)          # Read CSV with encoding fallback
    validator = DatasetValidator()              # Create the engine
    report = validator.validate_dataset(df)     # Run the logic
    print(f"[STATUS] Validation Result: ...")   # Print result
    return 0                                    # Always proceed (even if validation fails)
```
**Key design:** `cmd_validate` returns `0` even if validation finds anomalies. This is intentional — validation *identifies* problems but doesn't block the pipeline. The cleaner (Stage 3) will *fix* them.

---

### Section 6: `cmd_full_pipeline()` — The Master Orchestrator (Lines 498-573)
```python
def cmd_full_pipeline(self, input_path: str = "data/raw/incidents.csv") -> int:
    start_time = time.time()

    stages = [
        ("Stage 1: Check Input Dataset",      lambda: self._run_stage1_dataset_check(input_path)),
        ("Stage 2: Dataset Validation",        lambda: self.cmd_validate(input_path)),
        ("Stage 3: Data Intelligence Pipeline",lambda: self.cmd_pipeline(input_path, "data/processed")),
        ("Stage 4: Train Classifier",          lambda: self.cmd_train(target="assignment_group", ...)),
        ("Stage 5: Train Regressor",           lambda: self.cmd_train(target="resolution_time_hours", ...)),
        ("Stage 6: Evaluate Classification",   lambda: self.cmd_evaluate(...)),
        ("Stage 7: SHAP Explainability",       lambda: self.cmd_explain(...)),
        ("Stage 8: Generate Embeddings",       lambda: self.cmd_embed(...)),
        ("Stage 9: Build FAISS Index",         lambda: self.cmd_index(...)),
        ("Stage 10: Hybrid Recommendation",    lambda: self.cmd_recommend(...)),
    ]

    for idx, (stage_name, stage_fn) in enumerate(stages, 1):
        try:
            ret = stage_fn()                   # Execute the stage
            if ret != 0:
                overall_status = ret
                break                          # HALT on failure
        except Exception as e:
            overall_status = 1
            break                              # HALT on crash

    total_elapsed = time.time() - start_time
    # Print summary table of all stage results
```
**Key design:**
- Uses a **list of lambdas** — each stage is a `(name, function)` pair. This allows iterating through stages with a clean for-loop.
- **Halts on first failure** — if any stage returns non-zero or throws an exception, the pipeline stops immediately.
- **Note:** Stage 3 (`cmd_pipeline`) internally runs 5 sub-stages (Clean → Enrich → Engineer → TextPreprocess → EDA → Split), so the actual execution is 12 stages total.

---

### Section 7: `cmd_clean_workspace()` (Lines 575-724)
Deletes ALL generated artifacts:
- `reports/*.json, *.md, *.png, *.html`
- `models/*.pkl, *.npy`
- `indexes/*.index`
- `logs/*.log`
- `__pycache__/`, `.pytest_cache/`, `.coverage`
- `data/processed/*.csv`

Preserves directory structure and source code. Uses `shutil.rmtree()` for directories and `Path.unlink()` for files.

---

### Section 8: `run_interactive_menu()` (Lines 748-849)
Prints a 24-option ASCII menu and reads user input via `input()`. Maps choices 1-24 to `cmd_*` methods. If running in non-interactive mode (piped input), auto-executes `cmd_status()`.

---
---

# SCRIPT 3: `src/data/validation.py` (452 lines)
## Role: Stage 1 — Data Quality Police

---

### Section 1: Module Docstring (Lines 1-14)
```python
"""
Dataset Validation Framework — Enterprise Data Quality Engine.

Design Decisions:
    - Rule-Driven OOP Validation: Each rule is its own method returning CheckResult
    - Zero Premature Drop: Identifies anomalies WITHOUT mutating data
    - Enterprise Reporting: Outputs JSON + Markdown reports
"""
```
**3 key design decisions documented:**
1. Each of the 12 rules is a separate method → testable in isolation
2. Validation **never deletes or modifies data** → that's the cleaner's job
3. Outputs machine-readable JSON AND human-readable Markdown

---

### Section 2: Imports (Lines 16-29)
```python
import json                          # For JSON report output
from dataclasses import dataclass, asdict  # For CheckResult data structure
from datetime import datetime        # For audit timestamps
from pathlib import Path             # For file paths
from typing import ...               # Type annotations

import numpy as np                   # For abs() difference calculation (CHK-08)
import pandas as pd                  # For DataFrame operations

from src.utils.config_manager import ConfigManager   # Config access
from src.utils.logger import get_logger              # Logging
```

---

### Section 3: `CheckResult` Dataclass (Lines 32-41)
```python
@dataclass
class CheckResult:
    rule_id: str                        # e.g. "CHK-01"
    rule_name: str                      # e.g. "Missing Values Check"
    passed: bool                        # True if zero anomalies found
    error_count: int                    # Number of problematic records
    error_percentage: float             # error_count / total * 100
    details: str                        # Human-readable explanation
    sample_anomalies: List[Dict]        # Up to 3 example bad records
```
**Why dataclass?** Gives you `__init__`, `__repr__`, `__eq__` for free. `asdict()` converts it to a dictionary for JSON serialization.

---

### Section 4: `DatasetValidator` Class (Lines 44-58)
```python
class DatasetValidator:
    REQUIRED_FIELDS = [
        "number", "opened_at", "priority", "category",
        "assignment_group", "short_description", "description"
    ]

    def __init__(self, config=None):
        self.config = config or ConfigManager()
        self.results: List[CheckResult] = []    # Accumulates all 12 check results
```
**`REQUIRED_FIELDS`** — These 7 columns MUST exist in any valid dataset. If any is missing, the first check fails immediately.

---

### Section 5: `validate_dataset()` — Main Entry Point (Lines 60-138)
```python
def validate_dataset(self, df, save_report=True, report_dir=None) -> Dict:
    self.results.clear()                      # Reset from any previous run

    # Execute all 12 checks sequentially
    self.results.append(self._check_missing_values(df))          # CHK-01
    self.results.append(self._check_duplicate_incidents(df))     # CHK-02
    self.results.append(self._check_invalid_timestamps(df))      # CHK-03
    self.results.append(self._check_invalid_categories(df))      # CHK-04
    self.results.append(self._check_invalid_assignment_groups(df))# CHK-05
    self.results.append(self._check_invalid_priorities(df))      # CHK-06
    self.results.append(self._check_sla_inconsistencies(df))     # CHK-07
    self.results.append(self._check_resolution_time_inconsistencies(df))  # CHK-08
    self.results.append(self._check_invalid_cmdb_references(df)) # CHK-09
    self.results.append(self._check_invalid_business_services(df))# CHK-10
    self.results.append(self._check_empty_descriptions(df))      # CHK-11
    self.results.append(self._check_empty_short_descriptions(df))# CHK-12

    passed_checks = sum(1 for r in self.results if r.passed)
    is_valid = (passed_checks == 12)          # ALL must pass

    summary = {
        "validation_timestamp": datetime.now().isoformat(),
        "total_records": len(df),
        "is_valid": is_valid,
        "checks": [asdict(r) for r in self.results]   # Convert to dicts for JSON
    }

    if save_report:
        self.save_validation_report(summary, report_dir)

    return summary
```
**Flow:** Run all 12 checks → count passes → build summary dict → save report → return.

---

### Section 6: The 12 Validation Rules (Lines 140-390)

#### CHK-01: Missing Values (Lines 140-175)
```python
def _check_missing_values(self, df):
    # First: check if required columns even exist in the schema
    missing_cols = [col for col in self.REQUIRED_FIELDS if col not in df.columns]
    if missing_cols:
        return CheckResult("CHK-01", ..., passed=False, error_count=len(df), ...)

    # Then: count null values across required fields
    missing_counts = df[self.REQUIRED_FIELDS].isnull().sum()
    total_missing = int(missing_counts.sum())
    passed = total_missing == 0

    # Collect up to 3 sample bad records for the report
    if not passed:
        missing_rows = df[df[self.REQUIRED_FIELDS].isnull().any(axis=1)].head(3)
```
**Two-level check:** First checks if columns exist at all. Then checks for nulls within existing columns.

#### CHK-02: Duplicate Incidents (Lines 177-193)
```python
def _check_duplicate_incidents(self, df):
    dup_mask = df["number"].duplicated(keep=False)   # Marks ALL copies (not just 2nd+)
    error_count = int(dup_mask.sum())
```
**`keep=False`** marks every row that has a duplicate — both the first occurrence and subsequent ones. This ensures the count includes all copies.

#### CHK-03: Invalid Timestamps (Lines 195-219)
```python
def _check_invalid_timestamps(self, df):
    # Check: resolved_at < opened_at (ticket resolved before it was opened?)
    invalid_res = df[res_mask & (pd.to_datetime(df["resolved_at"]) < pd.to_datetime(df["opened_at"]))]

    # Check: closed_at < resolved_at (ticket closed before it was resolved?)
    invalid_close = df[close_mask & (pd.to_datetime(df["closed_at"]) < pd.to_datetime(df["resolved_at"]))]
```
**Business logic:** Timestamps must flow: `opened_at → resolved_at → closed_at`. Any reversal means data corruption.

#### CHK-06: Invalid Priorities (Lines 253-272)
```python
def _check_invalid_priorities(self, df):
    priority_str = df["priority"].astype(str).str.strip().str[0]   # Take first character
    valid_priorities = {"1", "2", "3", "4", "5"}
    invalid_mask = ~priority_str.isin(valid_priorities)
```
**Smart parsing:** Priority can be integer `1` or string `"1 - High"`. Taking `str[0]` (first character) handles both formats.

#### CHK-07: SLA Inconsistencies (Lines 274-298)
```python
def _check_sla_inconsistencies(self, df):
    sla_targets = {1: 4.0, 2: 12.0, 3: 48.0, 4: 120.0, 5: 240.0}  # Hours per priority

    for idx, r in df[res_mask].iterrows():
        target = sla_targets.get(r["priority"], 48.0)
        if r["made_sla"] and r["resolution_time_hours"] > (target + 0.05):
            error_count += 1  # Says SLA met but took longer than allowed!
```
**Business rule:** Priority 1 = 4-hour SLA, Priority 2 = 12 hours, etc. If `made_sla=True` but resolution time exceeds the target, the data is contradictory.

#### CHK-08: Resolution Time Inconsistencies (Lines 300-330)
```python
def _check_resolution_time_inconsistencies(self, df):
    # Check 1: Negative resolution times
    neg_mask = df["resolution_time_hours"] < 0

    # Check 2: Does resolution_time_hours match (resolved_at - opened_at)?
    dt_diff_hours = (pd.to_datetime(df["resolved_at"]) - pd.to_datetime(df["opened_at"])).dt.total_seconds() / 3600.0
    diff_error = np.abs(dt_diff_hours - df["resolution_time_hours"]) > 0.5   # 30-min tolerance
```
**Two checks in one rule:** First checks for impossible negative times, then verifies the resolution_time_hours column actually matches the timestamp difference (within 30 minutes tolerance).

#### CHK-09 to CHK-12: Simple Null/Empty Checks (Lines 332-390)
All follow the same pattern:
```python
empty_mask = df["column"].isnull() | (df["column"].astype(str).str.strip() == "")
```
Check if the column is null OR is just whitespace. Report the count and up to 3 sample bad records.

---

### Section 7: `save_validation_report()` (Lines 392-451)
```python
def save_validation_report(self, summary, report_dir=None):
    # Resolve output directory
    out_dir = Path(report_dir or "reports")

    # Save JSON (machine-readable)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Save Markdown (human-readable)
    lines = [
        "# Dataset Validation Report",
        "| Rule ID | Quality Rule Name | Status | Anomaly Count | Error % | Details |",
        "|:---:|---|:---:|:---:|:---:|---|"
    ]
    for check in summary["checks"]:
        status_icon = "✅ PASS" if check["passed"] else "⚠️ FAIL"
        lines.append(f"| `{check['rule_id']}` | **{check['rule_name']}** | {status_icon} | ...")

    # If failures exist, append sample anomalies as JSON code blocks
    if not summary["is_valid"]:
        for check in summary["checks"]:
            if not check["passed"] and check["sample_anomalies"]:
                lines.append(f"### `{check['rule_id']}` — {check['rule_name']}")
                lines.append("```json")
                lines.append(json.dumps(check["sample_anomalies"], indent=2))
                lines.append("```")
```
**Dual output format:**
- `validation_report.json` — for other scripts to read programmatically
- `validation_report.md` — for humans to read in GitHub/VS Code, with ✅/⚠️ icons and formatted tables
