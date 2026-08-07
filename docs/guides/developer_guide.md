# Developer Guide — AI-Powered Incident Intelligence Platform

**First Citizens Bank — Enterprise Technology Division**

---

## 1. Document Control

| Field               | Value                                           |
|----------------------|-------------------------------------------------|
| **Document Title**   | Developer Guide                                 |
| **Project**          | AI-Powered Incident Intelligence Platform       |
| **Version**          | 1.0.0                                           |
| **Classification**   | Internal — Engineering Use Only                 |
| **Author**           | Enterprise AI Engineering Team                  |
| **Last Updated**     | 2026-07-10                                      |
| **Review Cycle**     | Quarterly                                       |
| **Approved By**      | VP, Enterprise Technology                       |

### Revision History

| Version | Date       | Author                | Description                        |
|---------|------------|-----------------------|------------------------------------|
| 1.0.0   | 2026-07-10 | AI Engineering Team   | Initial release                    |

---

## 2. Introduction

### 2.1 Purpose

This guide provides First Citizens Bank engineers with everything required to set up, develop, test, and maintain the **AI-Powered Incident Intelligence Platform**. It establishes coding standards, architectural conventions, and operational procedures that every contributor must follow.

### 2.2 Intended Audience

- Software Engineers assigned to the Incident Intelligence project
- Data Scientists contributing ML models or feature engineering
- QA Engineers writing and maintaining the test suite
- DevOps / Platform Engineers managing deployment and infrastructure
- Technical Leads conducting code reviews

### 2.3 Prerequisites

Before beginning development, ensure you have:

| Prerequisite            | Minimum Version | Notes                                    |
|-------------------------|-----------------|------------------------------------------|
| Python                  | 3.11.x          | Must be 3.11; 3.12+ is **not** tested   |
| Conda (Miniconda/Ana.)  | 24.x+           | Used for environment isolation           |
| Git                     | 2.40+           | Required for version control             |
| VS Code (recommended)   | Latest stable   | Extensions listed in §3.5               |
| OS                      | Windows 10/11   | Local execution only — no cloud deploy   |

> **Important:** This platform processes ServiceNow incident data subject to internal data governance policies. All execution is **local only** — no data may leave the corporate network.

---

## 3. Environment Setup

### 3.1 Clone the Repository

```bash
git clone <repository-url> incident_classification
cd incident_classification
```

### 3.2 Create the Conda Environment

The project ships with an `environment.yml` that pins every dependency:

```bash
conda env create -f environment.yml
```

This installs the full stack:

| Package               | Purpose                                      |
|-----------------------|----------------------------------------------|
| scikit-learn          | Classification & regression models           |
| sentence-transformers | Semantic embeddings for incident text         |
| faiss-cpu             | Approximate nearest-neighbor similarity search|
| shap                  | Model explainability                         |
| streamlit             | Interactive dashboard                        |
| openpyxl              | Excel report generation                      |
| pandas / numpy        | Data manipulation and numerical computation  |
| PyYAML                | Configuration file parsing                   |
| pytest                | Test framework                               |
| pytest-cov            | Coverage reporting                           |

### 3.3 Activate the Environment

```bash
conda activate incident_classification
```

Verify activation by confirming the environment name appears in your terminal prompt.

### 3.4 Verify Installation

Run the verification script to confirm all packages are importable and versions are correct:

```bash
python -c "
import sklearn, sentence_transformers, faiss, shap, streamlit
import openpyxl, pandas, numpy, yaml
print('All core packages imported successfully.')
print(f'  scikit-learn : {sklearn.__version__}')
print(f'  pandas       : {pandas.__version__}')
print(f'  numpy        : {numpy.__version__}')
print(f'  streamlit    : {streamlit.__version__}')
"
```

Expected output: all imports succeed with no errors.

### 3.5 IDE Setup — VS Code

Install the following extensions:

| Extension                  | ID                                    |
|----------------------------|---------------------------------------|
| Python                     | `ms-python.python`                    |
| Pylance                    | `ms-python.vscode-pylance`            |
| Python Debugger            | `ms-python.debugpy`                   |
| YAML                       | `redhat.vscode-yaml`                  |
| Markdown All in One        | `yzhang.markdown-all-in-one`          |
| GitLens                    | `eamodio.gitlens`                     |

Recommended `settings.json` (workspace):

```json
{
    "python.defaultInterpreterPath": "~/miniconda3/envs/incident_classification/python.exe",
    "python.analysis.typeCheckingMode": "basic",
    "python.analysis.autoImportCompletions": true,
    "editor.formatOnSave": true,
    "editor.rulers": [100],
    "files.trimTrailingWhitespace": true,
    "[python]": {
        "editor.defaultFormatter": "ms-python.black-formatter",
        "editor.tabSize": 4
    }
}
```

---

## 4. Project Structure

```
incident_classification/
├── config/
│   └── config.yaml                 # Central configuration (paths, model params, thresholds)
├── data/
│   ├── raw/                        # Unmodified ServiceNow exports — NEVER commit these
│   ├── processed/                  # Cleaned, feature-engineered datasets
│   └── reports/                    # Generated Excel / HTML reports
├── docs/                           # All project documentation
├── logs/                           # Runtime log files (auto-rotated)
├── models/
│   ├── classifiers/                # Serialized assignment-prediction models (.pkl)
│   ├── regressors/                 # Serialized resolution-time models (.pkl)
│   ├── embeddings/                 # Sentence-transformer model cache
│   └── faiss_index/                # FAISS index files for similarity search
├── notebooks/                      # Jupyter notebooks for EDA and prototyping
├── projects/                       # Sub-project or sprint-specific workspaces
├── src/
│   ├── data/
│   │   ├── ingestion.py            # ServiceNow data loading and validation
│   │   ├── cleaning.py             # Data cleaning, null handling, deduplication
│   │   └── feature_engineering.py  # Feature extraction & transformation pipeline
│   ├── models/
│   │   ├── assignment_predictor.py # Assignment group classification (Random Forest)
│   │   ├── resolution_time_predictor.py  # Resolution time regression
│   │   ├── hyperparameter_optimizer.py   # GridSearch / RandomSearch tuning
│   │   └── model_registry.py      # Model versioning, serialization, loading
│   ├── resolution/
│   │   ├── similarity_engine.py    # FAISS-based semantic similarity search
│   │   ├── resolution_recommender.py  # Top-K resolution recommendation logic
│   │   └── rag_engine.py           # Retrieval-Augmented Generation pipeline
│   ├── explainability/
│   │   └── shap_explainer.py       # SHAP value computation & visualization
│   ├── dashboard/
│   │   ├── app.py                  # Streamlit application entry point
│   │   └── pages/                  # Multi-page Streamlit dashboard pages
│   ├── reporting/
│   │   └── excel_reporter.py       # Automated Excel report generation
│   └── utils/
│       ├── logger.py               # Centralized logging configuration
│       └── config_loader.py        # Singleton YAML config loader
├── tests/                          # All test code (mirrors src/ structure)
├── scripts/
│   └── setup.sh                    # One-command environment bootstrap
├── environment.yml                 # Conda environment specification
├── requirements.txt                # pip-compatible dependency list (fallback)
└── README.md                       # Project overview and quick-start
```

> **Rule:** The `src/` directory is the single source of truth for production code. Notebooks are for exploration only and must never be imported by `src/` modules.

---

## 5. Configuration System

### 5.1 config.yaml Structure

All runtime parameters live in `config/config.yaml`. Example:

```yaml
# config/config.yaml

project:
  name: "Incident Intelligence Platform"
  version: "1.0.0"
  environment: "development"          # development | staging | production

paths:
  raw_data: "data/raw"
  processed_data: "data/processed"
  reports: "data/reports"
  models: "models"
  logs: "logs"
  faiss_index: "models/faiss_index"

data:
  source: "servicenow"
  file_format: "xlsx"
  date_column: "opened_at"
  text_columns:
    - "short_description"
    - "description"
    - "close_notes"
  target_column: "assignment_group"
  max_categories: 50

models:
  assignment_predictor:
    algorithm: "random_forest"
    n_estimators: 200
    max_depth: 20
    min_samples_split: 5
    test_size: 0.2
    random_state: 42
  resolution_time:
    algorithm: "gradient_boosting"
    n_estimators: 150
    max_depth: 10
    test_size: 0.2

embeddings:
  model_name: "all-MiniLM-L6-v2"
  batch_size: 64
  max_seq_length: 256

similarity:
  top_k: 5
  similarity_threshold: 0.75

explainability:
  max_display_features: 20
  sample_size: 100

logging:
  level: "INFO"
  max_bytes: 10485760                 # 10 MB
  backup_count: 5

dashboard:
  page_title: "Incident Intelligence"
  theme: "dark"
  port: 8501
```

### 5.2 ConfigLoader — Singleton Pattern

The `ConfigLoader` class ensures configuration is loaded exactly once and shared across the application:

```python
# src/utils/config_loader.py

import yaml
from pathlib import Path
from typing import Any, Dict, Optional


class ConfigLoader:
    """Thread-safe singleton configuration loader.

    Loads config/config.yaml once and provides dictionary-style access
    to all configuration parameters.

    Usage:
        config = ConfigLoader()
        model_params = config.get("models.assignment_predictor")
    """

    _instance: Optional["ConfigLoader"] = None
    _config: Dict[str, Any] = {}

    def __new__(cls, config_path: str = "config/config.yaml") -> "ConfigLoader":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load(config_path)
        return cls._instance

    def _load(self, config_path: str) -> None:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            self._config = yaml.safe_load(f)

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a nested config value using dot notation.

        Args:
            key: Dot-separated path (e.g., 'models.assignment_predictor.n_estimators').
            default: Value returned if the key does not exist.

        Returns:
            The configuration value, or default if not found.
        """
        keys = key.split(".")
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
            if value is None:
                return default
        return value
```

### 5.3 Adding New Configuration Parameters

1. Add the parameter to `config/config.yaml` under the appropriate section.
2. Access it via `ConfigLoader().get("section.parameter")`.
3. Document the parameter with an inline YAML comment.
4. If the parameter has validation constraints, add validation in `ConfigLoader._load()`.

### 5.4 Environment-Specific Overrides

For environment-specific behavior, use the `project.environment` field:

```python
config = ConfigLoader()
env = config.get("project.environment")

if env == "production":
    log_level = "WARNING"
elif env == "development":
    log_level = "DEBUG"
```

> **Note:** Do **not** create separate config files per environment. Use conditional logic within the application to minimize configuration drift.

---

## 6. Logging System

### 6.1 Using the Logger

Every module should obtain its logger through the centralized utility:

```python
from src.utils.logger import get_logger

logger = get_logger(__name__)

logger.info("Pipeline started for %d records.", record_count)
logger.warning("Missing values detected in column '%s'.", col_name)
logger.error("Model training failed: %s", str(e))
```

### 6.2 Log Levels

| Level      | When to Use                                                        |
|------------|--------------------------------------------------------------------|
| `DEBUG`    | Detailed diagnostic info (feature shapes, intermediate values)     |
| `INFO`     | Routine operational events (pipeline start/end, model loaded)      |
| `WARNING`  | Unexpected but recoverable situations (fallback values used)       |
| `ERROR`    | Failures that prevent a specific operation from completing         |
| `CRITICAL` | System-wide failures requiring immediate attention                 |

### 6.3 Log File Location and Rotation

- **Location:** `logs/incident_platform.log`
- **Rotation:** Managed via `RotatingFileHandler`
  - Max file size: **10 MB** (configurable in `config.yaml`)
  - Backup count: **5** (keeps `incident_platform.log.1` through `.5`)
- **Console output:** `INFO` and above are also printed to stdout during development.

### 6.4 Logger Implementation Reference

```python
# src/utils/logger.py

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from src.utils.config_loader import ConfigLoader


def get_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """Create or retrieve a configured logger.

    Args:
        name: Logger name (typically __name__).
        log_file: Override log file path. Defaults to config value.

    Returns:
        Configured logging.Logger instance.
    """
    config = ConfigLoader()
    level = config.get("logging.level", "INFO")
    max_bytes = config.get("logging.max_bytes", 10_485_760)
    backup_count = config.get("logging.backup_count", 5)

    if log_file is None:
        log_dir = Path(config.get("paths.logs", "logs"))
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = str(log_dir / "incident_platform.log")

    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(getattr(logging, level.upper(), logging.INFO))

        formatter = logging.Formatter(
            fmt="%(asctime)s | %(name)-30s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Rotating file handler
        fh = RotatingFileHandler(
            log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
        )
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger
```

---

## 7. Coding Standards

### 7.1 SOLID Principles

| Principle                   | Application in This Project                                                                                  |
|-----------------------------|--------------------------------------------------------------------------------------------------------------|
| **Single Responsibility**   | `ingestion.py` handles only data loading; `cleaning.py` handles only cleaning — never both in one module.    |
| **Open/Closed**             | New models extend `BasePredictor` without modifying existing predictor classes.                               |
| **Liskov Substitution**     | Any class implementing `BasePredictor` can be swapped in without breaking the pipeline.                      |
| **Interface Segregation**   | Separate interfaces for `Trainable`, `Explainable`, and `Serializable` — a model only implements what it needs.|
| **Dependency Inversion**    | Pipeline stages depend on abstractions (`BasePredictor`), not concrete classes (`RandomForestPredictor`).    |

Example — Open/Closed with a base class:

```python
from abc import ABC, abstractmethod
from typing import Any, Dict
import pandas as pd


class BasePredictor(ABC):
    """Abstract base for all predictive models."""

    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the model on provided features and labels."""
        ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> Any:
        """Generate predictions for the input data."""
        ...

    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """Return current model hyperparameters."""
        ...
```

### 7.2 Design Patterns Used

| Pattern       | Where Used                          | Rationale                                            |
|---------------|-------------------------------------|------------------------------------------------------|
| **Singleton** | `ConfigLoader`, `Logger`            | Single shared instance, avoid redundant I/O          |
| **Strategy**  | `BasePredictor` subclasses          | Swap algorithms without changing calling code        |
| **Factory**   | `ModelRegistry.load(model_name)`    | Centralized model instantiation and versioning       |
| **Observer**  | Dashboard callbacks / Streamlit     | UI components react to state changes                 |

### 7.3 Type Hints — Mandatory

Every function signature **must** include type hints. Use `typing` module constructs for complex types:

```python
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np


def compute_feature_importance(
    model: "RandomForestClassifier",
    feature_names: List[str],
    top_n: int = 10,
) -> Dict[str, float]:
    """Return the top-N feature importances as a dictionary."""
    ...
```

### 7.4 Docstring Format — Google Style

All public classes, methods, and functions require Google-style docstrings:

```python
def clean_text(text: str, remove_stopwords: bool = True) -> str:
    """Normalize and clean raw incident text.

    Converts to lowercase, removes special characters, and optionally
    strips English stopwords.

    Args:
        text: Raw incident description string.
        remove_stopwords: If True, common English stopwords are removed.

    Returns:
        Cleaned text string suitable for embedding generation.

    Raises:
        ValueError: If text is empty or None.
    """
    ...
```

### 7.5 Naming Conventions

| Element         | Convention           | Example                         |
|-----------------|----------------------|---------------------------------|
| Module          | `snake_case`         | `feature_engineering.py`        |
| Class           | `PascalCase`         | `AssignmentPredictor`           |
| Function/Method | `snake_case`         | `compute_similarity()`          |
| Constant        | `UPPER_SNAKE_CASE`   | `MAX_SEQUENCE_LENGTH`           |
| Private member  | `_leading_underscore`| `_validate_input()`             |
| Variable        | `snake_case`         | `raw_dataframe`                 |

### 7.6 Import Ordering

Follow the **isort** default profile (compatible with Black):

```python
# 1. Standard library
import os
import logging
from pathlib import Path
from typing import Dict, List

# 2. Third-party packages
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# 3. Local application imports
from src.utils.config_loader import ConfigLoader
from src.utils.logger import get_logger
```

---

## 8. How to Add New Features / Modules

### 8.0 Registry Compliance & Pipeline Contracts (Mandatory `v1.5.0` Contract)

Before adding or modifying any feature column, model, or vector index, you **must** comply with the Central Enterprise Registry Layer:
1. **Never Hardcode Feature Names:** Downstream modules (`Random Forest`, `Embeddings`, `FAISS`, `Dashboard`, `API`, `RAG`) must retrieve authorized feature lists dynamically via `PipelineContractValidator` (`src/data/pipeline_contracts.py`).
2. **Feature Registration (`src/data/feature_registry.py`):** Any new dataset attribute must be registered as a `FeatureDefinition` across all 22 governance dimensions. Ensure proper setting of `target_leakage_classification` (`safe`, `warning`, or `blocked`).
3. **Lineage Documentation (`src/data/feature_lineage.py`):** If adding a derived feature, record its exact formula and parent dependencies using `LineageEdge`.
4. **Model SHA256 Verification (`src/ml/model_registry.py`):** When saving or loading `.joblib` model weights, use `ModelRegistry.register_model()` and `verify_and_load_model_path()`. Tampered files or models attempting to consume `blocked` leakage predictors will throw a `ModelValidationException`.

### 8.1 Adding a New ML Model

1. **Create the module** in `src/models/`:
   ```
   src/models/priority_predictor.py
   ```
2. **Extend `BasePredictor` and Query Pipeline Contracts:**
   ```python
   from src.models.base import BasePredictor
   from src.data.pipeline_contracts import PipelineContractValidator
   
   class PriorityPredictor(BasePredictor):
       def __init__(self):
           self.validator = PipelineContractValidator()
           self.authorized_features = self.validator.get_random_forest_features("priority")

       def train(self, X, y):
           # Verify compliance before training
           is_compliant, violations = self.validator.validate_dataframe_compliance(X, expected_usage="triage_prediction")
           if not is_compliant:
               raise ValueError(f"Dataframe violates pipeline contract: {violations}")
           ...
   ```
3. **Add configuration** to `config/config.yaml`:
   ```yaml
   models:
     priority_predictor:
       algorithm: "xgboost"
       n_estimators: 100
   ```
4. **Register the model artifact** in `ModelRegistry`:
   ```python
   ModelRegistry.get_instance().register_model(
       model_name="priority_predictor",
       version="v1.0.0",
       training_dataset_uri="datasets/synthetic/v1/incidents.csv",
       dataset_version="v1",
       hyperparameters={"algorithm": "xgboost", "n_estimators": 100},
       metrics={"f1_macro": 0.88},
       features_used=self.authorized_features,
       target_variable="priority",
       model_file_path="models/priority_predictor_v1.joblib"
   )
   ```
5. **Write tests** in `tests/models/test_priority_predictor.py`.
6. **Update documentation** in `docs/`.

### 8.2 Adding a New Dashboard Page

1. **Create the page file** in `src/dashboard/pages/`:
   ```
   src/dashboard/pages/3_Priority_Analysis.py
   ```
   (Streamlit uses the filename prefix for ordering.)
2. **Implement the page:**
   ```python
   import streamlit as st
   from src.utils.config_loader import ConfigLoader
   from src.utils.logger import get_logger
   
   logger = get_logger(__name__)
   config = ConfigLoader()
   
   st.set_page_config(page_title="Priority Analysis", layout="wide")
   st.title("Priority Analysis")
   
   # Page content here
   ```
3. **Add navigation entry** if using a custom sidebar.
4. **Write integration tests** for the page.

### 8.3 Adding a New Report Type

1. **Create a new reporter class** inheriting from a base reporter:
   ```python
   # src/reporting/pdf_reporter.py
   from src.reporting.base_reporter import BaseReporter
   
   class PDFReporter(BaseReporter):
       def generate(self, data, output_path):
           ...
   ```
2. **Add the output path** to `config.yaml` if needed.
3. **Register the report type** in the reporting factory.
4. **Write unit tests** with sample data.

### 8.4 New Feature Checklist

- [ ] Module follows SOLID principles and extends appropriate base class
- [ ] Type hints on all function signatures
- [ ] Google-style docstrings on all public APIs
- [ ] Configuration added to `config.yaml` (if applicable)
- [ ] Logging integrated via `get_logger(__name__)`
- [ ] Unit tests written with ≥80% coverage for the new module
- [ ] No raw data committed to version control
- [ ] Documentation updated
- [ ] Code reviewed and approved by at least one peer

---

## 9. Testing Guide

### 9.1 Test Structure

```
tests/
├── unit/
│   ├── test_ingestion.py
│   ├── test_cleaning.py
│   ├── test_feature_engineering.py
│   ├── test_assignment_predictor.py
│   └── test_config_loader.py
├── integration/
│   ├── test_pipeline_end_to_end.py
│   └── test_dashboard_pages.py
├── e2e/
│   └── test_full_workflow.py
└── conftest.py                     # Shared fixtures
```

### 9.2 Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run only unit tests
pytest tests/unit/ -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=html --cov-report=term-missing

# Run a specific test file
pytest tests/unit/test_assignment_predictor.py -v

# Run tests matching a keyword
pytest tests/ -k "test_train" -v
```

### 9.3 Writing Test Cases

```python
# tests/unit/test_assignment_predictor.py

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

from src.models.assignment_predictor import AssignmentPredictor


@pytest.fixture
def sample_training_data():
    """Generate synthetic training data for tests."""
    np.random.seed(42)
    X = pd.DataFrame({
        "feature_1": np.random.rand(100),
        "feature_2": np.random.rand(100),
        "feature_3": np.random.randint(0, 5, 100),
    })
    y = pd.Series(np.random.choice(["GroupA", "GroupB", "GroupC"], 100))
    return X, y


class TestAssignmentPredictor:
    """Unit tests for AssignmentPredictor."""

    def test_train_creates_model(self, sample_training_data):
        """Verify that training produces a non-None model."""
        X, y = sample_training_data
        predictor = AssignmentPredictor()
        predictor.train(X, y)
        assert predictor.model is not None

    def test_predict_returns_correct_shape(self, sample_training_data):
        """Predictions must have the same length as input."""
        X, y = sample_training_data
        predictor = AssignmentPredictor()
        predictor.train(X, y)
        predictions = predictor.predict(X)
        assert len(predictions) == len(X)

    def test_predict_without_training_raises(self):
        """Calling predict before train must raise RuntimeError."""
        predictor = AssignmentPredictor()
        with pytest.raises(RuntimeError, match="Model has not been trained"):
            predictor.predict(pd.DataFrame({"a": [1]}))
```

### 9.4 Mocking External Dependencies

Use `unittest.mock` to isolate units from file I/O, model loading, and other side effects:

```python
from unittest.mock import patch, mock_open

@patch("builtins.open", mock_open(read_data="key: value"))
def test_config_loader_reads_yaml():
    """ConfigLoader should parse YAML content from file."""
    from src.utils.config_loader import ConfigLoader
    ConfigLoader._instance = None  # Reset singleton for test isolation
    config = ConfigLoader("dummy_path.yaml")
    assert config.get("key") == "value"
```

### 9.5 Coverage Targets

| Scope          | Target  |
|----------------|---------|
| Overall        | ≥ 80%   |
| `src/models/`  | ≥ 85%   |
| `src/utils/`   | ≥ 90%   |
| `src/data/`    | ≥ 80%   |

Generate the HTML coverage report and review it before submitting a PR:

```bash
pytest tests/ --cov=src --cov-report=html
# Open htmlcov/index.html in a browser
```

---

## 10. Git Workflow

### 10.1 Branch Naming Conventions

| Branch Type   | Pattern                        | Example                               |
|---------------|--------------------------------|---------------------------------------|
| Feature       | `feature/<ticket>-<desc>`      | `feature/INC-42-add-priority-model`   |
| Bug Fix       | `bugfix/<ticket>-<desc>`       | `bugfix/INC-58-fix-null-handling`     |
| Hotfix        | `hotfix/<ticket>-<desc>`       | `hotfix/INC-99-critical-data-leak`    |
| Release       | `release/<version>`            | `release/1.2.0`                       |

### 10.2 Commit Message Format — Conventional Commits

```
<type>(<scope>): <short description>

[optional body]

[optional footer(s)]
```

**Types:**

| Type       | Usage                                          |
|------------|-------------------------------------------------|
| `feat`     | New feature                                     |
| `fix`      | Bug fix                                         |
| `docs`     | Documentation only                              |
| `refactor` | Code restructuring without behavior change      |
| `test`     | Adding or updating tests                        |
| `chore`    | Build scripts, CI config, dependency updates    |
| `perf`     | Performance improvement                         |

**Examples:**

```
feat(models): add PriorityPredictor with gradient boosting

Implements a new predictor for incident priority classification
using XGBoost. Includes hyperparameter defaults in config.yaml.

Refs: INC-42
```

```
fix(data): handle NaN values in short_description column

Previously, null descriptions caused a crash during embedding
generation. Now defaults to empty string before tokenization.

Fixes: INC-58
```

### 10.3 Pull Request Process

1. **Create a branch** from `main` using the naming convention above.
2. **Develop and test** locally — all tests must pass.
3. **Push** the branch and open a Pull Request.
4. **Fill in the PR template** (description, testing steps, screenshots if UI).
5. **Request review** from at least one team member.
6. **Address feedback** — push additional commits, do not force-push.
7. **Merge** via **squash merge** after approval.
8. **Delete** the feature branch after merge.

### 10.4 Code Review Checklist

- [ ] Code follows the project's coding standards (§7)
- [ ] All public APIs have Google-style docstrings
- [ ] Type hints are present on all function signatures
- [ ] No hardcoded paths or magic numbers — use `config.yaml`
- [ ] Unit tests cover new functionality (≥80% coverage)
- [ ] No sensitive data (credentials, PII) in the diff
- [ ] Logging is appropriate and not excessive
- [ ] No `print()` statements — use `logger` instead
- [ ] Import ordering follows the standard (§7.6)
- [ ] No breaking changes to existing public interfaces

---

## 11. Data Handling

### 11.1 Working with Incident Data

All incident data originates from **ServiceNow** exports (typically `.xlsx`). The standard workflow:

```
ServiceNow Export → data/raw/ → ingestion.py → cleaning.py → feature_engineering.py → data/processed/
```

- **Raw data** is placed in `data/raw/` and must never be modified in place.
- **Processed data** in `data/processed/` is the output of the cleaning and feature engineering pipeline.
- Always use `ingestion.py` to load data — never use `pd.read_excel()` directly in other modules.

### 11.2 Data Privacy Considerations

| Rule                                              | Enforcement                                      |
|---------------------------------------------------|--------------------------------------------------|
| Raw data must **never** be committed to Git        | `.gitignore` excludes `data/raw/`                |
| Processed data should not be committed             | `.gitignore` excludes `data/processed/`          |
| PII fields must be masked before any logging       | `cleaning.py` applies masking during processing |
| All execution is local — no cloud APIs             | Architecture enforced; no external HTTP calls    |
| Model artifacts may contain data fingerprints      | `models/` is excluded from Git                   |

### 11.3 .gitignore Essentials

Ensure the following entries are always present:

```gitignore
# Data — never commit
data/raw/
data/processed/
data/reports/

# Model artifacts
models/classifiers/
models/regressors/
models/embeddings/
models/faiss_index/

# Logs
logs/

# Environment
*.pyc
__pycache__/
.env
```

---

## 12. Troubleshooting

### Common Issues and Solutions

| # | Issue                                          | Cause                                                    | Solution                                                         |
|---|------------------------------------------------|----------------------------------------------------------|------------------------------------------------------------------|
| 1 | `ModuleNotFoundError: No module named 'src'`   | Running script from wrong directory                      | Run from project root: `python -m src.models.assignment_predictor` |
| 2 | `FileNotFoundError: config/config.yaml`        | Working directory is not the project root                | `cd incident_classification` before running                      |
| 3 | `CUDA not available` warning from transformers | Expected on CPU-only setups                              | Safe to ignore — `faiss-cpu` is used intentionally               |
| 4 | `MemoryError` during FAISS indexing            | Dataset too large for available RAM                      | Reduce dataset size or use `faiss.IndexIVFFlat` for approximate  |
| 5 | `openpyxl` cannot read `.xls` files            | File is old Excel format, not `.xlsx`                    | Re-export from ServiceNow as `.xlsx`                             |
| 6 | Streamlit port already in use                  | Previous session still running                           | `streamlit run src/dashboard/app.py --server.port 8502`          |
| 7 | `ConvergenceWarning` during model training     | Model did not converge with default iterations           | Increase `max_iter` in config or scale features                  |
| 8 | Import errors after pulling latest code        | New dependencies added                                   | `conda env update -f environment.yml --prune`                    |
| 9 | Tests fail with `singleton already initialized`| ConfigLoader singleton persists across tests             | Add `ConfigLoader._instance = None` in test fixtures             |
|10 | Git rejects push due to large files            | Model or data files accidentally staged                  | Remove from staging: `git reset HEAD <file>`, verify `.gitignore`|

---

## 13. Quick Reference — Common Commands

| Task                              | Command                                                              |
|-----------------------------------|----------------------------------------------------------------------|
| Create environment                | `conda env create -f environment.yml`                                |
| Activate environment              | `conda activate incident_classification`                             |
| Update environment                | `conda env update -f environment.yml --prune`                        |
| Run full test suite               | `pytest tests/ -v`                                                   |
| Run tests with coverage           | `pytest tests/ --cov=src --cov-report=html`                         |
| Launch dashboard                  | `streamlit run src/dashboard/app.py`                                 |
| Train assignment model            | `python -m src.models.assignment_predictor`                          |
| Generate reports                  | `python -m src.reporting.excel_reporter`                             |
| Check code style                  | `flake8 src/ --max-line-length=100`                                  |
| Sort imports                      | `isort src/ tests/`                                                  |
| Format code                       | `black src/ tests/ --line-length=100`                                |
| View logs                         | `Get-Content logs/incident_platform.log -Tail 50` (PowerShell)      |
| Build FAISS index                 | `python -m src.resolution.similarity_engine`                         |
| Run SHAP explainer                | `python -m src.explainability.shap_explainer`                        |

---

> **Questions or issues?** Contact the AI Engineering Team via the internal collaboration channel or raise a ticket in the project's issue tracker.

---

*This document is maintained by the Enterprise AI Engineering Team at First Citizens Bank. All rights reserved.*
