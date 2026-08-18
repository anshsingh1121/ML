# Enterprise AI Incident Intelligence Platform
## Complete Technical Documentation & Study Guide

> **Version:** 2.0.0 | **Total Codebase:** ~7,500 lines across 40+ Python files | **Architecture:** Dual-Engine Hybrid AI (CatBoost + FAISS)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Complete Technology Stack](#2-complete-technology-stack)
3. [Directory Structure](#3-directory-structure)
4. [End-to-End Pipeline Flow](#4-end-to-end-pipeline-flow)
5. [Entry Points & Execution](#5-entry-points--execution)
6. [Configuration Layer](#6-configuration-layer)
7. [Utility Layer (`src/utils/`)](#7-utility-layer-srcutils)
8. [Data Governance Layer (`src/data/`)](#8-data-governance-layer-srcdata)
9. [Preprocessing Layer (`src/preprocessing/`)](#9-preprocessing-layer-srcpreprocessing)
10. [Machine Learning Layer (`src/ml/`)](#10-machine-learning-layer-srcml)
11. [Hybrid Intelligence Engine (`src/ml/hybrid/`)](#11-hybrid-intelligence-engine-srcmlhybrid)
12. [Explainability Layer (`src/ml/explainability/`)](#12-explainability-layer-srcmlexplainability)
13. [Dashboard UI (`src/dashboard/`)](#13-dashboard-ui-srcdashboard)
14. [Testing Infrastructure](#14-testing-infrastructure)
15. [Key Algorithms & Mathematics](#15-key-algorithms--mathematics)
16. [File-by-File Reference Matrix](#16-file-by-file-reference-matrix)

---

## 1. Project Overview

The **Enterprise AI Incident Intelligence Platform** is a production-grade machine learning system designed for **IT Service Management (ITSM)** environments. It automates two critical IT operations tasks:

1. **Ticket Routing (Classification):** Predicts which specialized technical team (`assignment_group`) should handle an incoming IT incident.
2. **Resolution Time Estimation (Regression):** Predicts the Mean Time To Resolve (MTTR) in hours for each incident.

Additionally, it provides:
- **Semantic Precedent Search:** Finds the Top-K most similar historical incidents using neural text embeddings and vector similarity.
- **Explainable AI:** SHAP-based feature attribution explaining *why* the AI made each decision.
- **Hybrid Intelligence:** Fuses ML predictions with historical evidence to produce confidence-scored recommendations.

### Core Design Principles
- **Zero Target Leakage:** Post-resolution fields (`close_notes`, `resolved_at`) are mathematically blocked from entering prediction models.
- **Enterprise Governance:** Every feature is tracked across 22 dimensions (data type, leakage risk, encoding strategy, imputation rules, etc.).
- **100% On-Premise:** No data leaves the corporate network. All models run locally.
- **Robust I/O:** Multi-encoding file readers handle `utf-8`, `utf-8-sig`, `cp1252`, and `latin-1` automatically.

---

## 2. Complete Technology Stack

### Core Language
| Technology | Version | Purpose |
|---|---|---|
| **Python** | ≥ 3.11 | Primary programming language |

### Machine Learning & AI
| Library | Version | Purpose |
|---|---|---|
| **CatBoost** | 1.2.x | Gradient Boosting classifier & regressor optimized for categorical data |
| **scikit-learn** | 1.5.x | Pipeline construction, preprocessing, `RandomizedSearchCV` HPO, evaluation metrics |
| **FAISS (faiss-cpu)** | 1.8.x | Meta's vector similarity search engine (Flat/IVF indexes) |
| **SHAP** | 0.46.x | Game-theoretic feature attribution (TreeExplainer) |
| **NumPy** | 1.26.x | Numerical array operations, trigonometry, linear algebra |
| **SciPy** | 1.13.x | Scientific computing utilities |
| **Joblib** | 1.4.x | Serialization of scikit-learn pipelines (`.pkl` files) |

### Data Processing
| Library | Version | Purpose |
|---|---|---|
| **Pandas** | 2.2.x | DataFrame operations, CSV/Parquet I/O, datetime handling |
| **OpenPyXL** | 3.1.x | Excel file reading |
| **XlsxWriter** | 3.2.x | Excel report generation |

### NLP (Natural Language Processing)
| Component | Technology | Purpose |
|---|---|---|
| **Tokenization** | `TfidfVectorizer` (scikit-learn) | Converts text into TF-IDF weighted sparse vectors |
| **Dimensionality Reduction** | `TruncatedSVD` (scikit-learn) | Reduces TF-IDF to 384-dimensional dense vectors |
| **Normalization** | `sklearn.preprocessing.normalize` | L2 unit-length normalization for cosine similarity |
| **Text Cleaning** | Python `re`, `unicodedata` | Regex-based HTML stripping, IT lemmatization, stopword filtering |

### Visualization
| Library | Version | Purpose |
|---|---|---|
| **Matplotlib** | 3.9.x | Chart rendering (confusion matrices, ROC curves, feature importance) |
| **Seaborn** | 0.13.x | Statistical plots (heatmaps, bar charts, distributions) |
| **Plotly** | ≥ 5.18 | Interactive dashboard charts |

### Web UI
| Library | Version | Purpose |
|---|---|---|
| **Streamlit** | ≥ 1.30 | Real-time web dashboard for live predictions |

### Configuration & Environment
| Library | Version | Purpose |
|---|---|---|
| **PyYAML** | 6.0.x | YAML configuration file parsing |
| **Pydantic** | 2.10.x | Data validation and settings management |
| **python-dotenv** | 1.1.x | `.env` file loading for secrets and environment variables |

### Logging & CLI
| Library | Version | Purpose |
|---|---|---|
| **structlog** | 24.4.x | Structured logging framework |
| **Rich** | 13.9.x | Terminal UI formatting (colors, tables, progress bars) |
| **Click** | 8.1.x | CLI framework |
| **tqdm** | 4.67.x | Progress bars for batch processing |

### Testing & Quality
| Library | Version | Purpose |
|---|---|---|
| **pytest** | 8.3.x | Test framework (23 unit test files) |
| **pytest-cov** | 6.1.x | Code coverage reporting (80% threshold) |
| **pytest-mock** | 3.14.x | Mock objects for isolated unit testing |
| **Ruff** | — | Linter and formatter (120-char line length, 30+ rule categories) |
| **mypy** | — | Static type checking |

### Build & Packaging
| Technology | Purpose |
|---|---|
| **pyproject.toml** | PEP 518/621 project metadata, dependencies, tool configs |
| **setuptools** | Build backend for `pip install -e .` |
| **requirements.txt** | Pinned dependencies for `pip install -r` |

---

## 3. Directory Structure

```
incident_platform/
│
├── main.py                          # Root CLI entry point (154 lines)
├── run_dashboard.py                 # Streamlit dashboard launcher (7 lines)
├── .env                             # Environment variables & secrets
├── .env.example                     # Template for .env
├── requirements.txt                 # Pinned pip dependencies
├── pyproject.toml                   # PEP 518/621 build & tool config (250 lines)
├── environment.yml                  # Conda environment specification
│
├── config/                          # ── Configuration Files ──
│   ├── config.yaml                  # Platform-wide settings (139 lines)
│   ├── model_config.yaml            # ML hyperparameters & FAISS config (97 lines)
│   ├── logging.yaml                 # Python logging dictConfig (97 lines)
│   └── servicenow.yaml             # ServiceNow REST API mapping (52 lines)
│
├── data/                            # ── Data Storage ──
│   ├── raw/                         # Raw input CSV files
│   │   └── incidents.csv            # Corporate incident dataset (25 columns)
│   └── processed/                   # Pipeline output files
│       ├── train.csv                # Training partition (70-80%)
│       ├── val.csv                  # Validation partition (10-15%)
│       ├── test.csv                 # Test partition (10-15%)
│       └── master_engineered_incidents.csv
│
├── models/                          # ── Trained Model Artifacts ──
│   ├── catboost_assignment_group.pkl     # Classification pipeline
│   ├── catboost_resolution_time_hours.pkl # Regression pipeline
│   ├── model_registry.json          # Model metadata & SHA256 checksums
│   └── embeddings/                  # TF-IDF + SVD pipeline & vectors
│       ├── tfidf_svd_pipeline.pkl   # Fitted text embedding pipeline
│       ├── incident_embeddings.npy  # Dense 384-D vector matrix
│       └── incident_metadata.csv    # Aligned metadata for vectors
│
├── indexes/                         # ── FAISS Vector Indexes ──
│   ├── incident_semantic_index_latest.index  # Binary FAISS index
│   ├── incident_semantic_index_latest_metadata.csv
│   └── embedding_registry.json      # Index metadata & checksums
│
├── reports/                         # ── Generated Reports ──
│   ├── cleaning_report.json/.md     # Data cleaning audit
│   ├── eda_report.json/.md/.html    # Exploratory data analysis
│   ├── feature_engineering_report.json/.md
│   ├── text_preprocessing_report.json/.md
│   ├── split_report.json/.md        # Train/Val/Test partition audit
│   ├── validation_report.json/.md   # 12-rule data quality check
│   ├── ml_readiness_report.json/.md # Pre-training diagnostic
│   ├── classification_report.json/.md # Model evaluation metrics
│   ├── regression_report.json/.md
│   ├── feature_registry.json/.md    # Feature governance catalog
│   ├── feature_lineage.json/.md     # Feature derivation graph
│   ├── hybrid_prediction.json/.md/.csv  # Hybrid inference output
│   ├── similarity_results.csv/.md   # Semantic search results
│   └── figures/                     # EDA & evaluation charts (PNG)
│
├── logs/                            # ── Application Logs ──
│   ├── app.log                      # General application log (10MB rotation)
│   ├── ml_training.log              # ML training events
│   ├── api_access.log               # API access events
│   └── error.log                    # Error-only log
│
├── src/                             # ══ SOURCE CODE ══
│   ├── __init__.py                  # Package root (version 2.0.0)
│   │
│   ├── cli/                         # ── Command Line Interface ──
│   │   ├── __init__.py              # Console script entry point
│   │   └── main_cli.py             # EnterpriseCLI controller (849 lines)
│   │
│   ├── dashboard/                   # ── Web UI ──
│   │   ├── __init__.py
│   │   └── app.py                   # Streamlit dashboard (118 lines)
│   │
│   ├── data/                        # ── Data Governance ──
│   │   ├── __init__.py              # Package exports (25 lines)
│   │   ├── feature_registry.py      # 22-dim feature catalog (670 lines)
│   │   ├── feature_lineage.py       # Derivation graph tracker (273 lines)
│   │   ├── pipeline_contracts.py    # Schema contract validator (106 lines)
│   │   ├── quality_gate.py          # 6-gate certification engine (270 lines)
│   │   ├── readiness.py             # ML diagnostic evaluator (375 lines)
│   │   ├── validation.py            # 12-rule data quality engine (452 lines)
│   │   └── version_manager.py       # Dataset version control (215 lines)
│   │
│   ├── preprocessing/               # ── Data Preprocessing ──
│   │   ├── __init__.py              # Package exports
│   │   ├── cleaner.py               # 8-step data cleaning (454 lines)
│   │   ├── eda.py                   # Automated EDA engine (597 lines)
│   │   ├── engineer.py              # Feature engineering (470 lines)
│   │   ├── enricher.py              # External data enrichment (60 lines)
│   │   ├── splitter.py              # Train/Val/Test splitting (287 lines)
│   │   └── text_preprocessor.py     # NLP text normalization (273 lines)
│   │
│   ├── ml/                          # ── Machine Learning ──
│   │   ├── __init__.py              # Package exports
│   │   ├── model_registry.py        # Model artifact registry (248 lines)
│   │   ├── embedding_registry.py    # Vector index registry (188 lines)
│   │   │
│   │   ├── catboost/                # ── CatBoost Engine ──
│   │   │   ├── __init__.py
│   │   │   ├── trainer.py           # Training + HPO engine (536 lines)
│   │   │   ├── evaluator.py         # Metrics & charts (389 lines)
│   │   │   └── transformers.py      # Custom sklearn transformers (243 lines)
│   │   │
│   │   ├── semantic/                # ── FAISS Semantic Engine ──
│   │   │   ├── __init__.py
│   │   │   ├── embedding_generator.py  # TF-IDF + SVD embeddings (266 lines)
│   │   │   ├── faiss_index.py       # FAISS index controller (269 lines)
│   │   │   └── similarity_engine.py # Precedent search engine (260 lines)
│   │   │
│   │   ├── hybrid/                  # ── Hybrid Decision Engine ──
│   │   │   ├── __init__.py          # Package exports (25 lines)
│   │   │   ├── confidence_engine.py # Confidence scoring (100 lines)
│   │   │   ├── decision_engine.py   # ML + FAISS fusion (182 lines)
│   │   │   ├── reasoning_engine.py  # NL explanation generator (126 lines)
│   │   │   └── recommendation_engine.py # Master orchestrator (372 lines)
│   │   │
│   │   └── explainability/          # ── Explainable AI ──
│   │       ├── __init__.py
│   │       └── shap_explainer.py    # SHAP TreeExplainer (357 lines)
│   │
│   └── utils/                       # ── Utilities ──
│       ├── __init__.py              # robust_read_csv, robust_open (52 lines)
│       ├── config_manager.py        # Singleton YAML config loader (268 lines)
│       └── logger.py                # JSON logging & rotation (253 lines)
│
└── tests/                           # ══ TEST SUITE ══
    ├── __init__.py
    ├── conftest.py                  # Shared fixtures & singleton resets (89 lines)
    ├── unit/                        # 23 unit test files
    │   ├── test_catboost_trainer.py
    │   ├── test_cleaner.py
    │   ├── test_cli.py
    │   ├── test_config_manager.py
    │   ├── test_eda.py
    │   ├── test_embedding_registry.py
    │   ├── test_engineer.py
    │   ├── test_evaluator.py
    │   ├── test_feature_lineage.py
    │   ├── test_feature_registry.py
    │   ├── test_hybrid_engine.py
    │   ├── test_logger.py
    │   ├── test_model_registry.py
    │   ├── test_pipeline_contracts.py
    │   ├── test_readiness.py
    │   ├── test_semantic_similarity.py
    │   ├── test_shap_explainer.py
    │   ├── test_splitter.py
    │   ├── test_text_preprocessor.py
    │   ├── test_transformers_edge_cases.py
    │   ├── test_validation.py
    │   └── test_version_manager.py
    └── integration/
        └── __init__.py
```

---

## 4. End-to-End Pipeline Flow

The platform executes a **12-stage sequential pipeline** when `python main.py full-pipeline` is invoked:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    PHASE 1: DATA INTELLIGENCE                          │
│                                                                         │
│  Stage 1 ─► Data Validation (12 rules)                                 │
│              └─ DatasetValidator validates schema, timestamps, SLA      │
│                                                                         │
│  Stage 2 ─► ML Readiness Assessment                                    │
│              └─ MLReadinessEvaluator checks leakage, entropy, tokens    │
│                                                                         │
│  Stage 3 ─► Data Cleaning (8 steps)                                    │
│              └─ EnterpriseDataCleaner deduplicates, imputes, winsorizes │
│                                                                         │
│  Stage 4 ─► External Enrichment                                       │
│              └─ EnterpriseDataEnricher merges CMDB & shift data         │
│                                                                         │
│  Stage 5 ─► Feature Engineering                                        │
│              └─ FeatureEngineeringEngine creates temporal, cyclic,      │
│                 interaction, text-stat, frequency features               │
│                                                                         │
│  Stage 6 ─► NLP Text Preprocessing                                    │
│              └─ TextPreprocessor normalizes, lemmatizes, tokenizes      │
│                                                                         │
│  Stage 7 ─► Exploratory Data Analysis                                  │
│              └─ EnterpriseEDAEngine computes entropy, generates charts  │
│                                                                         │
│  Stage 8 ─► Train/Val/Test Splitting                                   │
│              └─ DatasetSplitter stratified split + zero-leakage verify  │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                    PHASE 2: MODEL TRAINING                              │
│                                                                         │
│  Stage 9 ─► CatBoost Classification Training                          │
│              └─ EnterpriseCatBoostTrainer trains assignment_group       │
│                 classifier with RandomizedSearchCV HPO                   │
│                                                                         │
│  Stage 10 ─► CatBoost Regression Training                             │
│               └─ EnterpriseCatBoostTrainer trains resolution_time_hours │
│                  regressor with log1p target transform                   │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                    PHASE 3: SEMANTIC INDEX                              │
│                                                                         │
│  Stage 11 ─► Embedding Generation + FAISS Index                       │
│               └─ SemanticEmbeddingGenerator creates TF-IDF+SVD vectors │
│               └─ FAISSVectorIndex builds FlatIP exact search index      │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                    PHASE 4: HYBRID INFERENCE                            │
│                                                                         │
│  Stage 12 ─► Hybrid Recommendation Demo                               │
│               └─ HybridRecommendationEngine fuses CatBoost + FAISS     │
│               └─ Outputs: Predicted Team, MTTR, Confidence, Precedents │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Data Flow Diagram

```
incidents.csv ──► Validator ──► Cleaner ──► Enricher ──► Engineer ──► TextPrep ──► EDA ──► Splitter
                                                                                              │
                                                                        ┌─────────────────────┤
                                                                        │                     │
                                                                   train.csv              val.csv
                                                                        │                     │
                                                          ┌─────────────┴──────────┐          │
                                                          │                        │          │
                                                    CatBoost               TF-IDF + SVD       │
                                                    Classifier              Embeddings        │
                                                    & Regressor                │               │
                                                          │              FAISS Index           │
                                                          │                    │               │
                                                          └────────┬───────────┘               │
                                                                   │                           │
                                                          HybridDecisionEngine                 │
                                                                   │                           │
                                                          HybridReasoningEngine                │
                                                                   │                           │
                                                          ┌────────▼────────┐                  │
                                                          │  FINAL OUTPUT   │    Evaluated on ──┘
                                                          │  • Team         │
                                                          │  • MTTR         │
                                                          │  • Confidence   │
                                                          │  • Precedents   │
                                                          │  • Reasoning    │
                                                          └─────────────────┘
```

---

## 5. Entry Points & Execution

### `main.py` (154 lines)
The root executable. Parses CLI arguments using `argparse` with 20 subcommands and dispatches to `EnterpriseCLI.run_command()`.

```bash
python main.py full-pipeline          # Run entire 12-stage pipeline
python main.py train --target assignment_group   # Train classifier only
python main.py recommend --text "Database timeout"  # Live inference
```

### `run_dashboard.py` (7 lines)
Launches the Streamlit web UI safely using `subprocess.run([sys.executable, '-m', 'streamlit', 'run', 'src/dashboard/app.py'])` to handle Windows PATH spaces.

```bash
python run_dashboard.py               # Opens browser dashboard
```

### `src/cli/main_cli.py` — `EnterpriseCLI` (849 lines)
The **central nervous system** of the entire platform. This single class:
- Creates all runtime directories on startup
- Self-heals missing artifacts (auto-triggers training if models are missing)
- Dispatches 19 CLI subcommands to their respective engines
- Provides a 24-option interactive terminal menu

**Key Methods:**
| Method | Purpose |
|---|---|
| `cmd_validate()` | Run 12 data quality rules |
| `cmd_readiness()` | ML feasibility diagnostic |
| `cmd_clean()` | 8-step data cleaning |
| `cmd_engineer()` | Feature generation |
| `cmd_split()` | Stratified train/val/test partitioning |
| `cmd_pipeline()` | 5-stage data processing pipeline |
| `cmd_train()` | CatBoost training with HPO |
| `cmd_evaluate()` | Model metrics & charts |
| `cmd_explain()` | SHAP feature attribution |
| `cmd_embed()` | Generate TF-IDF+SVD embeddings |
| `cmd_index()` | Build FAISS vector index |
| `cmd_similar()` | Semantic precedent search |
| `cmd_recommend()` | Full hybrid inference |
| `cmd_full_pipeline()` | Complete 12-stage pipeline |
| `cmd_clean_workspace()` | Reset all generated artifacts |

---

## 6. Configuration Layer

### `config/config.yaml` (139 lines)
Platform-wide settings including data paths, banking-specific incident distributions, report branding, and log rotation. Supports `${ENV_VAR:default}` interpolation.

### `config/model_config.yaml` (97 lines)
ML hyperparameters for:
- **CatBoost:** iterations=1000, depth=8, learning_rate=0.1
- **HPO:** RandomizedSearchCV with 5-fold CV, 30 iterations, `f1_weighted` scoring
- **Embeddings:** TF-IDF+SVD to 384 dimensions, L2 normalized
- **FAISS:** IndexFlatIP (exact inner product search)
- **Hybrid Engine:** CatBoost weight=0.6, Semantic weight=0.4, agreement bonus=0.1

### `config/logging.yaml` (97 lines)
Python `logging.config.dictConfig` spec with:
- Console handler (stdout)
- 4 rotating file handlers (`app.log`, `ml_training.log`, `api_access.log`, `error.log`)
- JSON formatter for machine-parseable log ingestion (ELK/Splunk)
- 10MB file size, 5 backup rotations

### `config/servicenow.yaml` (52 lines)
ServiceNow REST API integration config with connection settings, authentication, 19-column field mapping, and batch query parameters.

### `.env` (35 lines)
Environment variables for secrets (`SERVICENOW_URL`, `SERVICENOW_USER`, `SERVICENOW_PASS`), runtime paths, and port numbers.

### `src/utils/config_manager.py` — `ConfigManager` (268 lines)
**Thread-safe Singleton** that loads all YAML configs, resolves `${ENV_VAR:default}` patterns against `os.environ`, and provides dot-notation access (e.g., `config.get("ml.catboost.n_estimators")`).

---

## 7. Utility Layer (`src/utils/`)

### `src/utils/__init__.py` (52 lines) — Core I/O Functions
The **most critical utility file** in the project. Contains 4 functions that handle corporate Windows encoding issues:

| Function | Purpose |
|---|---|
| `_resolve_path(filepath)` | Resolves relative paths against project root |
| `robust_read_csv(filepath, **kwargs)` | Reads CSV trying encodings: `utf-8` → `utf-8-sig` → `cp1252` → `latin-1` |
| `robust_open(filepath, mode)` | Opens any text file with encoding fallback chain |
| `robust_json_load(filepath)` | Loads JSON via `robust_open` |

**Used by:** Every module in the project that reads files from disk.

### `src/utils/logger.py` (253 lines) — Enterprise Logging

| Class | Purpose |
|---|---|
| `JsonFormatter` | Formats log records as single-line JSON with UTC timestamps, exception info |
| `LoggerFactory` | Singleton factory that loads `config/logging.yaml`, creates rotating file handlers |

**Module-level function:** `get_logger(name)` — Convenience wrapper used by every module.

---

## 8. Data Governance Layer (`src/data/`)

### `src/data/feature_registry.py` (670 lines) — **The Single Source of Truth**

**Classes:**
- **`FeatureDefinition`** — Dataclass with 22 governance dimensions per feature:
  `business_name`, `technical_name`, `data_type`, `nullable`, `cardinality`, `missing_percentage`, `business_meaning`, `ml_importance`, `target_leakage_classification`, `encoding_strategy`, `imputation_strategy`, `scaling_strategy`, `feature_engineering_rules`, `catboost_usage`, `embedding_usage`, `faiss_metadata_usage`, `dashboard_usage`, `api_exposure`, `future_rag_usage`, `explainability_usage`, `required_or_optional`, `deprecated_status`

- **`FeatureRegistry`** — Singleton registry of 38 raw + 11 derived features. Prevents hardcoded column names anywhere in the codebase.

**Key Methods:**
| Method | Purpose |
|---|---|
| `get_catboost_predictors(target)` | Returns safe predictor list (no leakage columns) |
| `get_embedding_features()` | Returns text columns for NLP vectorization |
| `get_faiss_metadata_features()` | Returns structural metadata for FAISS |
| `resolve_business_name(tech_name)` | Translates `freq__category` → `Category (Frequency)` for SHAP plots |
| `get_features_by_leakage(tier)` | Returns `safe`, `warning`, or `blocked` features |

### `src/data/feature_lineage.py` (273 lines) — Feature Ancestry Graph

Tracks parent→child derivation relationships with exact mathematical formulas:
```
opened_at → opened_at_hour → opened_at_hour_sin [formula: sin(2π × hour / 24)]
priority + business_impact → priority_x_business_impact [formula: priority × impact]
```

### `src/data/validation.py` (452 lines) — 12-Rule Quality Engine

| Rule ID | Rule Name | What It Checks |
|---|---|---|
| CHK-01 | Missing Values | Nulls in critical fields (`number`, `priority`, `category`) |
| CHK-02 | Duplicate Incidents | Unique ticket `number` keys |
| CHK-03 | Invalid Timestamps | `resolved_at >= opened_at`, `closed_at >= resolved_at` |
| CHK-04 | Invalid Categories | Non-null, non-blank categories |
| CHK-05 | Invalid Assignment Groups | Non-null, non-empty groups |
| CHK-06 | Invalid Priorities | Priority in range [1, 5] |
| CHK-07 | SLA Inconsistencies | `made_sla` flag vs actual resolution time |
| CHK-08 | Resolution Time | Non-negative, agrees with timestamp delta |
| CHK-09 | Invalid CMDB References | Populated `cmdb_ci` |
| CHK-10 | Invalid Business Services | Populated `business_service` |
| CHK-11 | Empty Descriptions | Non-empty `description` |
| CHK-12 | Empty Short Descriptions | Non-empty `short_description` |

### `src/data/readiness.py` (375 lines) — ML Diagnostic Evaluator

Pre-training audit computing:
- Target leakage detection (blocks `close_notes`, `resolved_at`, `made_sla`)
- Shannon Entropy & Gini Impurity for class imbalance
- Text token capacity vs 256-token budget
- Pearson correlation matrix
- Actionable preprocessing recommendations

### `src/data/quality_gate.py` (270 lines) — 6-Gate Certification

| Gate | What It Certifies |
|---|---|
| Gate 1 | All 4 YAML config files exist and parse |
| Gate 2 | Automation batch scripts are present |
| Gate 3 | 10 mandatory documentation files exist |
| Gate 4 | Dataset contains all 38 enterprise columns |
| Gate 5 | 12 data quality rules pass (DatasetValidator) |
| Gate 6 | ML readiness certified (MLReadinessEvaluator) |

### `src/data/pipeline_contracts.py` (106 lines) — Schema Adapter
Provides verified feature subsets to downstream modules, preventing accidental leakage column inclusion.

### `src/data/version_manager.py` (215 lines) — Dataset Versioning
Manages immutable dataset snapshots under `datasets/synthetic/v{N}/` with `metadata.json` manifests and `version_history.json` catalog.

---

## 9. Preprocessing Layer (`src/preprocessing/`)

### `src/preprocessing/cleaner.py` (454 lines) — `EnterpriseDataCleaner`

Executes an **8-step sequential cleaning pipeline:**

| Step | Operation | Details |
|---|---|---|
| 1 | Duplicate Removal | Keeps latest by `opened_at`, deduplicates on `number` |
| 2 | Business Rule Validation | Standardizes `priority` (1-5), `urgency` (1-3), `business_impact` (1-3) from mixed text/numeric |
| 3 | Schema & Type Enforcement | Coerces datetimes, integers, floats, booleans via FeatureRegistry |
| 4 | Missing Value Imputation | Text→"Not Provided", numeric→median/zero, categorical→mode/"Unknown" |
| 5 | Category Validation | Maps null/blank categories to "Unknown" |
| 6 | Timestamp Correction | Fixes `resolved_at < opened_at` (adds 4h) and `closed_at < resolved_at` (adds 24h) |
| 7 | Outlier Winsorization | Caps `reassignment_count` at 99th%/15, `reopen_count` at 99th%/8 |
| 8 | String Normalization | Trims whitespace across all string columns |

**Outputs:** `reports/cleaning_report.json` + `.md`

### `src/preprocessing/engineer.py` (470 lines) — `FeatureEngineeringEngine`

Generates **7 categories of engineered features:**

| Category | Features Generated | Mathematical Formula |
|---|---|---|
| **Temporal** | `opened_at_hour`, `opened_at_dayofweek`, `opened_at_month`, `is_weekend` | Extracted from `opened_at` datetime |
| **Cyclic** | `opened_at_hour_sin/cos`, `opened_at_dayofweek_sin/cos` | $\sin(2\pi x / T)$, $\cos(2\pi x / T)$ |
| **Business Hours** | `is_business_hours`, `is_holiday` | Mon-Fri 08:00-18:00; US Federal banking holidays |
| **Interaction** | `priority_x_business_impact`, `category_assignment_interaction` | $p \times i$; string concatenation |
| **Resolution** | `resolution_time_hours` | $(resolved\_at - opened\_at)$ in hours |
| **Text Stats** | `short_description_word_count/char_count`, `description_word_count/char_count` | `len(str.split())`, `len(str)` |
| **Frequency** | `assignment_group_freq`, `business_service_freq`, `vendor_freq`, etc. | `value_counts(normalize=True)` |

Every new feature is automatically registered in `FeatureRegistry` and its derivation formula logged in `FeatureLineageTracker`.

### `src/preprocessing/text_preprocessor.py` (273 lines) — `TextPreprocessor`

**6-step NLP normalization pipeline:**

| Step | Operation |
|---|---|
| 1 | Unicode NFKC normalization + lowercasing |
| 2 | Strip HTML/XML tags and email headers (`From:`, `To:`, `Subject:`) |
| 3 | Replace error boilerplate (`[system error code: 0x...]` → `system_error`) |
| 4 | Remove punctuation while preserving IT symbols (`-`, `_`, `/`, `.`) |
| 5 | Filter stopwords while protecting 40+ IT keywords (`server`, `down`, `error`, `timeout`, `firewall`, `vpn`, `dns`, etc.) |
| 6 | IT domain lemmatization (`failures→failure`, `servers→server`, `crashes→crash`) |

**Token budget:** Estimates BPE tokens as $\lceil \text{word\_count} \times 1.3 \rceil$ and flags sequences exceeding 256 tokens.

### `src/preprocessing/enricher.py` (60 lines) — `EnterpriseDataEnricher`
Optional external data enrichment via left-merge with `data/raw/cmdb.csv` (CMDB Configuration Items) and `data/raw/shift_schedules.csv` (IT Shift Schedules). Gracefully degrades if files are missing.

### `src/preprocessing/splitter.py` (287 lines) — `DatasetSplitter`

**3 splitting strategies:**
| Strategy | Method |
|---|---|
| **Stratified** | Preserves class ratios using `sklearn.model_selection.train_test_split` with `stratify=y`. Auto-groups rare classes (< 3 samples) into "Rare / Other". |
| **Time-Based** | Chronological sort by `opened_at`, sequential slice (70/15/15). |
| **Random** | Standard two-stage random split. |

**Zero-Leakage Verification:** Performs set intersection checks on ticket `number` across Train∩Val, Train∩Test, Val∩Test. Raises `ValueError` if any overlap detected.

### `src/preprocessing/eda.py` (597 lines) — `EnterpriseEDAEngine`

Automated statistical analysis computing:
- Missing value percentages per column
- Numerical statistics (mean, std, skew, kurtosis, IQR, outlier counts)
- Categorical Shannon Entropy: $H = -\sum p \log_2 p$
- Categorical Gini Impurity: $G = 1 - \sum p^2$
- Datetime hourly/weekday arrival distributions
- Text character/word/token length distributions
- Pearson & Spearman correlation matrices
- Target leakage audit

**Generates 5 diagnostic charts:**
1. `01_category_distribution.png` — Top 10 categories barplot
2. `02_priority_vs_sla.png` — SLA compliance by priority
3. `03_hourly_arrival.png` — 24-hour arrival frequency
4. `04_numerical_correlation.png` — Correlation heatmap
5. `05_text_word_counts.png` — Word count distribution

---

## 10. Machine Learning Layer (`src/ml/`)

### `src/ml/catboost/trainer.py` (536 lines) — `EnterpriseCatBoostTrainer`

**The training engine.** Builds complete scikit-learn pipelines:

```
Pipeline:
  1. EnterpriseFeatureExtractor (interaction terms, cyclic features, combined text)
  2. ColumnTransformer:
     ├── FrequencyEncoder (high-cardinality categoricals)
     ├── OneHotEncoder (low-cardinality categoricals)
     ├── SimpleImputer(strategy="median") (numericals)
     └── TfidfVectorizer passthrough (text)
  3. CatBoostClassifier / CatBoostRegressor
```

**Hyperparameter Optimization:**
- Uses `sklearn.model_selection.RandomizedSearchCV`
- 5-fold cross-validation, 30 iterations
- Scoring: `f1_weighted` (classifier), `neg_mean_absolute_error` (regressor)
- `n_jobs=1` (avoids thread contention with CatBoost's internal multithreading)
- Search space: `depth=[4-10]`, `learning_rate=[0.01-0.3]`, `iterations=[300-2000]`, `l2_leaf_reg=[1-10]`

**Regression target transform:** `np.log1p(y)` during training, `np.expm1(pred)` during inference to handle right-skewed resolution times.

### `src/ml/catboost/evaluator.py` (389 lines) — `ModelEvaluator`

Evaluates trained models on test data:
- **Classification:** Top-1/Top-3 accuracy, weighted/macro precision/recall/F1, multi-class ROC-AUC
- **Regression:** MAE, MSE, RMSE, R²
- **Charts:** Confusion matrix heatmap, ROC curves, feature importance bar chart
- **Reports:** `classification_report.json/.md`, `regression_report.json/.md`

### `src/ml/catboost/transformers.py` (243 lines) — Custom Sklearn Transformers

| Transformer | Purpose |
|---|---|
| `DataFrameSelector` | Selects column subsets, fills missing with safe defaults |
| `EnterpriseFeatureExtractor` | Generates `combined_text`, `priority_x_business_impact`, cyclic sin/cos |
| `FrequencyEncoder` | Maps categories to normalized frequency probabilities |
| `SmoothedTargetEncoder` | Smoothed out-of-fold target encoding: $(count \times mean + smoothing \times global) / (count + smoothing)$ |

### `src/ml/semantic/embedding_generator.py` (266 lines) — `SemanticEmbeddingGenerator`

Converts text into 384-dimensional dense vectors:
1. Constructs composite semantic string: `[Category: X] [Service: Y] [Priority: Z] Short Description. Description`
2. Fits `TfidfVectorizer` (sparse matrix)
3. Applies `TruncatedSVD(n_components=384)` for dimensionality reduction
4. L2-normalizes all vectors for cosine similarity compatibility
5. Saves pipeline as `tfidf_svd_pipeline.pkl`, vectors as `.npy`, metadata as `.csv`

### `src/ml/semantic/faiss_index.py` (269 lines) — `FAISSVectorIndex`

FAISS vector index controller supporting:

| Index Type | Description | Use Case |
|---|---|---|
| `IndexFlatIP` | Exact inner product search | Default for < 10,000 vectors (100% recall) |
| `IndexFlatL2` | Exact L2 distance search | Alternative distance metric |
| `IndexIVFFlat` | Inverted file index with clustering | Future scalability for 1M+ vectors |

**Current config:** `IndexFlatIP` — Performs exhaustive exact search checking every single vector. Optimal for the 6,000-ticket dataset.

### `src/ml/semantic/similarity_engine.py` (260 lines) — `SemanticSimilarityEngine`

High-level orchestrator connecting `SemanticEmbeddingGenerator` and `FAISSVectorIndex`:
1. Embeds the entire training DataFrame into vectors
2. Builds and saves the FAISS index
3. For queries: embeds the query text, searches Top-K nearest neighbors, returns structured results with similarity scores
4. Computes routing consensus among retrieved precedents

### `src/ml/model_registry.py` (248 lines) — `ModelRegistry`

Singleton registry tracking trained model artifacts:
- SHA256 cryptographic checksums for integrity verification
- Feature compliance checking against `FeatureRegistry` (rejects blocked leakage features)
- Model versioning (`:latest`, `:v1`, etc.)
- Exports `model_registry.json` and `model_registry.md`

### `src/ml/embedding_registry.py` (188 lines) — `EmbeddingRegistry`

Similar to ModelRegistry but for FAISS vector indexes:
- Tracks index name, dimension, vector count, distance metric, FAISS version
- SHA256 checksums for index file integrity
- Exports `embedding_registry.json` and `embedding_registry.md`

---

## 11. Hybrid Intelligence Engine (`src/ml/hybrid/`)

The crown jewel of the architecture. Four specialized sub-engines work together:

### `recommendation_engine.py` (372 lines) — `HybridRecommendationEngine`

**The Master Orchestrator.** Executes a 6-step workflow:

| Step | Engine | Action |
|---|---|---|
| 1 | Input Parser | Parse JSON/dict/free-text into normalized ticket dict |
| 2 | CatBoost | Run classifier (predict group + probabilities) and regressor (predict MTTR) |
| 3 | FAISS | Search Top-K semantically similar historical tickets |
| 4 | Decision Engine | Fuse ML prediction with historical consensus |
| 5 | Reasoning Engine | Generate natural language explanation |
| 6 | Reporter | Export JSON/MD/CSV reports |

### `decision_engine.py` (182 lines) — `HybridDecisionEngine`

Fusion logic:
1. Extract CatBoost predicted group and confidence
2. Compute semantic consensus (mode of Top-K groups, consensus percentage)
3. Check agreement: `CatBoost prediction == FAISS consensus?`
4. **If agreed:** Use CatBoost prediction, apply agreement bonus (+0.1)
5. **If CatBoost dominant** (`rf_confidence > 0.7`): Use CatBoost, apply disagreement penalty (-0.05)
6. **If FAISS dominant:** Override with semantic consensus
7. Blend MTTR: `0.5 × CatBoost_MTTR + 0.5 × FAISS_median_MTTR`
8. Calculate historical success rate from precedents

### `confidence_engine.py` (100 lines) — `HybridConfidenceEngine`

Computes fused confidence score:

$$\text{score} = w_{rf} \times \text{rf\_conf} + w_{sem} \times \text{sem\_conf} + \text{bonus/penalty}$$

Where: $w_{rf}=0.6$, $w_{sem}=0.4$, agreement\_bonus=+0.1, disagreement\_penalty=-0.05

**Confidence Tiers:**
| Score Range | Tier |
|---|---|
| ≥ 0.88 | Very High |
| ≥ 0.75 | High |
| ≥ 0.60 | Moderate |
| ≥ 0.45 | Low |
| < 0.45 | Review Required |

### `reasoning_engine.py` (126 lines) — `HybridReasoningEngine`

Generates **deterministic natural language explanations** (zero LLMs/GenAI):
- Executive summary tailored to agreement/disagreement scenario
- Bullet breakdown (ML prediction, precedent consensus, fused MTTR, success rate)
- Historical evidence table with similarity scores

---

## 12. Explainability Layer (`src/ml/explainability/`)

### `shap_explainer.py` (357 lines) — `SHAPIntelligenceExplainer`

Uses **SHAP (SHapley Additive exPlanations)** for game-theoretic feature attribution:

- **Global Explanation:** Computes SHAP values across test sample, generates:
  - `shap_summary.png` — Beeswarm plot showing feature impact distributions
  - `shap_bar.png` — Mean absolute SHAP importance bar chart

- **Local Explanation:** Per-prediction SHAP decomposition:
  - Top 5 contributing features per prediction
  - `shap_waterfall_sample.png` — Waterfall contribution plot
  - `shap_decision_sample.png` — Decision plot

- **CatBoost Native SHAP:** Uses CatBoost's internal `get_feature_importance(type="ShapValues")` for faster, more accurate tree-native SHAP computation.

---

## 13. Dashboard UI (`src/dashboard/`)

### `app.py` (118 lines) — Streamlit Web Interface

**Layout:**
- Column 1: Core fields (Short Description, Full Description, Category, Subcategory, CMDB CI)
- Column 2: Severity fields (Priority, Business Impact, Severity) + Custom corporate fields (`u_caused_by`, `u_development_release_id`, `u_vendor_ticket_ref`, `u_describe_customer_impact`)

**On "Predict" click:**
1. Formats input into ticket dictionary
2. Calls `HybridRecommendationEngine.recommend(ticket_dict, top_k=5)`
3. Displays metric cards (Predicted Group, Confidence Score/Tier, Estimated MTTR)
4. Shows AI Reasoning summary
5. Renders Top-5 historical precedents table

**Caching:** Uses `@st.cache_resource` to load models once and reuse across UI refreshes.

---

## 14. Testing Infrastructure

### Framework: **pytest** with 23 unit test files

**Configuration** (`pyproject.toml`):
- Test discovery: `tests/` directory
- Coverage threshold: 80%
- Markers: `unit`, `integration`, `slow`, `api`, `dashboard`, `model`, `smoke`

**`tests/conftest.py` (89 lines):**
- `reset_singletons` fixture (autouse): Resets `ConfigManager` and `LoggerFactory` singletons before/after every test
- `temp_workspace` fixture: Creates isolated temp directory with config files, changes CWD, cleans up after

**Test Coverage:**
| Test File | Module Tested |
|---|---|
| `test_catboost_trainer.py` | CatBoost training pipeline |
| `test_cleaner.py` | 8-step data cleaning |
| `test_cli.py` | CLI command dispatch |
| `test_config_manager.py` | YAML config loading |
| `test_eda.py` | EDA metric calculations |
| `test_embedding_registry.py` | Vector index registration |
| `test_engineer.py` | Feature engineering formulas |
| `test_evaluator.py` | Model evaluation metrics |
| `test_feature_lineage.py` | Derivation graph |
| `test_feature_registry.py` | Feature governance rules |
| `test_hybrid_engine.py` | Hybrid decision fusion |
| `test_logger.py` | JSON logging |
| `test_model_registry.py` | Model artifact tracking |
| `test_pipeline_contracts.py` | Schema contract validation |
| `test_readiness.py` | ML readiness diagnostics |
| `test_semantic_similarity.py` | FAISS search |
| `test_shap_explainer.py` | SHAP attribution |
| `test_splitter.py` | Train/val/test splitting |
| `test_text_preprocessor.py` | NLP normalization |
| `test_transformers_edge_cases.py` | Custom transformer edge cases |
| `test_validation.py` | 12 data quality rules |
| `test_version_manager.py` | Dataset versioning |

---

## 15. Key Algorithms & Mathematics

### Cyclic Temporal Encoding
Converts periodic features (hour, day-of-week) into continuous sin/cos pairs to preserve cyclical proximity (hour 23 is close to hour 0):

$$\text{sin\_feature} = \sin\left(\frac{2\pi \times x}{T}\right), \quad \text{cos\_feature} = \cos\left(\frac{2\pi \times x}{T}\right)$$

Where $T = 24$ for hours, $T = 7$ for days.

### TF-IDF + Truncated SVD (Latent Semantic Analysis)
1. **TF-IDF:** Converts text corpus into sparse term-frequency inverse-document-frequency matrix
2. **TruncatedSVD:** Projects sparse matrix into dense 384-dimensional space via Singular Value Decomposition
3. **L2 Normalization:** Unit-length vectors enable cosine similarity via inner product

### FAISS Inner Product Search
For L2-normalized vectors: $\text{cosine\_similarity}(a, b) = a \cdot b = \text{inner\_product}(a, b)$

`IndexFlatIP` computes exact inner products against all vectors — $O(N)$ but guarantees 100% recall.

### CatBoost Gradient Boosting
Ordered boosting algorithm with:
- Native categorical feature handling (no manual one-hot encoding needed)
- Symmetric decision trees for faster inference
- Built-in L2 regularization (`l2_leaf_reg`)
- Log-transform target for regression: $y' = \log(1 + y)$, prediction: $\hat{y} = e^{\hat{y'}} - 1$

### Hybrid Confidence Fusion
$$C_{\text{fused}} = \text{clamp}\left(w_{rf} \cdot C_{rf} + w_{sem} \cdot C_{sem} + \delta, \; 0.0001, \; 1.0\right)$$

Where $\delta = +0.1$ (agreement) or $\delta = -0.05$ (disagreement).

### Shannon Entropy (Class Imbalance)
$$H = -\sum_{i=1}^{n} p_i \log_2 p_i$$

Higher entropy → more balanced classes → better for ML training.

### Gini Impurity
$$G = 1 - \sum_{i=1}^{n} p_i^2$$

Used in EDA to measure class distribution purity.

### SHAP (SHapley Additive exPlanations)
Based on cooperative game theory. For each prediction, SHAP computes the marginal contribution of each feature:

$$\phi_i = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|! \; (|N|-|S|-1)!}{|N|!} \left[ f(S \cup \{i\}) - f(S) \right]$$

Where $\phi_i$ is the SHAP value for feature $i$, measuring its contribution to the prediction.

---

## 16. File-by-File Reference Matrix

| # | File | Lines | Primary Class | Layer | Core Responsibility |
|:---:|---|:---:|---|---|---|
| 1 | `main.py` | 154 | — | Entry | CLI argument parsing & dispatch |
| 2 | `run_dashboard.py` | 7 | — | Entry | Streamlit launcher (subprocess) |
| 3 | `src/utils/__init__.py` | 52 | — | Utility | `robust_read_csv`, `robust_open`, `robust_json_load` |
| 4 | `src/utils/config_manager.py` | 268 | `ConfigManager` | Utility | Thread-safe singleton YAML config loader |
| 5 | `src/utils/logger.py` | 253 | `JsonFormatter`, `LoggerFactory` | Utility | JSON logging with rotating file handlers |
| 6 | `src/data/feature_registry.py` | 670 | `FeatureDefinition`, `FeatureRegistry` | Governance | 22-dimension feature catalog (49 features) |
| 7 | `src/data/feature_lineage.py` | 273 | `LineageEdge`, `FeatureLineageTracker` | Governance | Parent→child derivation graph |
| 8 | `src/data/pipeline_contracts.py` | 106 | `PipelineContractValidator` | Governance | Schema adapter & leakage gatekeeper |
| 9 | `src/data/quality_gate.py` | 270 | `QualityGateRunner` | Governance | 6-gate certification engine |
| 10 | `src/data/readiness.py` | 375 | `MLReadinessEvaluator` | Governance | Pre-training ML diagnostic |
| 11 | `src/data/validation.py` | 452 | `DatasetValidator` | Governance | 12-rule data quality engine |
| 12 | `src/data/version_manager.py` | 215 | `DatasetVersionManager` | Governance | Immutable dataset versioning |
| 13 | `src/preprocessing/cleaner.py` | 454 | `EnterpriseDataCleaner` | Preprocessing | 8-step cleaning pipeline |
| 14 | `src/preprocessing/eda.py` | 597 | `EnterpriseEDAEngine` | Preprocessing | Automated EDA with 5 charts |
| 15 | `src/preprocessing/engineer.py` | 470 | `FeatureEngineeringEngine` | Preprocessing | 7-category feature generation |
| 16 | `src/preprocessing/enricher.py` | 60 | `EnterpriseDataEnricher` | Preprocessing | External CMDB/shift data merge |
| 17 | `src/preprocessing/splitter.py` | 287 | `DatasetSplitter` | Preprocessing | Stratified splitting + zero-leakage verify |
| 18 | `src/preprocessing/text_preprocessor.py` | 273 | `TextPreprocessor` | Preprocessing | 6-step NLP normalization |
| 19 | `src/ml/model_registry.py` | 248 | `ModelRegistry` | ML Registry | SHA256 model artifact tracking |
| 20 | `src/ml/embedding_registry.py` | 188 | `EmbeddingRegistry` | ML Registry | FAISS index artifact tracking |
| 21 | `src/ml/catboost/trainer.py` | 536 | `EnterpriseCatBoostTrainer` | ML Training | CatBoost + HPO training engine |
| 22 | `src/ml/catboost/evaluator.py` | 389 | `ModelEvaluator` | ML Evaluation | Metrics, ROC curves, confusion matrices |
| 23 | `src/ml/catboost/transformers.py` | 243 | `DataFrameSelector`, `EnterpriseFeatureExtractor`, `FrequencyEncoder`, `SmoothedTargetEncoder` | ML Pipeline | Custom sklearn transformers |
| 24 | `src/ml/semantic/embedding_generator.py` | 266 | `SemanticEmbeddingGenerator` | Semantic | TF-IDF + SVD → 384-D vectors |
| 25 | `src/ml/semantic/faiss_index.py` | 269 | `FAISSVectorIndex` | Semantic | FAISS index build/search/persist |
| 26 | `src/ml/semantic/similarity_engine.py` | 260 | `SemanticSimilarityEngine` | Semantic | High-level precedent search |
| 27 | `src/ml/hybrid/confidence_engine.py` | 100 | `HybridConfidenceEngine` | Hybrid | Weighted confidence fusion |
| 28 | `src/ml/hybrid/decision_engine.py` | 182 | `HybridDecisionEngine` | Hybrid | ML + FAISS agreement/override logic |
| 29 | `src/ml/hybrid/reasoning_engine.py` | 126 | `HybridReasoningEngine` | Hybrid | Deterministic NL explanation |
| 30 | `src/ml/hybrid/recommendation_engine.py` | 372 | `HybridRecommendationEngine` | Hybrid | Master 6-step orchestrator |
| 31 | `src/ml/explainability/shap_explainer.py` | 357 | `SHAPIntelligenceExplainer` | XAI | SHAP global + local attribution |
| 32 | `src/dashboard/app.py` | 118 | — | UI | Streamlit web dashboard |
| 33 | `src/cli/main_cli.py` | 849 | `EnterpriseCLI` | CLI | 24-option command control plane |
| 34 | `config/config.yaml` | 139 | — | Config | Platform settings |
| 35 | `config/model_config.yaml` | 97 | — | Config | ML hyperparameters |
| 36 | `config/logging.yaml` | 97 | — | Config | Logging configuration |
| 37 | `config/servicenow.yaml` | 52 | — | Config | ServiceNow API mapping |
| 38 | `tests/conftest.py` | 89 | — | Testing | Shared fixtures & singleton resets |
| — | `tests/unit/*.py` (23 files) | ~4,000+ | — | Testing | Comprehensive unit test suite |

---

> **Total Production Code:** ~7,500 lines across 33 Python source files
> **Total Test Code:** ~4,000+ lines across 23 unit test files
> **Total Configuration:** ~385 lines across 4 YAML files + `.env`
