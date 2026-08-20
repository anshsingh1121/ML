# Script Flow Guide
## How Every File Connects — Simple Logical Flow

> This document covers **only** the 40 core scripts on your corporate system.
> No test files, no generated artifacts — just the source code you copy-pasted.

---

## HOW TO READ THIS GUIDE

```
File A
  └──► File B        means "File A calls/uses File B"
  │    └──► File C   means "File B then calls File C"
```

---

## 1. BOOT SEQUENCE — What Happens When You Run the Platform

```
You type: python main.py full-pipeline
                │
                ▼
┌─ main.py ──────────────────────────────────────────────────────┐
│  • Parses your CLI command (e.g. "full-pipeline", "train")     │
│  • Imports EnterpriseCLI from src/cli/main_cli.py              │
│  • Calls cli.run_command(args) and exits                       │
└──────────────────────┬─────────────────────────────────────────┘
                       ▼
┌─ src/cli/main_cli.py ─────────────────────────────────────────┐
│  EnterpriseCLI.__init__() runs FIRST:                          │
│    1. Loads config ──► src/utils/config_manager.py             │
│       └──► Reads config/config.yaml                            │
│       └──► Reads config/model_config.yaml                      │
│       └──► Reads config/servicenow.yaml                        │
│       └──► Reads .env for secrets (SERVICENOW_URL, etc.)       │
│                                                                │
│    2. Starts logging ──► src/utils/logger.py                   │
│       └──► Reads config/logging.yaml                           │
│       └──► Creates logs/app.log, ml_training.log, error.log    │
│                                                                │
│    3. Loads Feature Registry ──► src/data/feature_registry.py  │
│       └──► Registers 38 raw + 11 engineered feature specs      │
│                                                                │
│    4. Loads Model Registry ──► src/ml/model_registry.py        │
│       └──► Reads models/model_registry.json (if exists)        │
│                                                                │
│    5. Creates empty folders if missing:                         │
│       data/raw/, data/processed/, models/, reports/,           │
│       indexes/, logs/                                          │
│                                                                │
│  Then dispatches your command to the right stage ───────►      │
└────────────────────────────────────────────────────────────────┘
```

---

## 2. THE 12-STAGE PIPELINE — Exact Script Execution Order

When you run `python main.py full-pipeline`, `main_cli.py` calls these stages one after another:

---

### STAGE 1 → `src/data/validation.py`
**What it does:** Checks raw data quality (12 rules)
```
main_cli.py calls DatasetValidator.validate_dataset()
  │
  ├── Reads data/raw/incidents.csv
  │     └── via src/utils/__init__.py → robust_read_csv()
  │
  ├── Runs 12 checks: nulls, duplicates, timestamps, priorities, SLA, categories
  │
  └── Writes reports/validation_report.json + .md
```

---

### STAGE 2 → `src/data/readiness.py`
**What it does:** Checks if data is suitable for ML
```
main_cli.py calls MLReadinessEvaluator.evaluate_dataset()
  │
  ├── Detects target leakage (flags close_notes, resolved_at as blocked)
  ├── Computes class imbalance (Shannon Entropy, Gini Impurity)
  ├── Checks text token lengths vs 256-token budget
  ├── Computes correlation matrix
  │
  └── Writes reports/ml_readiness_report.json + .md
```

---

### STAGE 3 → `src/preprocessing/cleaner.py`
**What it does:** Cleans and fixes the raw data (8 steps)
```
main_cli.py calls EnterpriseDataCleaner.clean_dataset()
  │
  ├── Queries src/data/feature_registry.py for column types & imputation rules
  │
  ├── Step 1: Remove duplicates (keep latest per ticket number)
  ├── Step 2: Standardize priority/urgency/impact from text to integers
  ├── Step 3: Enforce data types (datetime, int, float, boolean)
  ├── Step 4: Fill missing values (text→"Not Provided", numeric→median)
  ├── Step 5: Fix blank categories → "Unknown"
  ├── Step 6: Fix broken timestamps (resolved_at < opened_at → add 4h)
  ├── Step 7: Cap outliers (reassignment_count capped at 15)
  ├── Step 8: Trim whitespace from all strings
  │
  └── Writes reports/cleaning_report.json + .md
```

---

### STAGE 4 → `src/preprocessing/enricher.py`
**What it does:** Merges external data (if available)
```
main_cli.py calls EnterpriseDataEnricher.enrich_dataset()
  │
  ├── Looks for data/raw/cmdb.csv → merges on "category" (if file exists)
  ├── Looks for data/raw/shift_schedules.csv → merges on "assignment_group"
  │
  └── Skips gracefully if files are missing (no crash)
```

---

### STAGE 5 → `src/preprocessing/engineer.py`
**What it does:** Creates new ML features from existing columns
```
main_cli.py calls FeatureEngineeringEngine.engineer_features()
  │
  ├── Creates temporal features: hour, dayofweek, month, is_weekend
  ├── Creates cyclic features: hour_sin, hour_cos (sin/cos encoding)
  ├── Creates business features: is_business_hours, is_holiday
  ├── Creates interaction: priority × business_impact
  ├── Creates resolution_time_hours: (resolved_at - opened_at) in hours
  ├── Creates text stats: word_count, char_count
  ├── Creates frequency encodings: assignment_group_freq, caller_freq
  │
  ├── For each new feature:
  │     ├──► src/data/feature_registry.py → registers the new feature
  │     └──► src/data/feature_lineage.py → logs parent→child formula
  │
  ├── Saves data/processed/master_engineered_incidents.csv
  └── Writes reports/feature_engineering_report.json + .md
```

---

### STAGE 6 → `src/preprocessing/text_preprocessor.py`
**What it does:** Cleans text fields for NLP
```
main_cli.py calls TextPreprocessor.preprocess_dataset()
  │
  ├── Queries src/data/pipeline_contracts.py for authorized text columns
  │
  ├── For short_description, description, close_notes:
  │     Step 1: Unicode normalize + lowercase
  │     Step 2: Strip HTML tags + email headers
  │     Step 3: Replace error codes → "system_error"
  │     Step 4: Remove punctuation (keep -, _, /, .)
  │     Step 5: Remove stopwords (but protect IT words: server, timeout, dns)
  │     Step 6: Lemmatize (failures→failure, servers→server)
  │
  ├── Creates: short_description_clean, description_clean, close_notes_clean
  └── Writes reports/text_preprocessing_report.json + .md
```

---

### STAGE 7 → `src/preprocessing/eda.py`
**What it does:** Automated statistical analysis
```
main_cli.py calls EnterpriseEDAEngine.analyze_dataset()
  │
  ├── Queries src/data/feature_registry.py to classify columns by type
  ├── Computes numerical stats (mean, std, skew, outliers)
  ├── Computes categorical entropy + Gini impurity
  ├── Computes text length distributions
  ├── Computes correlation matrices
  ├── Checks for target leakage
  │
  ├── Generates 5 PNG charts → reports/figures/
  └── Writes reports/eda_report.json + .md + .html
```

---

### STAGE 8 → `src/preprocessing/splitter.py`
**What it does:** Splits data into Train / Validation / Test
```
main_cli.py calls DatasetSplitter.split_dataset()
  │
  ├── Strategy: stratified split preserving class ratios (70/15/15)
  ├── Groups rare classes (<3 samples) into "Rare / Other"
  ├── Verifies ZERO leakage: Train ∩ Test = ∅ (raises error if overlap)
  │
  ├── Saves: data/processed/train.csv
  ├── Saves: data/processed/val.csv
  ├── Saves: data/processed/test.csv
  └── Writes reports/split_report.json + .md
```

---

### STAGE 9 → `src/ml/catboost/trainer.py` (Classifier)
**What it does:** Trains the Assignment Group prediction model
```
main_cli.py calls EnterpriseCatBoostTrainer.train_classifier()
  │
  ├── Reads data/processed/train.csv + val.csv
  │     └── via src/utils/__init__.py → robust_read_csv()
  │
  ├── Queries src/data/feature_registry.py → get safe predictor list
  ├── Verifies NO target leakage (rejects close_notes, resolved_at)
  │
  ├── Builds scikit-learn Pipeline:
  │     └── src/ml/catboost/transformers.py provides:
  │           ├── EnterpriseFeatureExtractor (combined_text, interactions, sin/cos)
  │           ├── FrequencyEncoder (high-cardinality categoricals)
  │           ├── OneHotEncoder (low-cardinality categoricals)
  │           ├── SimpleImputer (median for numericals)
  │           ├── TfidfVectorizer (text)
  │           └── CatBoostClassifier (final estimator)
  │
  ├── Runs RandomizedSearchCV (5-fold CV, 30 iterations, f1_weighted)
  ├── Evaluates on val.csv → accuracy, weighted F1
  │
  ├── Saves: models/catboost_assignment_group.pkl
  └──► src/ml/model_registry.py → registers model with SHA256 checksum
```

---

### STAGE 10 → `src/ml/catboost/trainer.py` (Regressor)
**What it does:** Trains the Resolution Time (MTTR) prediction model
```
main_cli.py calls EnterpriseCatBoostTrainer.train_regressor()
  │
  ├── Same pipeline as Stage 9 but:
  │     Target = resolution_time_hours (not assignment_group)
  │     Applies log1p(y) transform during training
  │     Applies expm1(prediction) during evaluation
  │     Excludes assignment_group from predictors
  │
  ├── Saves: models/catboost_resolution_time_hours.pkl
  └──► src/ml/model_registry.py → registers model with SHA256 checksum
```

---

### STAGE 11 → Semantic Embeddings + FAISS Index
**What it does:** Converts all tickets to vectors and builds search index
```
main_cli.py calls SemanticSimilarityEngine.build_index_from_dataframe()
  │
  ├── STEP A: Generate Embeddings
  │   └──► src/ml/semantic/embedding_generator.py
  │         ├── For each row: builds semantic string
  │         │     "[Category: X] [Service: Y] [Priority: Z] Description..."
  │         ├── Fits TfidfVectorizer (sparse matrix)
  │         ├── Applies TruncatedSVD → 384 dimensions (dense)
  │         ├── L2 normalizes all vectors
  │         ├── Saves: models/embeddings/tfidf_svd_pipeline.pkl
  │         ├── Saves: models/embeddings/incident_embeddings.npy
  │         └── Saves: models/embeddings/incident_metadata.csv
  │
  ├── STEP B: Build FAISS Index
  │   └──► src/ml/semantic/faiss_index.py
  │         ├── Creates faiss.IndexFlatIP (exact inner product search)
  │         ├── Adds all 384-D vectors to the index
  │         ├── Saves: indexes/incident_semantic_index_latest.index
  │         ├── Saves: indexes/incident_semantic_index_latest_metadata.csv
  │         └──► src/ml/embedding_registry.py → registers index with SHA256
  │
  └── Returns total vector count
```

---

### STAGE 12 → Hybrid Recommendation
**What it does:** Fuses CatBoost + FAISS for final prediction
```
main_cli.py calls HybridRecommendationEngine.recommend()
  │
  ├── Step 1: Parse input (JSON / dict / free text → ticket dict)
  │
  ├── Step 2: CatBoost prediction
  │     ├── Loads models/catboost_assignment_group.pkl
  │     │     └── src/ml/catboost/transformers.py runs inside the pipeline
  │     ├── predict() → assignment group
  │     ├── predict_proba() → confidence score
  │     ├── Loads models/catboost_resolution_time_hours.pkl
  │     └── predict() → MTTR hours (with expm1 inverse)
  │
  ├── Step 3: FAISS search
  │     └──► src/ml/semantic/similarity_engine.py
  │           ├──► embedding_generator.py → converts query to 384-D vector
  │           └──► faiss_index.py → searches Top-5 nearest neighbors
  │
  ├── Step 4: Decision fusion
  │     └──► src/ml/hybrid/decision_engine.py
  │           ├── Compares CatBoost prediction vs FAISS consensus
  │           ├── If agreed → use CatBoost, bonus +0.1
  │           ├── If CatBoost strong → use CatBoost, penalty -0.05
  │           ├── If FAISS dominant → override with FAISS consensus
  │           └──► src/ml/hybrid/confidence_engine.py
  │                 ├── Score = 0.6×CatBoost + 0.4×FAISS ± bonus
  │                 └── Maps to tier: Very High / High / Moderate / Low
  │
  ├── Step 5: Generate explanation
  │     └──► src/ml/hybrid/reasoning_engine.py
  │           └── Builds natural language summary (no LLMs, deterministic)
  │
  └── Step 6: Writes reports/hybrid_prediction.json + .md + .csv
```

---

## 3. DASHBOARD PATH — Separate Entry Point

```
You type: python run_dashboard.py
                │
                ▼
run_dashboard.py
  │  Runs: subprocess.run([python, -m, streamlit, run, src/dashboard/app.py])
  │
  └──► src/dashboard/app.py
         │
         ├── Loads config via src/utils/config_manager.py
         ├── Creates HybridRecommendationEngine (cached, loads once)
         │     └── Loads both CatBoost .pkl models + FAISS index
         │
         ├── Renders web form (description, category, priority, etc.)
         │
         └── On "Predict" click:
               └──► recommendation_engine.recommend(ticket_dict)
                      (Same Stage 12 flow as above)
               └── Displays: Team, Confidence, MTTR, Reasoning, Precedents
```

---

## 4. SUPPORTING FILES — Used by Multiple Stages

These files don't have their own "stage" — they are called by many stages:

| File | What It Provides | Who Uses It |
|---|---|---|
| `src/utils/__init__.py` | `robust_read_csv()`, `robust_open()`, `robust_json_load()` — reads files trying 4 encodings | Every file that reads from disk |
| `src/utils/config_manager.py` | `ConfigManager` — loads all YAML configs + .env | Loaded once at boot, queried everywhere |
| `src/utils/logger.py` | `get_logger(name)` — JSON logging with file rotation | Imported at the top of every single module |
| `src/data/feature_registry.py` | `FeatureRegistry` — 49 features × 22 governance dimensions | cleaner, engineer, eda, splitter, trainer, evaluator, shap_explainer |
| `src/data/feature_lineage.py` | `FeatureLineageTracker` — parent→child derivation formulas | engineer.py writes it, trainer.py reads it |
| `src/data/pipeline_contracts.py` | `PipelineContractValidator` — returns authorized feature lists | eda, text_preprocessor, trainer |
| `src/data/quality_gate.py` | `QualityGateRunner` — 6-gate certification (configs, docs, schema, quality, readiness) | main_cli.py (optional gate check) |
| `src/data/version_manager.py` | `DatasetVersionManager` — immutable dataset snapshots | main_cli.py (optional versioning) |
| `src/ml/model_registry.py` | `ModelRegistry` — SHA256 checksums, feature compliance, versioning | trainer, evaluator, recommendation_engine, shap_explainer |
| `src/ml/embedding_registry.py` | `EmbeddingRegistry` — tracks FAISS index metadata | faiss_index.py |
| `src/ml/catboost/transformers.py` | `EnterpriseFeatureExtractor`, `FrequencyEncoder` — custom sklearn transformers | Embedded inside .pkl pipelines, runs during training AND inference |
| `src/ml/catboost/evaluator.py` | `ModelEvaluator` — confusion matrix, ROC, feature importance | main_cli.py (evaluate command) |
| `src/ml/explainability/shap_explainer.py` | `SHAPIntelligenceExplainer` — SHAP global + local attribution | main_cli.py (explain command) |

---

## 5. CONFIG FILES — What Each One Controls

| File | Controls |
|---|---|
| `config/config.yaml` | Data paths, app settings, report branding, log rotation |
| `config/model_config.yaml` | CatBoost hyperparams, HPO search space, FAISS settings, hybrid engine weights |
| `config/logging.yaml` | Log formatters, handlers (console + 4 rotating files), log levels |
| `config/servicenow.yaml` | ServiceNow API URL, auth, field mapping, query filters |
| `.env` | Secrets: `SERVICENOW_URL`, `SERVICENOW_USER`, `SERVICENOW_PASS`, `DATA_INPUT_PATH` |

---

## 6. EMPTY FOLDERS — Required at Root

These must exist before running the pipeline (main_cli.py creates them automatically, but create them manually if needed):

```
data/raw/            ← Put incidents.csv here
data/processed/      ← Pipeline writes train.csv, val.csv, test.csv here
models/              ← Pipeline saves .pkl model files here
models/embeddings/   ← Pipeline saves TF-IDF+SVD pipeline + vectors here
indexes/             ← Pipeline saves FAISS .index files here
reports/             ← Pipeline writes all audit reports here
reports/figures/     ← EDA writes PNG charts here
logs/                ← Logger writes rotating log files here
```

---

## 7. QUICK REFERENCE — One-Line Summary Per Script

| # | Script | One-Line Purpose |
|---|---|---|
| 1 | `main.py` | Parses CLI args → dispatches to main_cli.py |
| 2 | `run_dashboard.py` | Launches Streamlit dashboard via subprocess |
| 3 | `src/cli/main_cli.py` | Central controller — 19 commands, 24-option menu |
| 4 | `src/dashboard/app.py` | Streamlit web UI for live predictions |
| 5 | `src/utils/__init__.py` | Multi-encoding CSV/JSON/file readers |
| 6 | `src/utils/config_manager.py` | Singleton YAML config loader with env var interpolation |
| 7 | `src/utils/logger.py` | JSON log formatter + rotating file handler factory |
| 8 | `src/data/feature_registry.py` | Single source of truth for all 49 features (22 dims each) |
| 9 | `src/data/feature_lineage.py` | Tracks parent→child feature derivation formulas |
| 10 | `src/data/pipeline_contracts.py` | Returns authorized feature lists per use case |
| 11 | `src/data/quality_gate.py` | 6-gate quality certification engine |
| 12 | `src/data/readiness.py` | ML feasibility diagnostic (leakage, entropy, tokens) |
| 13 | `src/data/validation.py` | 12-rule data quality checker |
| 14 | `src/data/version_manager.py` | Immutable dataset version snapshots |
| 15 | `src/preprocessing/cleaner.py` | 8-step data cleaning pipeline |
| 16 | `src/preprocessing/eda.py` | Automated EDA with 5 diagnostic charts |
| 17 | `src/preprocessing/engineer.py` | Creates 7 categories of derived ML features |
| 18 | `src/preprocessing/enricher.py` | Merges external CMDB/shift data (optional) |
| 19 | `src/preprocessing/splitter.py` | Stratified train/val/test split with zero-leakage check |
| 20 | `src/preprocessing/text_preprocessor.py` | 6-step NLP text normalization |
| 21 | `src/ml/model_registry.py` | Tracks model files, SHA256 checksums, feature compliance |
| 22 | `src/ml/embedding_registry.py` | Tracks FAISS index files and metadata |
| 23 | `src/ml/catboost/trainer.py` | CatBoost classifier + regressor training with HPO |
| 24 | `src/ml/catboost/evaluator.py` | Model evaluation: metrics, ROC, confusion matrix |
| 25 | `src/ml/catboost/transformers.py` | Custom sklearn transformers (feature extraction, frequency encoding) |
| 26 | `src/ml/explainability/shap_explainer.py` | SHAP global + local feature attribution |
| 27 | `src/ml/semantic/embedding_generator.py` | TF-IDF + SVD → 384-D dense vectors |
| 28 | `src/ml/semantic/faiss_index.py` | FAISS index build, search, save, load |
| 29 | `src/ml/semantic/similarity_engine.py` | Orchestrates embedding + FAISS for precedent search |
| 30 | `src/ml/hybrid/confidence_engine.py` | Computes fused confidence score + tier |
| 31 | `src/ml/hybrid/decision_engine.py` | Fuses CatBoost vs FAISS (agree/override logic) |
| 32 | `src/ml/hybrid/reasoning_engine.py` | Generates natural language explanation |
| 33 | `src/ml/hybrid/recommendation_engine.py` | Master orchestrator: 6-step hybrid workflow |
