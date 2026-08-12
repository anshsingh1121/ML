# Incident Intelligence Platform - Architecture Documentation

## 1. System Overview
The First Citizens Bank Incident Intelligence Platform (IIP) is an enterprise-grade machine learning system designed to automate IT service management tasks, specifically ticket classification and resolution time estimation. 

## 2. Core Components

### 2.1. Feature Registry (`src/data/feature_registry.py`)
The `FeatureRegistry` serves as the centralized source of truth for all data attributes. It defines:
- **Encoding Strategies**: e.g., one_hot, frequency, text, target.
- **Leakage Classification**: e.g., safe, blocked, warning.
- **Lineage Tracking**: Enables tracing the origin and transformation steps of all features.

### 2.2. Data Processing Pipeline (`src/preprocessing/`)
- **Cleaner (`cleaner.py`)**: Standardizes input data, enforces schemas, handles missing values, and drops invalid records.
- **Engineer (`engineer.py`)**: Derives advanced ML features such as datetime cyclic transforms and categorical interactions.
- **Splitter (`splitter.py`)**: Stratifies datasets securely into train/val/test partitions while avoiding time-based leakage.
- **Text Preprocessing (`text_preprocessor.py`)**: Prepares raw text fields for semantic analysis.

### 2.3. Quality Governance (`src/data/`)
- **Validation (`validation.py`)**: Runs enterprise checks against timestamps, SLAs, priorities, and CMDB schemas.
- **Readiness (`readiness.py`)**: Checks for target leakage, class imbalances, and overall ML viability before training.
- **Quality Gate (`quality_gate.py`)**: The overarching system that combines all validation checks into a Phase 1 certification.

### 2.4. Machine Learning Module (`src/ml/catboost/`)
- **Trainer (`trainer.py`)**: Constructs scikit-learn pipelines with zero-leakage preprocessing. Utilizes `CatBoost` for classification (Assignment Group) and regression (Resolution Time).
- **Evaluator (`evaluator.py`)**: Generates rigorous metrics and diagnostic charts.
- **HPO (`hpo.py`)**: Performs Hyperparameter Optimization with Optuna.
- **Explainability (`src/ml/explainability/shap_explainer.py`)**: Extracts SHAP values to explain predictions locally and globally.

### 2.5. Semantic Search (`src/ml/semantic/`)
Instead of using large language models or external API calls, semantic search runs 100% locally:
- **Embedding Generator**: Combines `TF-IDF` and `TruncatedSVD` to create 384-dimensional dense vectors from text fields.
- **FAISS Vector Index**: A high-performance local nearest-neighbor search index that finds historical precedents (similar tickets) efficiently.
- **Hybrid Recommendation**: Combines `CatBoost` predictions and FAISS historical precedents to provide well-rounded resolutions.

### 2.6. CLI & Orchestration (`src/cli/main_cli.py`)
An interactive, robust command-line interface managing all aspects of the ML lifecycle, from initial data loading and EDA to final model evaluation and inference.

## 3. Strict Operating Constraints
- **Zero Cloud Dependencies**: The system relies on local TF-IDF and SVD, avoiding internet downloads for transformer models.
- **CatBoost Consolidation**: The project has migrated off Random Forest implementations in favor of the more performant CatBoost library for core supervised tasks.
