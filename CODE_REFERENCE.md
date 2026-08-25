# Codebase Documentation

This document provides a comprehensive, section-wise overview of the core functionality within the codebase, extracted directly from code definitions and docstrings.

## Table of Contents
- [Main Entry Point](#main-entry-point)
- [CLI (Command Line Interface)](#cli-command-line-interface)
- [Dashboard](#dashboard)
- [Data Management](#data-management)
- [Machine Learning Models](#machine-learning-models)
- [Preprocessing & Features](#preprocessing-features)
- [Utilities](#utilities)

---

## Main Entry Point

### File: `main.py`

**Module Docstring:**
Enterprise Incident Intelligence Platform (IIP) — First Citizens Bank (`v2.0.0-alpha`).
Root executable entry point (`main.py`).

Provides interactive menu interface when invoked without arguments (`python main.py`)
or direct subcommand automation (`python main.py generate/validate/eda/clean/engineer/split/pipeline/status/train/evaluate/explain/models/predict/embed/index/similar`).

### Function: `main`
Parse command line arguments and execute platform CLI.


---

## CLI (Command Line Interface)

### File: `src/cli/main_cli.py`

**Module Docstring:**
Enterprise CLI & Operational Control Plane (`src/cli/main_cli.py`).

Provides robust command-line subcommands (`generate`, `validate`, `readiness`, `eda`,
`clean`, `engineer`, `split`, `pipeline`, `status`, `train`, `evaluate`, `explain`, `models`, `predict`)
and an interactive terminal menu (1-15) for operating the First Citizens Bank Incident Intelligence Platform (`v2.0.0`).

### Class: `EnterpriseCLI`
Unified operational command-line controller for the AI-Powered Incident Intelligence Platform.
Enforces enterprise exception handling, directory auto-creation, and comprehensive console reporting.

#### Method: `__init__`
Initialize CLI engine, Feature Registry, and Model Registry with automatic directory creation.

#### Method: `_check_and_self_heal`
Check if core runtime artifacts are missing and trigger automatic self-healing if required.

#### Method: `run_command`
Dispatch subcommand arguments to appropriate execution pipeline with self-healing interlock.

#### Method: `cmd_validate`
Run enterprise dataset validation framework.

#### Method: `cmd_readiness`
Run ML readiness verification.

#### Method: `cmd_eda`
Run automated Exploratory Data Analysis (EDA) engine.

#### Method: `cmd_clean`
Execute automated data cleaning and remediation pipeline.

#### Method: `cmd_engineer`
Execute feature engineering and registry synchronization pipeline.

#### Method: `cmd_split`
Partition dataset into Train/Val/Test with zero boundary leakage.

#### Method: `cmd_pipeline`
Run complete end-to-end Data Intelligence Pipeline (Clean -> Engineer -> Preprocess -> Split).

#### Method: `cmd_train`
Train CatBoost pipelines (`assignment_group` or `resolution_time_hours`).

#### Method: `cmd_evaluate`
Evaluate trained model pipeline across test partition (`test.csv`).

#### Method: `cmd_explain`
Run SHAP Explainable AI diagnostics on trained model pipeline.

#### Method: `cmd_models`
List and audit all registered models in ModelRegistry.

#### Method: `cmd_predict`
Execute zero-manual-preprocessing inference and export structured prediction metadata.

#### Method: `cmd_embed`
Execute Phase 4 offline local neural embedding generation (`TF-IDF + SVD`).

#### Method: `cmd_index`
Build, persist, and register FAISS vector similarity index.

#### Method: `cmd_similar`
Retrieve Top-K semantically similar historical incidents.

#### Method: `cmd_recommend`
Run deterministic Hybrid Incident Intelligence Engine recommendation.

#### Method: `_run_stage1_dataset_check`
Stage 1 helper: Check if real/existing dataset exists; fail if missing.

#### Method: `cmd_full_pipeline`
Orchestrate all 12 stages of the Enterprise Incident Intelligence Platform sequentially.

#### Method: `cmd_clean_workspace`
Remove generated runtime artifacts while preserving source files and directory hierarchy.

#### Method: `cmd_status`
Display real-time platform health and registry status.

#### Method: `run_interactive_menu`
Run non-blocking interactive terminal control menu (Options 1-21).


---

## Dashboard

### File: `src/dashboard/app.py`

### Function: `get_engine`
Cache the engine initialization so it doesn't reload models on every UI interaction.


---

## Data Management

### File: `src/data/feature_lineage.py`

**Module Docstring:**
Feature Lineage Tracker (`v1.5.0`).

Tracks parent-child derivation relationships and mathematical transformation formulas
for all engineered attributes in the AI-Powered Incident Intelligence Platform.

### Class: `LineageEdge`
Represents a direct transformation step from source attribute(s) to derived attribute.

#### Method: `to_dict`
Convert lineage edge to dictionary.

### Class: `FeatureLineageTracker`
Centralized governance graph tracking exact ancestry and formulas for all derived features.
Supports singleton access via get_instance() and direct edge creation.

#### Method: `get_instance`
Retrieve or initialize the singleton FeatureLineageTracker instance.

#### Method: `__init__`
Initialize FeatureLineageTracker and populate standard enterprise derivation rules.

#### Method: `add_lineage`
Register a feature derivation step.

#### Method: `add_edge`
Helper wrapper to create or merge a LineageEdge from string or list arguments.

#### Method: `get_lineage`
Retrieve lineage definition for a specific derived feature.

#### Method: `get_ancestry_chain`
Recursively trace back all ancestral raw source features.

#### Method: `export_json`
Export full feature lineage graph to feature_lineage.json.

#### Method: `export_markdown`
Export full feature lineage to feature_lineage.md specification table and tree.

#### Method: `_populate_default_lineage`
Populate default derivation rules for temporal and relationship flags.


### File: `src/data/feature_registry.py`

**Module Docstring:**
Enterprise Feature Registry — Single Source of Truth (`v1.5.0`).

Centralizes definitions, schemas, data types, leakage classifications, ML usages,
and transformation rules for every attribute (raw & derived) in the AI-Powered
Incident Intelligence Platform.

Downstream modules (Random Forest, SentenceTransformer, FAISS, Dashboard, API, RAG)
MUST consume this registry instead of relying on hardcoded feature lists.

### Class: `FeatureDefinition`
Formal specification for a single enterprise feature attribute across 22 governance dimensions.

#### Method: `to_dict`
Convert feature definition to dictionary.

### Class: `FeatureRegistry`
Singleton-style centralized Feature Registry managing all 38 raw ServiceNow columns
and 11 derived engineering features.

#### Method: `__init__`
Initialize FeatureRegistry and populate with enterprise catalog if requested.

#### Method: `get_instance`
Get or create singleton FeatureRegistry instance.

#### Method: `reset_instance`
Reset the singleton instance and clear registry definitions (for unit testing).

#### Method: `register_feature`
Register or overwrite a feature definition.

#### Method: `get_feature`
Retrieve a specific feature definition by technical name.

#### Method: `list_all_features`
List all registered feature definitions.

#### Method: `get_features_by_usage`
Filter features by a specific usage classification.

#### Method: `get_features_by_leakage`
Retrieve all features matching a leakage tier ('safe', 'warning', 'blocked').

#### Method: `get_catboost_predictors`
Retrieve safe predictor column names authorized for Random Forest training.

#### Method: `get_embedding_features`
Retrieve text column names authorized as primary/secondary embedding inputs.

#### Method: `get_faiss_metadata_features`
Retrieve column names authorized as structural FAISS metadata filters/boosts.

#### Method: `resolve_business_name`
Resolve any raw or pipeline-transformed technical feature name to its exact enterprise business name.
Handles one-hot expanded indicators (`category_Hardware`) and cyclic interaction terms (`priority_x_business_impact`).

#### Method: `export_json`
Export full feature registry to feature_registry.json.

#### Method: `export_markdown`
Export full feature registry to feature_registry.md enterprise table.

#### Method: `_populate_default_registry`
Populate the registry with the complete 38 raw ServiceNow schema + 11 derived features.


### File: `src/data/pipeline_contracts.py`

**Module Docstring:**
Enterprise Pipeline Contracts (`v1.5.0`).

Provides standardized API adapters that downstream modules call to obtain authorized
feature sets directly from the central Feature Registry, preventing hardcoded columns
and verifying dataframe schema/leakage compliance before any model processing.

### Class: `PipelineContractValidator`
Standard adapter interface governing feature access across 6 downstream modules:
Random Forest, Embeddings, FAISS, Dashboard, ServiceNow API, and Future RAG.

#### Method: `__init__`
Initialize validator with FeatureRegistry singleton.

#### Method: `get_catboost_features`
Retrieve safe predictor column names authorized for Random Forest training.

#### Method: `get_embedding_text_features`
Retrieve text column names authorized for neural embedding tokenization.

#### Method: `get_faiss_metadata_features`
Retrieve column names authorized for structural exact matching and FAISS filtering.

#### Method: `get_dashboard_kpi_features`
Retrieve column names authorized for dashboard filters and interactive chart axes.

#### Method: `get_api_request_schema`
Retrieve JSON schema of required and optional payload fields for REST API ingestion.

#### Method: `get_rag_knowledge_features`
Retrieve column names authorized as knowledge chunks or citations in RAG retrieval.

#### Method: `validate_dataframe_compliance`
Verify that a candidate dataframe conforms to registry boundaries and contains no Blocked features.

Args:
    df: Candidate pandas DataFrame.
    expected_usage: 'triage_prediction' (strict exclusion of post-resolution fields) or 'post_resolution_analytics'.

Returns:
    Tuple[bool, List[str]]: (is_compliant, list_of_violations)


### File: `src/data/quality_gate.py`

**Module Docstring:**
Enterprise Quality Gate Certification Engine — Phase 1.5 Gatekeeper.

Systematically validates configuration, schema, synthetic dataset quality,
automation batch scripts, and architectural documentation before certifying
readiness to transition into Phase 2 (Exploratory Data Analysis).

Design Decisions:
    - Multi-Tier Certification: Divides governance checks into 6 explicit domains
      to isolate failures and prevent incomplete pipelines from entering ML training.
    - Automated Certification Generation: Outputs an executive audit artifact at
      reports/quality_gate_certification.md summarizing all gate results.

### Class: `QualityGateRunner`
Executes comprehensive system quality checks across 6 critical domains
to verify enterprise readiness before Phase 2 initiation.

#### Method: `__init__`
Initialize QualityGateRunner with core services.

#### Method: `get_project_root`
Resolve the primary project repository root.

#### Method: `run_all_gates`
Execute all 6 quality gates and issue a formal certification summary.

Args:
    save_certification: Whether to output quality_gate_certification.md.
    report_dir: Custom output folder. Defaults to reports/.

Returns:
    Dictionary detailing PASS/FAIL status per gate and overall certification.

#### Method: `validate_configuration`
Verify YAML configs exist, parse cleanly, and Singleton works.

#### Method: `validate_automation_scripts`
Verify Windows batch automation scripts exist and are non-empty.

#### Method: `validate_documentation`
Verify all mandatory enterprise documentation and Mermaid diagrams exist.

#### Method: `validate_schema`
Verify dataset adheres to the full 35+ attribute ServiceNow schema.

#### Method: `save_certification_report`
Save formal Quality Gate Certification summary to reports/quality_gate_certification.md.


### File: `src/data/readiness.py`

**Module Docstring:**
ML Readiness Evaluation Framework — Enterprise Feature & Leakage Analysis.

Automates the diagnostic inspection of synthetic or real ServiceNow incident
datasets before model training. Identifies target leakage, class imbalance,
textual token capacity, high cardinality, and multi-collinearity.

Design Decisions:
    - Target Leakage Protection: Proactively segregates time-of-ticket-creation
      features from post-resolution operational data to guarantee zero leakage during
      Assignment Group classifier training.
    - Information Theoretic Metrics: Computes Shannon Entropy and Gini Impurity for
      categorical targets to quantify class imbalance severity and guide resampling.
    - NLP Readiness Diagnostics: Evaluates token and character distributions against
      TF-IDF and TruncatedSVD components properly bounded.

### Class: `MLReadinessEvaluator`
Evaluates dataset quality, statistical distributions, target leakage risks,
and feature transformations required for ML model training.

#### Method: `__init__`
Initialize MLReadinessEvaluator with optional ConfigManager.

#### Method: `evaluate_dataset`
Conduct a comprehensive ML readiness audit of the dataset.

Args:
    df: Pandas DataFrame containing incident records.
    target_column: Primary target variable for classification readiness.
    save_report: Whether to output reports/ml_readiness_report.md & .json.
    report_dir: Custom directory path for saving reports.

Returns:
    Dictionary containing all diagnostic metrics, leakage flags, and recommendations.

#### Method: `_compute_missing_percentage`
Calculate missing value percentage per column.

#### Method: `_compute_duplicate_percentage`
Calculate duplicate record statistics.

#### Method: `_compute_cardinality`
Calculate distinct value counts and cardinality ratios across categorical columns.

#### Method: `_detect_target_leakage`
Identify potential target leakage features that must be dropped during early triage modeling.

#### Method: `_analyze_class_imbalance`
Compute entropy, Gini impurity, and imbalance ratios across core target fields.

#### Method: `_compute_text_statistics`
Compute character and estimated token length statistics for text columns.

#### Method: `_compute_correlation_matrix`
Compute numeric correlation summary across continuous and ordinal fields.

#### Method: `_generate_recommendations`
Generate specific preprocessing and feature engineering action items.

#### Method: `save_readiness_report`
Save ML Readiness Report to reports/ml_readiness_report.md and .json.


### File: `src/data/validation.py`

**Module Docstring:**
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

### Class: `CheckResult`
Standardized result representing the outcome of a single validation rule check.

### Class: `DatasetValidator`
Automated validation engine verifying 12 strict data quality requirements across
ServiceNow attributes, timestamps, domain catalogs, and SLA mechanics.

#### Method: `__init__`
Initialize the DatasetValidator with optional ConfigManager.

#### Method: `validate_dataset`
Execute all 12 validation checks on the provided dataset.

Args:
    df: Pandas DataFrame containing incident records to validate.
    save_report: Whether to save validation report to disk.
    report_dir: Custom output directory for reports. Defaults to reports/.

Returns:
    Dictionary containing overall validation status, check counts, and detailed results.

#### Method: `_check_missing_values`
Rule 1: Check for missing or null values across critical required fields.

#### Method: `_check_duplicate_incidents`
Rule 2: Check for exact duplicate number keys.

#### Method: `_check_invalid_timestamps`
Rule 3: Check for logical timestamp anomalies (resolved < opened, closed < resolved).

#### Method: `_check_invalid_categories`
Rule 4: Check if categories exist and are not null.

#### Method: `_check_invalid_assignment_groups`
Rule 9: Check if assignment groups exist and are not null.

#### Method: `_check_invalid_priorities`
Rule 6: Check if priorities fall cleanly within integer levels 1 through 5, or string formats starting with 1-5.

#### Method: `_check_sla_inconsistencies`
Rule 7: Check if SLA status (made_sla) logically aligns with resolution time vs targets.

#### Method: `_check_resolution_time_inconsistencies`
Rule 8: Check for negative resolution times or divergence from resolved_at - opened_at.

#### Method: `_check_invalid_cmdb_references`
Rule 9: Check if cmdb_ci references are populated or adhere to CI naming patterns.

#### Method: `_check_invalid_business_services`
Rule 10: Check if business_service is populated with valid service catalog mappings.

#### Method: `_check_empty_descriptions`
Rule 11: Check for empty or whitespace-only description fields.

#### Method: `_check_empty_short_descriptions`
Rule 12: Check for empty or whitespace-only short_description fields.

#### Method: `save_validation_report`
Save validation results to disk in both JSON and Markdown format.

Args:
    summary: The validation summary dictionary returned by validate_dataset().
    report_dir: Target directory path. Defaults to reports/.

Returns:
    Tuple containing paths to the saved (JSON, Markdown) report files.


### File: `src/data/version_manager.py`

**Module Docstring:**
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

### Class: `DatasetVersionManager`
Manages non-destructive dataset versioning under datasets/synthetic/vX/,
metadata manifest generation (`metadata.json`), and version history tracking.

#### Method: `__init__`
Initialize DatasetVersionManager with base directory path.

#### Method: `get_latest_version_number`
Determine the highest version number currently stored on disk.

#### Method: `get_next_version_id`
Get the next version string ID (e.g., 'v1', 'v2').

#### Method: `save_versioned_dataset`
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

#### Method: `_generate_metadata`
Compute statistical distributions and format metadata manifest.

#### Method: `_update_history`
Append metadata summary to version_history.json.

#### Method: `load_metadata`
Retrieve metadata manifest for a specific dataset version.

#### Method: `list_all_versions`
Return the complete list of historical dataset versions.


---

## Machine Learning Models

### File: `src/ml/model_registry.py`

**Module Docstring:**
Central Model Registry Architecture (`v1.5.0`).

Tracks model versions, hyperparameters, evaluation metrics, features used,
dataset provenance, SHA256 cryptographic checksums, and Feature Registry compliance.
Enforces zero schema drift and immutability across all trained classifiers and regressors.

### Class: `ModelMetadata`
Formal specification for a registered ML model version.

#### Method: `to_dict`
Convert model metadata to dictionary.

### Class: `ModelValidationException`
Raised when a model fails SHA256 verification or feature schema compliance.

### Class: `ModelRegistry`
Singleton-style Central Model Registry storing metadata and enforcing checksum compliance.

#### Method: `__init__`
Initialize ModelRegistry at base_dir or default models/ directory.

#### Method: `get_instance`
Get or create singleton ModelRegistry instance.

#### Method: `compute_sha256`
Compute SHA256 cryptographic hash of a file on disk.

#### Method: `register_model`
Register a new trained model, compute SHA256, and save to registry.

#### Method: `get_model_metadata`
Retrieve model metadata by name and exact/latest version.

#### Method: `verify_and_load_model_path`
Verify SHA256 checksum and Feature Registry version compliance before authorizing model loading.

#### Method: `get_model_path`
Retrieve the canonical path to a model file using the Model Registry as the single source of truth.
If unregistered or in an unmaterialized test environment, safely checks fallback path inside base_dir.

#### Method: `export_markdown`
Export model registry to model_registry.md.

#### Method: `_load_registry`
Load registry from JSON disk if present.

#### Method: `_save_registry`
Save registry dictionary to JSON disk.


### File: `src/ml/embedding_registry.py`

**Module Docstring:**
Embedding & FAISS Index Registry (`v1.5.0`).

Manages vector generation specifications, neural model boundaries, chunking strategies,
distance metrics, vector counts, index file paths, and SHA256 checksums.

### Class: `EmbeddingIndexMetadata`
Formal specification for a registered FAISS vector index.

#### Method: `to_dict`
Convert metadata to dictionary.

### Class: `EmbeddingRegistry`
Singleton-style registry tracking vector indexes and embedding model configurations.

#### Method: `__init__`
Initialize EmbeddingRegistry at base_dir or default indexes/ directory.

#### Method: `get_instance`
Get or create singleton EmbeddingRegistry instance.

#### Method: `compute_sha256`
Compute SHA256 cryptographic hash of index file on disk.

#### Method: `register_index`
Register a FAISS vector index.

#### Method: `get_index_metadata`
Retrieve index metadata by name and version.

#### Method: `export_markdown`
Export index registry to embedding_registry.md.

#### Method: `_load_registry`
Load from JSON disk if present.

#### Method: `_save_registry`
Save dictionary to JSON disk.


### File: `src/ml/catboost/evaluator.py`

**Module Docstring:**
Enterprise Model Evaluation & Feature Importance Engine (`v1.5.0`).

Evaluates complete scikit-learn `Pipeline` objects across test partitions (`data/processed/test.csv`).
Computes multi-class Top-K accuracy, weighted/macro precision/recall/F1, ROC-AUC (`ovr`),
regression RMSE/MAE/$R^2$, Confusion Matrix/ROC charts (`reports/`), and Feature Importance rankings.

### Class: `ModelEvaluator`
Enterprise ML Evaluator for classification (`assignment_group`) and regression (`resolution_time_hours`).
Generates professional visual plots and structured Feature Importance audit artifacts.

#### Method: `__init__`
_No docstring provided._

#### Method: `_get_safe_predictor_matrix`
Ensure all predictor columns exist in the dataframe before slicing, initializing missing with safe defaults.

#### Method: `_resolve_pipeline_and_test_data`
Resolve model pipeline file path and load predictors/targets from test partition.

#### Method: `extract_and_export_feature_importances`
Extract tree feature importances, align with FeatureRegistry business names, and export CSV/MD/PNG.

#### Method: `evaluate_classification`
Run complete classification evaluation, rendering Confusion Matrix, ROC curves, and formal metrics.

#### Method: `evaluate_regression`
Run complete regression evaluation (`MAE`, `RMSE`, `R2`) on actual resolution hours.


### File: `src/ml/catboost/trainer.py`

**Module Docstring:**
Enterprise Random Forest Trainer (`v1.5.0`).

Orchestrates data loading, zero-leakage preprocessing pipeline construction (`scikit-learn`),
multi-baseline model training (Decision Tree, Random Forest, Extra Trees, and optional XGBoost/LightGBM),
target leakage interlock verification, `joblib` Pipeline persistence, and formal registration inside `ModelRegistry`.

### Class: `EnterpriseCatBoostTrainer`
Enterprise-grade Random Forest and baseline ML trainer for First Citizens Bank Incident Intelligence Platform.
Ensures zero target leakage, complete preprocessing + estimator pipeline persistence, and model registry compliance.

#### Method: `__init__`
_No docstring provided._

#### Method: `_verify_no_target_leakage`
Interlock: Verify that no unauthorized or blocked target leakage columns enter the predictor matrix.

#### Method: `_get_safe_predictor_matrix`
Ensure all predictor columns exist in the dataframe before slicing, initializing missing with safe defaults.

#### Method: `build_preprocessing_pipeline`
Construct a self-contained scikit-learn preprocessing pipeline for zero-leakage inference.
Partition predictors into frequency encoded, one-hot encoded, and numerical scaled branches.

#### Method: `train_baselines_and_compare`
Train multiple baseline models (Decision Tree, Random Forest, Extra Trees, optional XGBoost/LightGBM).
Compare validation metrics, generate formal reports, and automatically identify the best-performing model.

#### Method: `_export_baseline_comparison_report`
Export multi-baseline comparison results to markdown and json reports.

#### Method: `train_classifier`
Train primary CatBoost (and multi-baseline) classification pipeline on triage predictors.
Persist complete sklearn Pipeline to disk (`models/catboost_assignment_group.pkl`), and register inside ModelRegistry.

#### Method: `train_regressor`
Train primary CatBoost (and multi-baseline) regression pipeline on triage predictors.
Applies log1p target transformation (`np.log1p`) to normalize right-skewed resolution windows.
Persist complete sklearn Pipeline to disk (`models/catboost_resolution_time_hours.pkl`), and register inside ModelRegistry.


### File: `src/ml/catboost/transformers.py`

**Module Docstring:**
Scikit-Learn Compatible Custom Transformers (`v1.5.0`).

Provides enterprise-grade preprocessing classes that implement `BaseEstimator` and `TransformerMixin`.
Enables persisting complete end-to-end `sklearn.pipeline.Pipeline` objects (preprocessing + model)
to disk (`joblib.dump`), allowing zero-manual-preprocessing inference at prediction time.

### Class: `DataFrameSelector`
Selects a subset of column names from a pandas DataFrame and returns a DataFrame or numpy array.
Guarantees consistent column ordering and safe handling of missing columns during inference.

#### Method: `__init__`
_No docstring provided._

#### Method: `fit`
Store input feature names.

#### Method: `transform`
Select specified columns, filling any missing features with default neutral values.

#### Method: `get_feature_names_out`
Return output feature names corresponding to selected attributes.

### Class: `EnterpriseFeatureExtractor`
Extracts non-linear interactions (`priority_x_business_impact`, `priority_x_urgency`) and cyclic temporal shifts
if raw columns are present. Ensures raw prediction payloads can be ingested without external engineering steps.

#### Method: `__init__`
_No docstring provided._

#### Method: `fit`
Record input feature names and compute output feature names.

#### Method: `transform`
Extract interactions and cyclic shifts safely.

#### Method: `get_feature_names_out`
Return exact feature names outputted after interaction and cyclic shift extraction.

### Class: `FrequencyEncoder`
Learns normalized frequency distributions (`count / total_rows`) for high-cardinality categorical
columns during training (`fit`), and maps them cleanly during inference (`transform`).
Handles unknown inference labels by assigning `0.0001` (minimum smoothing frequency).

#### Method: `__init__`
_No docstring provided._

#### Method: `fit`
Fit normalized frequency distributions across specified columns.

#### Method: `transform`
Map categorical values to learned frequency numbers.

#### Method: `get_feature_names_out`
Return output feature names (identical to input feature names as encoding is in-place).

### Class: `SmoothedTargetEncoder`
Smoothed Out-of-Fold Target Encoder for high-cardinality categorical attributes (`subcategory`, `business_service`).
Computes `(count * mean + smoothing * global_mean) / (count + smoothing)` during `fit`.
Handles unknown inference labels by imputing the learned `global_mean_`.

#### Method: `__init__`
_No docstring provided._

#### Method: `fit`
Compute smoothed target averages across categories.

#### Method: `transform`
Map categories to learned smoothed target values.

#### Method: `get_feature_names_out`
Return output feature names (identical to input feature names as encoding is in-place).


### File: `src/ml/explainability/shap_explainer.py`

**Module Docstring:**
Explainable AI & Structured Prediction Engine (`v1.5.0`).

Integrates `shap.TreeExplainer` across complete scikit-learn `Pipeline` objects.
Generates global summary (`shap_summary.png`, `shap_bar.png`) and local diagnostic plots
(`shap_waterfall_sample.png`, `shap_decision_sample.png`).
Exports structured prediction metadata (`predicted_class`, `confidence_score`,
`top_contributing_features`, `feature_importances`, `prediction_timestamp`) for Hybrid Similarity integration.

### Class: `SHAPIntelligenceExplainer`
Enterprise Explainable AI Engine for First Citizens Bank Incident Intelligence Platform.
Provides mathematically verified local and global feature attribution via game-theoretic SHAP values.

#### Method: `__init__`
_No docstring provided._

#### Method: `_resolve_pipeline`
Resolve pipeline model object and file path.

#### Method: `_get_safe_predictor_matrix`
Ensure all predictor columns exist in the dataframe before slicing, initializing missing with safe defaults.

#### Method: `_get_transformed_dataframe_with_business_names`
Construct transformed DataFrame with verified enterprise business feature names via get_feature_names_out + FeatureRegistry.

#### Method: `_compute_shap_values`
Intercept CatBoost to use native NLP SHAP, or fallback to standard TreeExplainer.

#### Method: `explain_global`
Compute global SHAP explanations (`TreeExplainer`) and generate summary/bar plots (`reports/`).

#### Method: `explain_prediction`
Execute zero-manual-preprocessing inference, calculate local SHAP attribution,
and export structured prediction metadata (`predicted_class`, `confidence_score`,
`top_contributing_features`, `feature_importances`, `prediction_timestamp`).
Saves structured artifacts (`reports/prediction_metadata.json` & `.csv`).


### File: `src/ml/hybrid/confidence_engine.py`

**Module Docstring:**
Enterprise Hybrid Confidence Engine (`v2.0.0-alpha` - Phase 5).

Calculates deterministic numerical confidence (`0.0` to `1.0`) and categorical confidence
tiers (`Very High`, `High`, `Moderate`, `Low`, `Review Required`) based on cross-engine
agreement between Random Forest classification confidence and Semantic FAISS consensus.

Governance Mandate:
- Zero hardcoded magic numbers (`ConfigManager` reads `ml.hybrid`).
- 100% deterministic mathematical evaluation.

### Class: `HybridConfidenceEngine`
Configuration-driven confidence fusion engine.

Synthesizes numerical probability from Random Forest classifiers (`rf_confidence`)
and average/consensus similarity score from Top-K FAISS retrieval (`sem_confidence`)
into a calibrated overall recommendation confidence and category tier.

#### Method: `__init__`
Initialize HybridConfidenceEngine reading parameters from `ConfigManager`.

#### Method: `calculate_confidence`
Compute fused numerical confidence score and classification tier.

Args:
    rf_confidence: Classification probability/confidence from Random Forest (`0.0 to 1.0`).
    sem_confidence: Consensus similarity or average similarity from FAISS Top-K (`0.0 to 1.0`).
    agreement: `True` if RF predicted group matches the mode/majority assignment group across Top-K precedents.
    top_k_matches: Number of semantic precedents retrieved.

Returns:
    Tuple of `(numerical_confidence_score, categorical_tier)`.


### File: `src/ml/hybrid/decision_engine.py`

**Module Docstring:**
Enterprise Hybrid Decision Engine (`v2.0.0-alpha` - Phase 5).

Deterministically fuses Random Forest ML predictions with FAISS Semantic Top-K
precedents to produce a verified enterprise recommendation, estimated resolution time,
and historical success rate.

Governance Mandate:
- Zero hardcoded magic numbers (`ConfigManager` reads `ml.hybrid`).
- 100% deterministic rule and statistical blending.

### Class: `HybridDecisionEngine`
Orchestration decision layer for fusing ML outputs and Semantic search precedents.

#### Method: `__init__`
Initialize HybridDecisionEngine.

#### Method: `fuse_recommendation`
Execute deterministic fusion logic combining RF prediction and Top-K precedents.

Args:
    rf_prediction: Dictionary containing `assignment_group`, `confidence_score` (or `confidence`),
                   `resolution_time_hours`, and optional `priority`.
    semantic_matches: List of Top-K dictionary matches from `FAISSVectorIndex`.

Returns:
    Structured decision dictionary containing recommended group, fused confidence,
    estimated resolution time, historical success rate, and intermediate consensus metrics.


### File: `src/ml/hybrid/reasoning_engine.py`

**Module Docstring:**
Enterprise Hybrid Reasoning Engine (`v2.0.0-alpha` - Phase 5).

Synthesizes quantitative output metrics from Random Forest predictions and FAISS
Top-K semantic precedents into audit-ready, deterministic natural language justifications.

Governance Mandate:
- Zero LLMs, Zero GenAI, Zero cloud APIs.
- 100% deterministic template and quantitative rule synthesis.

### Class: `HybridReasoningEngine`
Synthesizes quantitative decision metrics and historical precedents into
human-readable explanations and formatted audit summaries.

#### Method: `generate_reasoning`
Generate comprehensive reasoning explanation and formatted Historical Evidence tables.

Args:
    decision: Output dictionary from `HybridDecisionEngine.fuse_recommendation()`.
    semantic_matches: List of Top-K dictionary matches from FAISS index.

Returns:
    Dictionary containing `executive_summary`, `bullet_breakdown`, and `historical_evidence_table`.


### File: `src/ml/hybrid/recommendation_engine.py`

**Module Docstring:**
Enterprise Hybrid Recommendation Engine (`v2.0.0-alpha` - Phase 5).

Single operational controller orchestrating end-to-end incident intelligence:
Step 1: Ingests raw ticket input (JSON file path, dictionary, or free natural language text).
Step 2: Executes Random Forest classification (`assignment_group`) and regression (`resolution_time_hours`).
Step 3: Executes FAISS Top-K semantic similarity search across millions of historical tickets.
Step 4: Fuses both Intelligence streams via `HybridDecisionEngine`.
Step 5: Synthesizes explainable justifications via `HybridReasoningEngine`.
Step 6: Exports structured intelligence reports (`reports/hybrid_prediction.json`, `.md`, `.csv`).

Governance Mandate:
- Zero LLMs, Zero GenAI, Zero cloud APIs.
- 100% deterministic and configuration-driven.
- Full Windows file-lock resilience (`_latest`).

### Class: `HybridRecommendationEngine`
Master orchestration controller for the Enterprise Hybrid Incident Intelligence Engine.

#### Method: `__init__`
Initialize HybridRecommendationEngine and load required sub-engines and models.

#### Method: `_parse_input_payload`
Parse raw JSON path, dictionary, or free-text string into a structured ticket dict.

#### Method: `_sync_features_for_model`
Ensure input DataFrame contains all expected features required by scikit-learn model.

#### Method: `_predict_rf`
Execute Random Forest prediction on ticket dictionary.

#### Method: `recommend`
Orchestrate end-to-end Hybrid Incident Intelligence recommendation.

Args:
    input_payload: JSON file path (`sample.json`), dict, or free text (`"ATM cash jam"`).
    top_k: Number of semantic precedents to retrieve via FAISS.
    export_reports: Whether to save `reports/hybrid_prediction.json`, `.md`, `.csv`.

Returns:
    Comprehensive recommendation dictionary containing fused outputs, Historical Evidence,
    and explainable reasoning justification.

#### Method: `export_reports`
Export formal enterprise recommendation reports (`.json`, `.md`, `.csv`)
with complete Windows file-lock resilience (`_latest`).


### File: `src/ml/semantic/embedding_generator.py`

**Module Docstring:**
Enterprise Semantic Embedding Generator (`v1.5.0` - Phase 4).

Generates dense neural representations (`384-D`) of ServiceNow incidents using a local
Scikit-Learn TF-IDF + TruncatedSVD pipeline. Ensures exact separation of
dense embedding matrices (`.npy`) and structured incident metadata (`.csv`).

Governance Mandate:
- Zero cloud data egress (`device='cpu'` by default, local disk cache).
- Standardized multi-field semantic composition:
  `[Category | Subcategory] [Service | CI] [Priority] Short Description. Description`

### Class: `SemanticEmbeddingGenerator`
Enterprise embedding engine responsible for transforming raw ServiceNow incident records
and natural language query strings into L2-normalized dense vector representations (`384-D`).

#### Method: `__init__`
Initialize the embedding generator.

Args:
    model_name: Identifier for the local model.
    cache_dir: Local storage directory for pre-trained weights and embeddings.
    device: Compute device (kept for compatibility).
    normalize_output: If True, apply L2 normalization to output embeddings (enables Cosine/IP distance).
    n_components: Dimension of the output embeddings.

#### Method: `model`
Lazy-load and return the Scikit-learn Pipeline.

#### Method: `get_embedding_dimension`
Return exact vector dimension (`384`).

#### Method: `construct_semantic_text`
Construct a structured, domain-rich semantic document string by intelligently
combining critical IT incident attributes.

#### Method: `embed_dataframe`
Generate embeddings for an entire historical or production incident DataFrame.

#### Method: `embed_text`
Encode raw natural language text (e.g., query string or free text) into L2-normalized vectors.

#### Method: `save_embeddings`
Store dense embeddings (`.npy`) and structured metadata (`.csv` / `.parquet`)
separately on local disk to enforce clean storage isolation.

#### Method: `load_embeddings`
Load separated dense embeddings and structured metadata from disk.


### File: `src/ml/semantic/faiss_index.py`

**Module Docstring:**
Enterprise FAISS Vector Index Manager (`v1.5.0` - Phase 4).

Provides production-grade vector similarity indexing, exact inner product (Cosine) and
inverted file (IVFFlat) approximate search, incremental updates, index persistence,
and cryptographic registration via `EmbeddingRegistry`.

Governance Mandate:
- Offline local execution using `faiss-cpu`.
- Cryptographic SHA256 integrity validation upon index registration.

### Class: `FAISSVectorIndex`
Enterprise FAISS index controller managing in-memory vector indexing, incremental
incident additions, top-K nearest neighbor search, and metadata mapping.

#### Method: `__init__`
Initialize FAISS index controller.

Args:
    dimension: Vector embedding dimension (typically from TF-IDF + SVD).
    index_type: FAISS index structure (`FlatIP`, `FlatL2`, `IVFFlat`).
    nlist: Number of Voronoi cells/centroids for `IVFFlat`.
    nprobe: Number of centroids to visit during `IVFFlat` search.
    index_dir: Base directory for storing `.index` files and metadata.
    index_name: Unique name for registration and file persistence.
    version: Index version tag (`v1.5.0`, `latest`, etc.).

#### Method: `create_index`
Build and initialize a new FAISS vector index (`FlatIP`, `FlatL2`, or `IVFFlat`).
If initial_embeddings is provided (`IVFFlat` requirement), train the index immediately.

#### Method: `add_embeddings`
Incrementally add vectors and aligned metadata to the active FAISS index.

Returns:
    Total vector count in the index after addition.

#### Method: `search`
Search the FAISS vector index for the Top-K most semantically similar records.

Args:
    query_vector: Dense query vector of shape `(1, D)` or `(D,)`.
    top_k: Number of nearest neighbors to retrieve.

Returns:
    List of structured match dictionaries containing similarity score and all metadata fields.

#### Method: `save_index`
Persist the FAISS binary `.index` file and structured `.csv` metadata table to disk,
and register the index with cryptographic SHA256 checksums inside `EmbeddingRegistry`.

#### Method: `load_index`
Load an existing FAISS binary `.index` and corresponding `.csv` metadata from disk.


### File: `src/ml/semantic/similarity_engine.py`

**Module Docstring:**
Enterprise Semantic Similarity Engine (`v1.5.0` - Phase 4).

Orchestrates zero-cloud neural text embedding (`SemanticEmbeddingGenerator`) and
high-performance vector retrieval (`FAISSVectorIndex`) to identify historical incident
precedents, predict possible assignment routing, and estimate resolution times based
on semantic distance.

Governance Mandate:
- Offline local execution (zero cloud API dependencies).
- Standardized reporting (`reports/similarity_results.csv` & `.md`).
- Resilience against Windows file locks via automatic fallback (`_latest`).

### Class: `SemanticSimilarityEngine`
High-level similarity retrieval controller connecting neural embeddings,
FAISS vector search, and standardized audit reporting.

#### Method: `__init__`
Initialize the Semantic Similarity Engine.

Args:
    embedding_generator: Controller for `TF-IDF + SVD` vector creation.
    faiss_index: Controller for FAISS similarity index.
    reports_dir: Output directory for similarity results and audit tables.

#### Method: `build_index_from_dataframe`
End-to-end pipeline: embed all historical records in df, initialize and populate FAISS,
save binary artifacts to disk, and register in EmbeddingRegistry.

#### Method: `find_similar_incidents`
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

#### Method: `_export_similarity_reports`
Export standardized CSV and Markdown reports (`reports/similarity_results.csv` & `.md`)
with complete PermissionError resilience.


---

## Preprocessing & Features

### File: `src/preprocessing/cleaner.py`

**Module Docstring:**
Enterprise Data Cleaner (`src/preprocessing/cleaner.py`).

Performs automated cleaning, missing value imputation, duplicate removal, timestamp
progression correction, categorical domain mapping, and outlier clipping across
ServiceNow incident datasets. All operations consume the Central Feature Registry
imputation rules (`imputation_strategy`) and document every single modification
inside comprehensive audit reports (`reports/cleaning_report.md` and `.json`).

### Class: `EnterpriseDataCleaner`
Automated data cleaning and quality remediation engine for ServiceNow incident data.
Enforces no silent modifications: every row modification, imputation, duplicate removal,
or outlier clip is precisely tracked, quantified, and exported to audit logs.

#### Method: `__init__`
Initialize cleaning engine with centralized Feature Registry and Config.

#### Method: `clean_dataset`
Execute automated data cleaning pipeline and generate complete audit reports.

Args:
    df: Raw input ServiceNow pandas DataFrame.
    output_dir: Directory where `cleaning_report.md` and `.json` will be saved.
    strict_mode: If True, drops rows violating severe temporal or business boundaries
                 instead of correcting them.

Returns:
    Tuple[pd.DataFrame, Dict[str, Any]]: (Cleaned DataFrame, Audit Report Dict)

#### Method: `_remove_duplicates`
Remove duplicate incident numbers or duplicate rows.

#### Method: `_enforce_data_types`
Enforce types per Feature Registry definitions.

#### Method: `_handle_missing_values`
Apply Feature Registry imputation strategies (`median`, `mode`, `constant_unknown`, `zero`).

#### Method: `_validate_and_clean_categories`
Validate categorical values against authorized ITSM catalogs and map invalid strings.

#### Method: `_validate_business_rules`
Enforce numeric ranges. Extract digits or map raw strings for priority (1-5) and severity (1-3).

#### Method: `_validate_timestamps`
Check opened_at <= resolved_at <= closed_at progression.

#### Method: `_handle_outliers`
Winsorize extreme numerical counters (`reassignment_count`, `reopen_count`).

#### Method: `_normalize_strings`
Strip leading/trailing whitespaces across string attributes.

#### Method: `_export_markdown_report`
Export formal markdown cleaning log.


### File: `src/preprocessing/eda.py`

**Module Docstring:**
Enterprise Exploratory Data Analysis (EDA) Engine (`src/preprocessing/eda.py`).

Performs comprehensive automated analysis of ServiceNow incident datasets across
Numerical, Categorical, Boolean, Datetime, and Text dimensions without hardcoded
column lists by consuming the Central Feature Registry via Pipeline Contracts.
Generates rich markdown, JSON, and HTML reports along with explanatory visual charts.

### Class: `EnterpriseEDAEngine`
Automated EDA & Feature Intelligence Engine governing ServiceNow data exploration.
Enforces contract validation, computes information-theoretic statistics, generates
diagnostic visual charts with explanations, and outputs formal executive reports.

#### Method: `__init__`
Initialize EDA engine with Registry and Contract Validator instances.

#### Method: `analyze_dataset`
Execute full automated exploration across all column types and generate reports.

Args:
    df: Input ServiceNow pandas DataFrame.
    target_column: Target label column name (`assignment_group`).
    output_dir: Base directory to save `eda_report.md`, `eda_report.json`, and figures.
    generate_figures: Whether to generate and save PNG visual charts.

Returns:
    Dict[str, Any]: Complete structured dictionary of EDA metrics and analysis results.

#### Method: `_get_known_text_cols`
Identify long-form unstructured text columns.

#### Method: `_compute_missing_report`
Compute exact missing value counts, percentages, and registry thresholds.

#### Method: `_compute_numerical_analysis`
Compute descriptive statistics, skewness, kurtosis, and IQR outlier bounds.

#### Method: `_compute_categorical_analysis`
Compute cardinality, top categories, Shannon Entropy, and Gini Impurity.

#### Method: `_compute_boolean_analysis`
Compute True/False/Null counts and percentages.

#### Method: `_compute_datetime_analysis`
Analyze temporal spans and hourly/daily arrival patterns.

#### Method: `_compute_text_analysis`
Compute character length, word count, and token estimations across text fields.

#### Method: `_compute_correlation_analysis`
Compute Pearson ($r$) and Spearman ($ho$) correlation matrices.

#### Method: `_compute_target_leakage_audit`
Cross-check dataset columns against Feature Registry leakage tiers.

#### Method: `_generate_visual_charts`
Generate high-resolution PNG charts with structured ML decision explanations.

#### Method: `_export_markdown_report`
Export executive markdown report with embedded charts and statistical tables.

#### Method: `_export_html_report`
Export self-contained HTML executive summary report.


### File: `src/preprocessing/engineer.py`

**Module Docstring:**
Enterprise Feature Engineering Engine (`src/preprocessing/engineer.py`).

Generates production-ready temporal, interaction, text-statistical, linkage, and
historical frequency features from cleaned incident datasets.
Automatically registers all newly created attributes inside the Central Feature Registry
and records exact mathematical/logical transformation graphs in the Feature Lineage Tracker.

### Class: `FeatureEngineeringEngine`
Automated feature generation and domain transformation engine.
Produces high-signal tabular features and guarantees synchronized governance
by updating FeatureRegistry and FeatureLineageTracker upon every transformation.

#### Method: `__init__`
Initialize engine with Registry and Lineage tracker singletons.

#### Method: `engineer_features`
Execute comprehensive feature engineering pipeline and sync governance registries.

Args:
    df: Cleaned input pandas DataFrame.
    output_dir: Directory where `feature_engineering_report.md` & `.json` will be saved.

Returns:
    Tuple[pd.DataFrame, Dict[str, Any]]: (Feature-engineered DataFrame, Engineering Report Dict)

#### Method: `_register_new_feature`
Helper to automatically sync FeatureRegistry and FeatureLineageTracker.

#### Method: `_generate_datetime_features`
Extract temporal components and cyclic encodings from opened_at.

#### Method: `_generate_holiday_and_business_features`
Compute business hours flag and US Federal/Banking Holiday indicator.

#### Method: `_generate_interaction_features`
Create high-signal cross-interactions between priority, impact, urgency, and categories.

#### Method: `_generate_resolution_features`
Compute exact resolution duration in hours and strictly classify as BLOCKED target leakage.

#### Method: `_generate_text_statistics`
Extract word count and character count metrics across description fields.

#### Method: `_generate_linkage_flags`
Create explicit 0/1 boolean linkage indicators.

#### Method: `_generate_frequency_encodings`
Encode high-cardinality categorical strings into normalized historical frequency probabilities.

#### Method: `_compute_importance_recommendations`
Recommend specific features for downstream model inputs and explain rationale.

#### Method: `_export_markdown_report`
Export formal markdown feature engineering audit report.


### File: `src/preprocessing/enricher.py`

**Module Docstring:**
Module for Data Enrichment from external sources (e.g. CMDB, Shift Schedules).

### Class: `EnterpriseDataEnricher`
Safely merges external context data into the incident dataframe if files exist.

#### Method: `__init__`
_No docstring provided._

#### Method: `enrich_dataset`
Look for optional external datasets (cmdb.csv, shift_schedules.csv) and merge them.


### File: `src/preprocessing/splitter.py`

**Module Docstring:**
Enterprise Dataset Splitter (`src/preprocessing/splitter.py`).

Divides cleaned and feature-engineered incident datasets into Train, Validation, and Test partitions
using Random, Stratified (`assignment_group`/`priority`), or Time-Based (`opened_at`) strategies.
Strictly verifies zero boundary leakage and exports formal split metadata and audit reports.

### Class: `DatasetSplitter`
Automated dataset partitioning engine ensuring zero data leakage and balanced
class representation across Train, Validation, and Test splits.

#### Method: `__init__`
Initialize DatasetSplitter with random seed and FeatureRegistry.

#### Method: `split_dataset`
Execute dataset partitioning and verify zero data leakage.

Args:
    df: Feature-engineered input DataFrame.
    strategy: Split strategy (`random`, `stratified`, `time_based`).
    target_column: Target class label (`assignment_group` or `priority`).
    train_ratio: Proportion for training set (`0.70`).
    val_ratio: Proportion for validation set (`0.15`).
    test_ratio: Proportion for test set (`0.15`).
    output_dir: Target directory for `train.csv`, `val.csv`, `test.csv`.
    report_dir: Directory for `split_report.md` & `.json`.

Returns:
    Tuple[DataFrame, DataFrame, DataFrame, Dict]: (Train, Val, Test, Audit Report)

#### Method: `_random_split`
Perform standard random shuffling split.

#### Method: `_stratified_split`
Perform stratified split ensuring proportional minority representation.

#### Method: `_time_based_split`
Sort chronologically by opened_at and partition sequentially.

#### Method: `_verify_zero_leakage`
Verify strict zero intersection of incident numbers across partitions.

#### Method: `_compute_class_distributions`
Calculate class distribution percentages across partitions.

#### Method: `_export_markdown_report`
Export executive markdown split report.


### File: `src/preprocessing/text_preprocessor.py`

**Module Docstring:**
Enterprise Text Preprocessing & Token Truncation Verification Engine (`src/preprocessing/text_preprocessor.py`).

Normalizes, cleans, filters stopwords, lemmatizes, and verifies token bounds across
unstructured ServiceNow incident text fields (`short_description`, `description`, `close_notes`)
to prepare clean textual inputs for upcoming semantic embedding 
models (`TF-IDF + SVD`).
Does NOT generate vector embeddings (reserved for Phase 3/4).

### Class: `TextPreprocessor`
Automated NLP normalization and token verification engine for ServiceNow incident data.
Ensures optimal semantic density and compliance with neural embedding token budgets.

#### Method: `__init__`
Initialize TextPreprocessor with PipelineContractValidator and token boundaries.

#### Method: `preprocess_dataset`
Execute full text normalization across unstructured columns and generate readiness audit.

Args:
    df: Input pandas DataFrame.
    output_dir: Directory to export `text_preprocessing_report.md` & `.json`.
    remove_stopwords: Whether to strip non-diagnostic English stopwords.
    lemmatize: Whether to apply IT domain suffix normalization.

Returns:
    Tuple[pd.DataFrame, Dict[str, Any]]: (DataFrame with `_clean` columns, Audit Report Dict)

#### Method: `normalize_text`
Apply complete normalization sequence to a single string.

#### Method: `_estimate_tokens`
Estimate BPE/WordPiece token count (`len(words) * 1.3`).

#### Method: `_compute_recommendations`
Compute actionable recommendations for downstream Sentence Transformer batching.

#### Method: `_export_markdown_report`
Export executive markdown text readiness report.


---

## Utilities

### File: `src/utils/config_manager.py`

**Module Docstring:**
Configuration Manager — Singleton YAML Configuration Loader.

Provides centralized, type-safe access to all application configuration.
Supports environment variable interpolation using ${VAR:default} syntax.

Design Decisions:
    - Singleton Pattern: One config instance shared across the entire application.
      Prevents redundant file I/O and ensures consistency.
    - Environment Variable Override: Secrets and environment-specific values
      (URLs, credentials) are injected via env vars, never stored in YAML.
    - Immutable After Load: Config is loaded once at startup. Explicit reload()
      is required to pick up changes (prevents mid-request config drift).
    - Dot-notation Access: config.get("ml.catboost.n_estimators") for
      clean, readable access to nested values.

Usage:
    from src.utils.config_manager import ConfigManager

    config = ConfigManager()
    db_host = config.get("database.host", default="localhost")
    ml_config = config.get_section("ml")

### Class: `ConfigManager`
Thread-safe Singleton configuration manager.

Loads YAML configuration files and provides dot-notation access
to nested configuration values with environment variable interpolation.

#### Method: `__new__`
Ensure only one instance exists (thread-safe Singleton).

#### Method: `get_instance`
Return the Singleton instance of ConfigManager.

#### Method: `__init__`
Initialize the configuration manager.

Args:
    config_dir: Path to the configuration directory.
                Defaults to 'config/' relative to project root.

#### Method: `_resolve_config_dir`
Resolve the configuration directory path.

#### Method: `_load_all_configs`
Load all YAML configuration files from the config directory.

#### Method: `_load_yaml`
Load and parse a YAML file.

Args:
    filepath: Path to the YAML file.

Returns:
    Parsed YAML content as a dictionary.

Raises:
    FileNotFoundError: If the file does not exist.
    yaml.YAMLError: If the YAML is malformed.

#### Method: `_resolve_env_vars`
Recursively resolve environment variable placeholders.

Supports ${VAR_NAME} and ${VAR_NAME:default_value} syntax.

Args:
    config: Configuration value (dict, list, or scalar).

Returns:
    Configuration with environment variables resolved.

#### Method: `get`
Get a configuration value using dot-notation.

Args:
    key: Dot-separated key path (e.g., "ml.catboost.n_estimators").
    default: Default value if key is not found.

Returns:
    The configuration value, or default if not found.

Examples:
    >>> config = ConfigManager()
    >>> config.get("app.name")
    'Incident Intelligence Platform'
    >>> config.get("ml.catboost.n_estimators", 100)
    200

#### Method: `get_section`
Get an entire configuration section as a dictionary.

Args:
    section: Dot-separated section path.

Returns:
    Configuration section as a dictionary, or empty dict.

#### Method: `has`
Check if a configuration key exists.

Args:
    key: Dot-separated key path.

Returns:
    True if the key exists, False otherwise.

#### Method: `all`
Return the complete configuration dictionary (read-only copy).

#### Method: `reload`
Reload all configuration files from disk.

Use sparingly — typically only for testing or admin operations.

#### Method: `reset`
Reset the Singleton instance.

Used exclusively in testing to ensure clean state between tests.

#### Method: `get_hybrid_config`
Get the hybrid recommendation engine configuration section (`ml.hybrid`).
Returns dictionary containing weights, bonuses, thresholds, and MTTR fusion settings.

#### Method: `__repr__`
_No docstring provided._


### File: `src/utils/logger.py`

**Module Docstring:**
Logging Factory — Enterprise Structured Logging Setup.

Provides centralized logging configuration for the entire application.
Supports both human-readable and JSON-structured log formats.

Design Decisions:
    - JSON Formatter: Enterprise systems require machine-parseable logs for
      log aggregation (Splunk, ELK). JSON format enables this without agents.
    - Rotating File Handlers: Prevents disk exhaustion on long-running services.
      10MB per file, 5 backups = max 60MB disk usage per log category.
    - Separate Log Files: app.log, ml_training.log, api_access.log, error.log
      allow independent monitoring and retention policies.
    - Module-based Loggers: Each module gets its own logger via __name__,
      enabling granular log level control per component.

Usage:
    from src.utils.logger import LoggerFactory

    logger = LoggerFactory.get_logger(__name__)
    logger.info("Model training started", extra={"model": "rf_v1", "samples": 10000})

### Class: `JsonFormatter`
JSON log formatter for structured logging.

Produces one JSON object per log line, suitable for ingestion by
enterprise log aggregation systems (Splunk, ELK, Datadog).

#### Method: `format`
Format a log record as a JSON string.

Args:
    record: The log record to format.

Returns:
    JSON-formatted log string.

### Class: `LoggerFactory`
Factory for creating and configuring application loggers.

Loads logging configuration from YAML and ensures log directories
exist before any handlers attempt to write.

#### Method: `configure`
Configure the logging system from a YAML configuration file.

This should be called once at application startup. Subsequent calls
are no-ops unless force=True.

Args:
    config_path: Path to logging.yaml. Auto-detected if None.
    log_dir: Override log directory. Uses config value if None.

#### Method: `get_logger`
Get a named logger, configuring the system if not yet done.

Args:
    name: Logger name (typically __name__ of the calling module).

Returns:
    Configured logging.Logger instance.

Usage:
    logger = LoggerFactory.get_logger(__name__)
    logger.info("Processing started")

#### Method: `_find_config`
Locate the logging configuration file.

#### Method: `_configure_from_yaml`
Load logging configuration from YAML file.

Resolves relative log file paths against the project root.

#### Method: `_configure_fallback`
Configure basic logging as a fallback.

#### Method: `reset`
Reset the logging configuration state.

Used exclusively in testing to ensure clean state between tests.

### Function: `get_logger`
Convenience function for getting a configured logger.

This is the primary entry point for logging throughout the application.

Args:
    name: Logger name (typically __name__).

Returns:
    Configured logging.Logger instance.

Usage:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)


---

