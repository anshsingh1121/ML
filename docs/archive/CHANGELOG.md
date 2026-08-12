# Changelog

All notable changes to the Incident Intelligence Platform will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.5.0-alpha] - 2026-07-11

### Added

- **Dataset Validation Framework (`src/data/validation.py`)**: Automated OOP quality checks covering 12 enterprise domains (missing values, duplicate IDs, timestamp sequence, SLA mechanics, cmdb_ci references, short/full descriptions) with dual JSON & Markdown reporting (`reports/validation_report.*`).
- **ML Readiness Evaluation Suite (`src/data/readiness.py`)**: Pre-training diagnostic engine computing Shannon Entropy, Gini Impurity, target leakage detection, categorical imbalance ratios, and text token length distributions against SentenceTransformer sequence boundaries (`tfidf-svd-384`).
- **Immutable Dataset Versioning (`src/data/version_manager.py`)**: Automated non-destructive directory allocation (`datasets/synthetic/v1/`, `v2/`, etc.) with comprehensive `metadata.json` manifests and centralized audit tracking (`version_history.json`).
- **Standardized Banking Benchmarks (`src/data/benchmark.py`)**: One-click benchmark generation suite supporting 5 enterprise scale profiles (`Small` 10K, `Medium` 50K, `Large` 100K, `Enterprise` 500K, `XL` 1M) with integrated validation and readiness certification.
- **Enterprise Quality Gate Runner (`src/data/quality_gate.py`)**: 6-domain governance gatekeeper verifying configuration, batch scripts, documentation, schema, quality, and ML readiness prior to Phase 2 transitions (`reports/quality_gate_certification.*`).
- **Comprehensive Data Dictionary & Feature Catalog**: Added `docs/data_dictionary.md` specifying all 38 ServiceNow incident attributes, and `docs/feature_catalog.md` detailing encoding strategies and target leakage boundaries.
- **Feature Pipeline Architecture (`projects/feature_pipeline_v1.md`)**: High-fidelity Mermaid diagram illustrating end-to-end flow from raw ingestion through feature stores to hybrid similarity search.

### Changed

- **SLA Calculation Precision**: Refined log-normal sampling inside `DatasetGenerator` (`src/data/dataset_generator.py`) to round `resolution_times` before SLA compliance evaluation, eliminating floating-point threshold discrepancies.

## [1.0.0-alpha] - 2026-07-10

### Added

- **Incident Classification Engine**: Multi-target Random Forest models for assignment group prediction, category classification, priority assessment, and resolution time estimation.
- **Semantic Similarity Search**: Sentence Transformer embeddings (tfidf-svd-384) with FAISS vector store for high-performance incident matching and retrieval.
- **Resolution Recommendation Engine**: RAG-based resolution retrieval combining semantic search with historical resolution data to suggest actionable remediation steps.
- **SHAP-based Model Explainability**: Integrated SHAP explanations for all classification predictions, providing transparent and interpretable decision support.
- **Interactive TF-IDF Dashboard**: Multi-page analytics dashboard with real-time incident insights, model performance metrics, similarity explorer, and operational reporting.
- **TF-IDF REST API**: Production-grade API with Pydantic validation, structured routing, middleware support, and comprehensive endpoint documentation.
- **Data Pipeline**: Automated data ingestion from ServiceNow with configurable field mapping, batch processing, and CSV export capabilities.
- **Preprocessing Pipeline**: Feature engineering for text, categorical, and numerical features with configurable transformations and target column support.
- **Hyperparameter Optimization**: Randomized search with cross-validation for automated model tuning across all classification targets.
- **Comprehensive Test Suite**: Unit and integration test framework for end-to-end validation of all platform components.
- **Modular Package Architecture**: SOLID-compliant package structure with clear separation of concerns across API, ML, data, preprocessing, resolution, dashboard, reporting, and utility modules.
- **Configuration Management**: YAML-based configuration for model hyperparameters, ServiceNow connectivity, and platform settings with environment variable support.

### Changed

- N/A (initial release)

### Deprecated

- N/A (initial release)

### Removed

- N/A (initial release)

### Fixed

- N/A (initial release)

### Security

- Environment variable-based credential management for ServiceNow integration (no hardcoded secrets).
- SSL verification enabled by default for all external API connections.
- Local-only execution architecture to comply with banking data privacy and regulatory requirements.
