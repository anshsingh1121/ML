#!/usr/bin/env python
"""
Enterprise Incident Intelligence Platform (IIP) — First Citizens Bank (`v2.0.0-alpha`).
Root executable entry point (`main.py`).

Provides interactive menu interface when invoked without arguments (`python main.py`)
or direct subcommand automation (`python main.py generate/validate/eda/clean/engineer/split/pipeline/status/train/evaluate/explain/models/predict/embed/index/similar`).
"""

import argparse
import sys
from pathlib import Path

# Add workspace root to python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.cli.main_cli import EnterpriseCLI
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> int:
    """Parse command line arguments and execute platform CLI."""
    parser = argparse.ArgumentParser(
        description="First Citizens Bank — AI-Powered Incident Intelligence Platform (`v2.0.0-alpha`)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                                           # Launch interactive menu
  python main.py status                                    # Check platform architecture & health
  python main.py train --target assignment_group --hpo     # Run HPO & train classification model
  python main.py evaluate --target assignment_group        # Evaluate classification pipeline
  python main.py explain --target assignment_group         # Generate SHAP explainability plots
  python main.py models                                    # Audit registered SHA256 model catalog
  python main.py predict --input sample.json               # Run zero-leakage inference & SHAP
  python main.py embed                                     # Generate local embeddings (tfidf-svd-384)
  python main.py index                                     # Build and register FAISS vector index
  python main.py similar --incident INC0012345             # Find Top-K similar historical incidents
  python main.py similar --text "ATM withdrawal failing"   # Semantic search from free text
"""
    )

    subparsers = parser.add_subparsers(dest="command", help="Operational Subcommands")

    # 1. menu / status
    subparsers.add_parser("menu", help="Launch interactive menu (1-16)")
    subparsers.add_parser("status", help="Check system readiness and registry status")



    # 3. validate / readiness
    val_parser = subparsers.add_parser("validate", help="Run dataset validation checks")
    val_parser.add_argument("--input", type=str, default="data/raw/incidents.csv", help="Input CSV path")

    read_parser = subparsers.add_parser("readiness", help="Run ML readiness checks")
    read_parser.add_argument("--input", type=str, default="data/raw/incidents.csv", help="Input CSV path")

    # 4. eda
    eda_parser = subparsers.add_parser("eda", help="Run automated Exploratory Data Analysis engine")
    eda_parser.add_argument("--input", type=str, default="data/raw/incidents.csv", help="Input CSV path")
    eda_parser.add_argument("--output-dir", type=str, default="reports", help="Output directory for reports & figures")

    # 5. clean
    clean_parser = subparsers.add_parser("clean", help="Run enterprise data cleaner and remediation")
    clean_parser.add_argument("--input", type=str, default="data/raw/incidents.csv", help="Input CSV path")
    clean_parser.add_argument("--output", type=str, default="data/processed/cleaned_incidents.csv", help="Cleaned CSV output path")
    clean_parser.add_argument("--strict", action="store_true", help="Enable strict mode (drops out-of-bounds rows)")

    # 6. engineer
    eng_parser = subparsers.add_parser("engineer", help="Run feature engineering engine")
    eng_parser.add_argument("--input", type=str, default="data/processed/cleaned_incidents.csv", help="Cleaned input CSV path")
    eng_parser.add_argument("--output", type=str, default="data/processed/engineered_incidents.csv", help="Engineered CSV output path")

    # 7. split
    split_parser = subparsers.add_parser("split", help="Partition dataset into Train/Val/Test with zero leakage")
    split_parser.add_argument("--input", type=str, default="data/processed/engineered_incidents.csv", help="Engineered input CSV path")
    split_parser.add_argument("--strategy", type=str, default="stratified", choices=["random", "stratified", "time_based"], help="Splitting strategy")
    split_parser.add_argument("--target", type=str, default="assignment_group", help="Target column for stratification")
    split_parser.add_argument("--output-dir", type=str, default="data/processed", help="Directory to save train.csv, val.csv, test.csv")

    # 8. pipeline
    pipe_parser = subparsers.add_parser("pipeline", help="Run complete end-to-end Data Intelligence Pipeline")
    pipe_parser.add_argument("--input", type=str, default="data/raw/incidents.csv", help="Input CSV path")
    pipe_parser.add_argument("--output-dir", type=str, default="data/processed", help="Directory to save processed splits and master dataset")
    pipe_parser.add_argument("--all", action="store_true", help="Execute all pipeline stages automatically")

    # 9. train (Phase 3)
    train_parser = subparsers.add_parser("train", help="Train Random Forest & baseline ML models")
    train_parser.add_argument("--target", type=str, default="assignment_group", choices=["assignment_group", "resolution_time_hours", "category", "priority"], help="Target variable to train")
    train_parser.add_argument("--hpo", action="store_true", help="Run Hyperparameter Optimization before training")
    train_parser.add_argument("--compare-baselines", action="store_true", default=True, help="Compare Decision Tree, Random Forest, Extra Trees, XGBoost, LightGBM")
    train_parser.add_argument("--train-data", type=str, default="data/processed/train.csv", help="Path to training partition CSV")
    train_parser.add_argument("--val-data", type=str, default="data/processed/val.csv", help="Path to validation partition CSV")

    # 10. evaluate (Phase 3)
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate trained model pipeline and generate charts")
    eval_parser.add_argument("--model-key", type=str, default="catboost_assignment_group:latest", help="Model key (`name:version`) or absolute file path")
    eval_parser.add_argument("--test-data", type=str, default="data/processed/test.csv", help="Path to test partition CSV")
    eval_parser.add_argument("--target", type=str, default="assignment_group", help="Target variable evaluated")

    # 11. explain (Phase 3)
    exp_parser = subparsers.add_parser("explain", help="Run SHAP TreeExplainer diagnostics and generate summary plots")
    exp_parser.add_argument("--model-key", type=str, default="catboost_assignment_group:latest", help="Model key (`name:version`) or absolute file path")
    exp_parser.add_argument("--input", type=str, default="data/processed/test.csv", help="Input dataset for SHAP background/samples")
    exp_parser.add_argument("--target", type=str, default="assignment_group", help="Target variable explained")

    # 12. models (Phase 3)
    subparsers.add_parser("models", help="Audit Central Model Registry catalog and SHA256 checksums")

    # 13. predict (Phase 3)
    pred_parser = subparsers.add_parser("predict", help="Execute zero-manual-preprocessing inference and export prediction metadata")
    pred_parser.add_argument("--input", type=str, required=True, help="Path to input JSON payload or CSV file")
    pred_parser.add_argument("--model-key", type=str, default="catboost_assignment_group:latest", help="Model key (`name:version`) or file path")
    pred_parser.add_argument("--target", type=str, default="assignment_group", help="Target variable predicted")

    # 14. embed (Phase 4)
    embed_parser = subparsers.add_parser("embed", help="Generate local neural embeddings (`tfidf-svd-384`) from incident dataset")
    embed_parser.add_argument("--input", type=str, default="data/processed/train.csv", help="Input historical dataset CSV path")
    embed_parser.add_argument("--batch-size", type=int, default=64, help="Batch size for neural embedding inference")

    # 15. index (Phase 4)
    index_parser = subparsers.add_parser("index", help="Build and register FAISS vector similarity index")
    index_parser.add_argument("--input", type=str, default="data/processed/train.csv", help="Input dataset CSV path for index generation")
    index_parser.add_argument("--index-name", type=str, default="incident_semantic_index", help="Name of the registered FAISS index")

    # 16. similar (Phase 4)
    sim_parser = subparsers.add_parser("similar", help="Retrieve Top-K semantically similar historical incidents")
    sim_parser.add_argument("--incident", type=str, help="Historical Incident Number (e.g. INC0012345) to query")
    sim_parser.add_argument("--text", type=str, help="Free natural language query text (e.g. 'ATM cash withdrawal failing')")
    sim_parser.add_argument("--top-k", type=int, default=10, help="Number of similar incidents to retrieve")
    # 17. recommend (Phase 5)
    rec_parser = subparsers.add_parser("recommend", help="Run deterministic Hybrid Incident Intelligence Engine recommendation")
    rec_parser.add_argument("--input", type=str, help="Path to input JSON payload or CSV file")
    rec_parser.add_argument("--text", type=str, help="Free natural language query text (e.g. 'ATM cash withdrawal failing')")
    rec_parser.add_argument("--top-k", type=int, default=5, help="Number of semantic precedents to retrieve")

    # 18. full-pipeline (Phase 7 Enterprise Packaging)
    full_parser = subparsers.add_parser("full-pipeline", help="Execute complete end-to-end Enterprise Incident Intelligence Pipeline")
    full_parser.add_argument("--input", type=str, default="data/raw/incidents.csv", help="Input raw CSV path for pipeline processing")

    # 19. clean-workspace (Phase 7 Enterprise Packaging)
    subparsers.add_parser("clean-workspace", help="Clean generated runtime artifacts while preserving source files and directory structure")

    args = parser.parse_args()

    cli = EnterpriseCLI()
    return cli.run_command(args)


if __name__ == "__main__":
    sys.exit(main())

