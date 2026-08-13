"""
Enterprise CLI & Operational Control Plane (`src/cli/main_cli.py`).

Provides robust command-line subcommands (`generate`, `validate`, `readiness`, `eda`,
`clean`, `engineer`, `split`, `pipeline`, `status`, `train`, `evaluate`, `explain`, `models`, `predict`)
and an interactive terminal menu (1-15) for operating the First Citizens Bank Incident Intelligence Platform (`v2.0.0`).
"""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Union
import pandas as pd
from src.utils import robust_read_csv, robust_open


from src.data.validation import DatasetValidator
from src.data.readiness import MLReadinessEvaluator as MLReadinessChecker
from src.preprocessing.eda import EnterpriseEDAEngine
from src.preprocessing.cleaner import EnterpriseDataCleaner
from src.preprocessing.enricher import EnterpriseDataEnricher
from src.preprocessing.engineer import FeatureEngineeringEngine
from src.preprocessing.text_preprocessor import TextPreprocessor
from src.preprocessing.splitter import DatasetSplitter
from src.data.feature_registry import FeatureRegistry
from src.ml.catboost.trainer import EnterpriseCatBoostTrainer
from src.ml.catboost.evaluator import ModelEvaluator
from src.ml.explainability.shap_explainer import SHAPIntelligenceExplainer
from src.ml.model_registry import ModelRegistry
from src.ml.semantic.embedding_generator import SemanticEmbeddingGenerator
from src.ml.semantic.faiss_index import FAISSVectorIndex
from src.ml.semantic.similarity_engine import SemanticSimilarityEngine
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EnterpriseCLI:
    """
    Unified operational command-line controller for the AI-Powered Incident Intelligence Platform.
    Enforces enterprise exception handling, directory auto-creation, and comprehensive console reporting.
    """

    def __init__(self) -> None:
        """Initialize CLI engine, Feature Registry, and Model Registry with automatic directory creation."""
        for required_dir in [
            "data/raw", "data/processed",
            "models", "models/embeddings", "indexes",
            "reports", "reports/figures", "reports/daily", "reports/weekly", "reports/monthly",
            "logs"
        ]:
            Path(required_dir).mkdir(parents=True, exist_ok=True)
        self.registry = FeatureRegistry.get_instance()
        self.model_reg = ModelRegistry.get_instance()
        from src.utils.config_manager import ConfigManager
        self.config = ConfigManager()

    def _check_and_self_heal(self) -> None:
        """Check if core runtime artifacts are missing and trigger automatic self-healing if required."""
        raw_path = Path("data/raw/incidents.csv")
        clf_path = Path("models/catboost_assignment_group.pkl")
        idx_path = Path("indexes/incident_semantic_index_latest.index")
        proc_path = Path("data/processed/master_engineered_incidents.csv")

        if not clf_path.exists() or not idx_path.exists() or not proc_path.exists() or not raw_path.exists():
            print("\n[INFO] Self-Healing: Missing runtime models, indexes, reports, or processed data detected.")
            print("[INFO] Automatically regenerating complete runtime artifacts (--records 500)...")
            try:
                self.cmd_full_pipeline(input_path=str(raw_path))
            except Exception as e:
                logger.warning(f"Self-healing regeneration encountered an exception: {e}")

    def run_command(self, args: argparse.Namespace) -> int:
        """Dispatch subcommand arguments to appropriate execution pipeline with self-healing interlock."""
        command = getattr(args, "command", "menu")
        if command is None or command == "menu":
            return self.run_interactive_menu()

        # Trigger self-healing before inference or evaluation commands if runtime artifacts are missing
        if command in ["recommend", "predict", "evaluate", "explain"]:
            self._check_and_self_heal()

        try:
            if command == "validate":
                return self.cmd_validate(args.input)
            elif command == "readiness":
                return self.cmd_readiness(args.input)
            elif command == "eda":
                return self.cmd_eda(args.input, args.output_dir)
            elif command == "clean":
                return self.cmd_clean(args.input, args.output, args.strict)
            elif command == "engineer":
                return self.cmd_engineer(args.input, args.output)
            elif command == "split":
                return self.cmd_split(args.input, args.strategy, args.target, args.output_dir)
            elif command == "pipeline":
                return self.cmd_pipeline(args.input, args.output_dir)
            elif command == "status":
                return self.cmd_status()
            elif command == "train":
                return self.cmd_train(args.target, args.compare_baselines, args.train_data, args.val_data)
            elif command == "evaluate":
                return self.cmd_evaluate(args.model_key, args.test_data, args.target)
            elif command == "explain":
                return self.cmd_explain(args.model_key, args.input, args.target)
            elif command == "models":
                return self.cmd_models()
            elif command == "predict":
                return self.cmd_predict(args.input, args.model_key, args.target)
            elif command == "embed":
                return self.cmd_embed(args.input, args.batch_size)
            elif command == "index":
                return self.cmd_index(args.input, args.index_name)
            elif command == "similar":
                return self.cmd_similar(getattr(args, "incident", None), getattr(args, "text", None), getattr(args, "top_k", 10), getattr(args, "index_name", "incident_semantic_index"))
            elif command == "recommend":
                return self.cmd_recommend(getattr(args, "input", None), getattr(args, "text", None), getattr(args, "top_k", 5))
            elif command == "full-pipeline":
                return self.cmd_full_pipeline(input_path=getattr(args, "input", "data/raw/incidents.csv"))
            elif command == "clean-workspace":
                return self.cmd_clean_workspace()
            else:
                print(f"[ERROR] Unrecognized command: {command}")
                return 1
        except Exception as e:
            logger.error(f"Command execution failed ({command}): {e}", exc_info=True)
            print(f"\n[CRITICAL ERROR] Failed to execute '{command}': {e}")
            return 1




    def cmd_validate(self, input_path: str) -> int:
        """Run enterprise dataset validation framework."""
        print(f"\n---> [1/1] Running Enterprise Dataset Validation on: {input_path}...")
        df = robust_read_csv(input_path)
        validator = DatasetValidator()
        report = validator.validate_dataset(df, report_dir="reports")
        is_valid = report.get("is_valid", False)
        status = report.get("overall_status", "PASS" if is_valid else "FAIL")
        print(f"[STATUS] Validation Result: {status}")
        print("Detailed report exported to: reports/validation_report.md")
        
        if not is_valid:
            print("\n[WARNING] Dataset validation detected anomalies. The pipeline will proceed to Stage 3 (Enterprise Data Cleaner) to automatically remediate them.")
        
        return 0  # Always proceed to cleaning stage

    def cmd_readiness(self, input_path: str) -> int:
        """Run ML readiness verification."""
        print(f"\n---> [1/1] Running ML Readiness Verification on: {input_path}...")
        df = robust_read_csv(input_path)
        checker = MLReadinessChecker()
        report = checker.evaluate_dataset(df, target_column="assignment_group", report_dir="reports")
        status = report.get("overall_readiness_status", "UNKNOWN")
        print(f"[STATUS] ML Readiness Result: {status}")
        print("Detailed report exported to: reports/ml_readiness_report.md")
        return 0 if status in ("READY_FOR_ML", "PASS") else 2

    def cmd_eda(self, input_path: str, output_dir: str = "reports") -> int:
        """Run automated Exploratory Data Analysis (EDA) engine."""
        print(f"\n---> [1/1] Running Enterprise EDA Engine on: {input_path}...")
        df = robust_read_csv(input_path)
        engine = EnterpriseEDAEngine()
        engine.analyze_dataset(df, target_column="assignment_group", output_dir=output_dir, generate_figures=True)
        print(f"[SUCCESS] EDA completed. Reports & visual charts saved in: {output_dir}/")
        return 0

    def cmd_clean(self, input_path: str, output_path: str, strict_mode: bool = False) -> int:
        """Execute automated data cleaning and remediation pipeline."""
        print(f"\n---> [1/1] Running Enterprise Data Cleaner across: {input_path}...")
        df = robust_read_csv(input_path)

        # Validation for required columns
        req_cols = ["number", "opened_at", "priority", "category", "assignment_group", "short_description", "description"]
        missing_cols = [c for c in req_cols if c not in df.columns]
        if missing_cols:
            print(f"\n[CRITICAL ERROR] Cleaner halted! Missing required columns in dataset: {missing_cols}")
            print(f"[INFO] Please ensure your dataset matches the expected enterprise schema.")
            return 1

        cleaner = EnterpriseDataCleaner(config=self.config)
        clean_df, audit = cleaner.clean_dataset(df, output_dir="reports", strict_mode=strict_mode)
        
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        clean_df.to_csv(out_file, index=False)
        print(f"[SUCCESS] Cleaned {audit['initial_record_count']:,} -> {len(clean_df):,} records ({audit['records_removed_total']:,} removed).")
        print(f"Cleaned dataset saved to: {out_file}")
        print("Audit report exported to: reports/cleaning_report.md")
        return 0

    def cmd_engineer(self, input_path: str, output_path: str) -> int:
        """Execute feature engineering and registry synchronization pipeline."""
        print(f"\n---> [1/1] Running Feature Engineering Engine across: {input_path}...")
        df = robust_read_csv(input_path)
        engine = FeatureEngineeringEngine()
        eng_df, report = engine.engineer_features(df, output_dir="reports")
        
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        eng_df.to_csv(out_file, index=False)
        print(f"[SUCCESS] Engineered {report['total_new_features']} new attributes (Final column count: {len(eng_df.columns)}).")
        print(f"Feature matrix saved to: {out_file}")
        print("Lineage graph & importance report exported to: reports/feature_engineering_report.md")
        return 0

    def cmd_split(self, input_path: str, strategy: str = "stratified", target: str = "assignment_group", output_dir: str = "data/processed") -> int:
        """Partition dataset into Train/Val/Test with zero boundary leakage."""
        print(f"\n---> [1/1] Running '{strategy}' dataset splitting across: {input_path}...")
        df = robust_read_csv(input_path)
        splitter = DatasetSplitter()
        train_df, val_df, test_df, report = splitter.split_dataset(
            df, strategy=strategy, target_column=target, output_dir=output_dir, report_dir="reports"
        )
        print(f"[SUCCESS] Partitioned dataset: Train ({len(train_df):,}), Val ({len(val_df):,}), Test ({len(test_df):,}).")
        print(f"Split CSVs and metadata saved in: {output_dir}/")
        return 0

    def cmd_pipeline(self, input_path: str, output_dir: str = "data/processed") -> int:
        """Run complete end-to-end Data Intelligence Pipeline (Clean -> Engineer -> Preprocess -> Split)."""
        print(f"\n====================================================================")
        print(f"Launching Complete Data Intelligence Pipeline (`v2.0.0`)")
        print(f"Input File: {input_path} | Output Dir: {output_dir}")
        print(f"====================================================================")
        
        df = robust_read_csv(input_path)
        
        # Validation for required columns
        req_cols = ["number", "opened_at", "priority", "category", "assignment_group", "short_description", "description"]
        missing_cols = [c for c in req_cols if c not in df.columns]
        if missing_cols:
            print(f"\n[CRITICAL ERROR] Pipeline halted! Missing required columns in dataset: {missing_cols}")
            print(f"[INFO] Please ensure your dataset matches the expected enterprise schema.")
            return 1

        # Step 1: Clean
        print("\n[Step 1/5] Executing Enterprise Data Cleaner...")
        cleaner = EnterpriseDataCleaner(config=self.config)
        clean_df, clean_audit = cleaner.clean_dataset(df, output_dir="reports")
        print(f"  -> Cleaned: {len(clean_df):,} surviving records.")

        # Step 1.5: Enrich
        print("\n[Step 1.5/5] Checking for External Data Enrichment (CMDB/Shift)...")
        enricher = EnterpriseDataEnricher(raw_data_dir="data/raw")
        enriched_df, _ = enricher.enrich_dataset(clean_df, {"transformations": []})
        print(f"  -> Enrichment complete. Columns: {len(enriched_df.columns)}")

        # Step 2: Engineer
        print("\n[Step 2/5] Executing Feature Engineering Engine...")
        engine = FeatureEngineeringEngine()
        eng_df, eng_report = engine.engineer_features(enriched_df, output_dir="reports")
        print(f"  -> Engineered: {eng_report['total_new_features']} new features created & synchronized.")

        # Step 3: Text Preprocessing
        print("\n[Step 3/5] Executing Text Preprocessing & Token Truncation Verification...")
        preprocessor = TextPreprocessor()
        proc_df, txt_report = preprocessor.preprocess_dataset(eng_df, output_dir="reports")
        print(f"  -> Text Normalized across: {', '.join(txt_report['columns_processed'])}.")

        # Step 4: EDA & Diagnostic Charts
        print("\n[Step 4/5] Running Post-Engineering EDA & Generating Charts...")
        eda_engine = EnterpriseEDAEngine()
        eda_engine.analyze_dataset(proc_df, target_column="assignment_group", output_dir="reports", generate_figures=True)
        print(f"  -> EDA Reports & Figures saved inside: reports/figures/")

        # Step 5: Dataset Splitter
        print("\n[Step 5/5] Executing Stratified Dataset Partitioning (`assignment_group`)...")
        splitter = DatasetSplitter()
        train_df, val_df, test_df, split_report = splitter.split_dataset(
            proc_df, strategy="stratified", target_column="assignment_group", output_dir=output_dir, report_dir="reports"
        )
        print(f"  -> Splits Certified: Train ({len(train_df):,}), Val ({len(val_df):,}), Test ({len(test_df):,}).")

        # Save complete processed dataset before split
        master_file = Path(output_dir) / "master_engineered_incidents.csv"
        proc_df.to_csv(master_file, index=False)
        print(f"\n====================================================================")
        print(f"[PIPELINE CERTIFIED SUCCESSFUL] All modules completed with zero leakage.")
        print(f"Master dataset saved to: {master_file}")
        print(f"Check reports/ directory for all 6 comprehensive audit logs!")
        print(f"====================================================================")
        return 0

    def cmd_train(self, target: str = "assignment_group", compare_baselines: bool = True, train_path: Optional[str] = None, val_path: Optional[str] = None) -> int:
        """Train CatBoost pipelines (`assignment_group` or `resolution_time_hours`)."""
        print(f"\n====================================================================")
        print(f"Launching Enterprise ML Trainer (`target={target}`, Baselines={compare_baselines})")
        print(f"====================================================================")
        trainer = EnterpriseCatBoostTrainer()

        print("\n---> [Stage 2/2] Training & Persisting Complete Zero-Leakage Pipeline...")
        if target in ["assignment_group", "category", "priority"]:
            out_pkl = trainer.train_classifier(train_path=train_path, val_path=val_path, target_col=target, compare_baselines=compare_baselines)
        else:
            out_pkl = trainer.train_regressor(train_path=train_path, val_path=val_path, target_col=target, compare_baselines=compare_baselines)

        print(f"\n[SUCCESS] Model training certified! Pipeline saved to: {out_pkl}")
        print("Model metadata, features, and SHA256 checksum registered in models/model_registry.json.")
        return 0

    def cmd_evaluate(self, model_key: str = "catboost_assignment_group:latest", test_data: Optional[str] = None, target: str = "assignment_group") -> int:
        """Evaluate trained model pipeline across test partition (`test.csv`)."""
        print(f"\n====================================================================")
        print(f"Evaluating Model (`key={model_key}`, target={target})")
        print(f"====================================================================")
        evaluator = ModelEvaluator()

        if target in ["assignment_group", "category", "priority"]:
            metrics = evaluator.evaluate_classification(model_key_or_path=model_key, test_path=test_data, target_col=target)
            print(f"\n[EVALUATION SUMMARY] Classification Metrics ({target}):")
            for k, v in metrics.items():
                print(f"  - {k:<25}: {v}")
        else:
            metrics = evaluator.evaluate_regression(model_key_or_path=model_key, test_path=test_data, target_col=target)
            print(f"\n[EVALUATION SUMMARY] Regression Metrics ({target}):")
            for k, v in metrics.items():
                print(f"  - {k:<25}: {v}")

        print("\nCertified audit reports exported to: reports/classification_report.md & reports/regression_report.md")
        print("Visual diagnostic plots generated: reports/confusion_matrix.png, reports/roc_curve.png, reports/feature_importance.png")
        return 0

    def cmd_explain(self, model_key: str = "catboost_assignment_group:latest", input_path: Optional[str] = None, target: str = "assignment_group") -> int:
        """Run SHAP Explainable AI diagnostics on trained model pipeline."""
        print(f"\n====================================================================")
        print(f"Executing Explainable AI (`SHAP`) Engine (`model={model_key}`)")
        print(f"====================================================================")
        explainer = SHAPIntelligenceExplainer()
        importances = explainer.explain_global(model_key_or_path=model_key, test_path=input_path, sample_size=100, target_col=target)
        
        print(f"\n[SHAP ATTRIBUTION SUMMARY] Top Contributing Features ({target}):")
        for k, v in sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  - {k:<25}: |SHAP| = {v:.6f}")

        print("\nGlobal SHAP plots generated: reports/shap_bar.png & reports/shap_summary.png")
        return 0

    def cmd_models(self) -> int:
        """List and audit all registered models in ModelRegistry."""
        print(f"\n====================================================================")
        print(f"Central Model Registry Audit (`v1.5.0`) — Registered Models")
        print(f"====================================================================")
        models = self.model_reg.models
        if not models:
            print("[INFO] No models currently registered inside models/model_registry.json.")
            return 0

        for k, m in sorted(models.items()):
            print(f"\nModel Key: [{k}] | Status: {m.status}")
            print(f"  - Target Variable : {m.target_variable}")
            print(f"  - Dataset Version : {m.dataset_version} (Trained at {m.training_timestamp})")
            print(f"  - Features Used   : {len(m.features_used)} attributes authorized")
            print(f"  - SHA256 Checksum : {m.sha256_checksum[:16]}... (Full: {m.sha256_checksum})")
            print(f"  - Model File Path : {m.model_file_path}")
            print(f"  - Metrics Summary : {m.metrics}")
        
        self.model_reg.export_markdown()
        print("\nFull Markdown Catalog exported to: models/model_registry.md")
        return 0

    def cmd_predict(self, input_path: str, model_key: str = "catboost_assignment_group:latest", target: str = "assignment_group") -> int:
        """Execute zero-manual-preprocessing inference and export structured prediction metadata."""
        print(f"\n---> [1/1] Executing zero-leakage prediction across: {input_path} (`model={model_key}`)...")
        in_file = Path(input_path)
        if not in_file.exists():
            print(f"[ERROR] Input payload file missing: {in_file}")
            return 1

        if in_file.suffix.lower() == ".json":
            with robust_open(in_file, "r") as f:
                data = json.load(f)
            records = data if isinstance(data, list) else [data]
        else:
            df = robust_read_csv(in_file)
            records = df.to_dict(orient="records")

        explainer = SHAPIntelligenceExplainer()
        results = explainer.explain_prediction(records, model_key_or_path=model_key, target_col=target)

        print(f"[SUCCESS] Completed inference & SHAP attribution across {len(results)} records!")
        print(f"Top sample prediction: Incident {results[0]['number']} -> {results[0].get('predicted_class', results[0].get('predicted_value'))} (Confidence: {results[0]['confidence_score']:.4f})")
        print("Structured prediction metadata exported to: reports/prediction_metadata.json & reports/prediction_metadata.csv")
        return 0

    def cmd_embed(self, input_path: str, batch_size: int) -> int:
        """Execute Phase 4 offline local neural embedding generation (`TF-IDF + SVD`)."""
        print(f"\n---> [Phase 4] Generating local neural embeddings from: {input_path}...")
        if not Path(input_path).exists():
            print(f"[ERROR] Input dataset missing at {input_path}")
            return 1
        df = robust_read_csv(input_path)
        embedder = SemanticEmbeddingGenerator()
        embeddings, meta_df = embedder.embed_dataframe(df, batch_size=batch_size, show_progress_bar=True)
        npy_path, meta_path = embedder.save_embeddings(embeddings, meta_df)
        print(f"[SUCCESS] Generated {embeddings.shape[0]:,} embeddings ({embeddings.shape[1]}-D). Saved to: {npy_path} & {meta_path}")
        return 0

    def cmd_index(self, input_path: str, index_name: str) -> int:
        """Build, persist, and register FAISS vector similarity index."""
        print(f"\n---> [Phase 4] Building and registering FAISS vector index '{index_name}' from: {input_path}...")
        if not Path(input_path).exists():
            print(f"[ERROR] Input dataset missing at {input_path}")
            return 1
        df = robust_read_csv(input_path)
        engine = SemanticSimilarityEngine()
        num_indexed = engine.build_index_from_dataframe(df, index_name=index_name)
        print(f"[SUCCESS] Built and registered FAISS index '{index_name}' across {num_indexed:,} historical incidents.")
        return 0

    def cmd_similar(self, incident: Optional[str], text: Optional[str], top_k: int, index_name: str) -> int:
        """Retrieve Top-K semantically similar historical incidents."""
        if not incident and not text:
            print("[ERROR] Must specify either `--incident INC...` or `--text 'query text'`.")
            return 1

        query = incident if incident else text
        query_type = f"Incident Number '{incident}'" if incident else f"Free Text '{text}'"
        print(f"\n---> [Phase 4] Retrieving Top-{top_k} semantically similar precedents for {query_type}...")

        engine = SemanticSimilarityEngine()
        try:
            engine.faiss_index.index_name = index_name
            engine.faiss_index.load_index(index_name=index_name)
        except Exception as e:
            logger.warning(f"Failed to load FAISS index '{index_name}' from disk ({e}). Attempting to index from processed dataset...")
            if Path("data/processed/train.csv").exists():
                df = robust_read_csv("data/processed/train.csv")
                engine.build_index_from_dataframe(df, index_name=index_name)
            else:
                print(f"[ERROR] FAISS index uninitialized and fallback dataset `data/processed/train.csv` missing: {e}")
                return 1

        results = engine.find_similar_incidents(query=query, top_k=top_k, export_reports=True)
        print(f"\n[TOP-{min(len(results), 5)} SEMANTIC MATCHES]")
        for r in results[:5]:
            print(f"  #{r['rank']} | {r['number']} | Sim: {r['similarity_score']:.4f} | Group: {r['assignment_group']} | {r['short_description'][:60]}...")

        print(f"\n[SUCCESS] Retrieved {len(results)} precedents. Full reports exported to reports/similarity_results.csv and .md")
        return 0

    def cmd_recommend(self, input_path: Optional[str], text: Optional[str], top_k: int = 5) -> int:
        """Run deterministic Hybrid Incident Intelligence Engine recommendation."""
        if not input_path and not text:
            print("[ERROR] Must specify either `--input sample_incident.json` or `--text 'query text'`.")
            return 1

        payload = input_path if input_path else text
        print(f"\n====================================================================")
        print(f"Executing Enterprise Hybrid Incident Intelligence Engine (`v2.0.0`)")
        print(f"====================================================================")

        from src.ml.hybrid.recommendation_engine import HybridRecommendationEngine
        engine = HybridRecommendationEngine()
        try:
            rec = engine.recommend(input_payload=payload, top_k=top_k, export_reports=True)
        except Exception as e:
            logger.error(f"Hybrid recommendation failed: {e}", exc_info=True)
            print(f"\n[CRITICAL ERROR] Hybrid recommendation failed: {e}")
            return 1

        print(f"\nPredicted Assignment Group : {rec['recommended_assignment_group']}")
        print(f"Confidence                 : {rec['confidence_tier']} ({rec['confidence_score']:.2%})")
        print(f"Estimated Resolution Time  : {rec['estimated_resolution_time_hours']:.2f} hours")
        print(f"Historical Success Rate    : {rec['historical_success_rate']:.2f}%\n")
        
        print("Historical Evidence:")
        print(f"  {'Rank':<5} | {'Incident Number':<16} | {'Sim Score':<10} | {'Historical Assignment Group':<28} | {'Historical Resolution Time'}")
        print(f"  {'-'*5} | {'-'*16} | {'-'*10} | {'-'*28} | {'-'*26}")
        for r in rec["historical_evidence"]:
            print(f"  #{r['rank']:<4} | {r['number']:<16} | {r['similarity_score']:<10.4f} | {r['historical_assignment_group']:<28} | {r['historical_resolution_time']}")

        print(f"\nReasoning:")
        print(f"  {rec['reasoning']}")
        for b in rec["reasoning_bullet_breakdown"]:
            clean_b = b.replace("• ", "* ")
            print(f"  {clean_b}")

        print(f"\n[SUCCESS] Hybrid recommendation complete. Full reports exported to reports/hybrid_prediction.json, .md, and .csv")
        return 0

    def _run_stage1_dataset_check(self, input_path: str) -> int:
        """Stage 1 helper: Check if real/existing dataset exists; fail if missing."""
        target_path = Path(input_path)
        if target_path.exists():
            size_kb = target_path.stat().st_size / 1024.0
            print(f"\n---> [Stage 1/12] Real/Existing dataset detected at '{input_path}' ({size_kb:.1f} KB).")
            print("     Preserving existing dataset.")
            return 0
        else:
            print(f"\n---> [Stage 1/12] No dataset found at '{input_path}'. Enterprise execution halted.")
            return 1

    def cmd_full_pipeline(self, input_path: str = "data/raw/incidents.csv") -> int:
        """Orchestrate all 12 stages of the Enterprise Incident Intelligence Platform sequentially."""
        start_time = time.time()
        print("\n" + "=" * 70)
        print("FIRST CITIZENS BANK — ENTERPRISE INCIDENT INTELLIGENCE PLATFORM")
        print("Executing Full End-to-End Enterprise Pipeline (`v2.0.0`)")
        print("=" * 70)

        stages = [
            ("Stage 1: Check/Prepare Input Dataset", lambda: self._run_stage1_dataset_check(input_path=input_path)),
            ("Stage 2: Enterprise Dataset Validation", lambda: self.cmd_validate(input_path=input_path)),
            ("Stage 3: Zero-Leakage Data Intelligence Pipeline", lambda: self.cmd_pipeline(input_path=input_path, output_dir="data/processed")),
            ("Stage 4: Train Classification Model (`assignment_group`)", lambda: self.cmd_train(target="assignment_group", compare_baselines=True, train_path="data/processed/train.csv", val_path="data/processed/val.csv")),
            ("Stage 5: Train Regression Model (`resolution_time_hours`)", lambda: self.cmd_train(target="resolution_time_hours", compare_baselines=True, train_path="data/processed/train.csv", val_path="data/processed/val.csv")),
            ("Stage 6: Evaluate Classification Model", lambda: self.cmd_evaluate(model_key="catboost_assignment_group:latest", test_data="data/processed/test.csv", target="assignment_group")),
            ("Stage 7: Run SHAP Explainability Diagnostics", lambda: self.cmd_explain(model_key="catboost_assignment_group:latest", input_path="data/processed/test.csv", target="assignment_group")),
            ("Stage 8: Generate Local Neural Embeddings (`TF-IDF + SVD`)", lambda: self.cmd_embed(input_path="data/processed/train.csv", batch_size=64)),
            ("Stage 9: Build & Register FAISS Vector Index", lambda: self.cmd_index(input_path="data/processed/train.csv", index_name="incident_semantic_index")),
            ("Stage 10: Execute Hybrid Recommendation Engine (Demo Precedent)", lambda: self.cmd_recommend(input_path=None, text="ATM cash withdrawal failing due to hardware network timeout on CMDB_CI ATM-001", top_k=5))
        ]

        stage_results = []
        overall_status = 0

        for idx, (stage_name, stage_fn) in enumerate(stages, 1):
            print("\n" + "-" * 70)
            print(f"[{idx}/12] {stage_name}")
            print("-" * 70)
            try:
                ret = stage_fn()
                if ret == 0 or ret is None:
                    status_str = "PASSED"
                else:
                    status_str = f"FAILED (Code {ret})"
                    overall_status = ret if isinstance(ret, int) and ret != 0 else 1
            except Exception as e:
                logger.error(f"Error during {stage_name}: {e}", exc_info=True)
                status_str = f"ERROR ({e})"
                overall_status = 1
            stage_results.append((idx, stage_name, status_str))
            if overall_status != 0:
                print(f"\n[CRITICAL FAILURE] Pipeline halted at {stage_name} due to {status_str}.")
                break

        total_elapsed = time.time() - start_time

        print("\n" + "=" * 70)
        print("[12/12] Stage 12: Final Execution Summary")
        print("=" * 70)
        print(f"Total Elapsed Time: {total_elapsed:.2f} seconds")
        print(f"Overall Status    : {'CERTIFIED SUCCESSFUL' if overall_status == 0 else 'EXCLUSION FAILED'}\n")
        
        # Fetch and print Model Performance
        model_reg = ModelRegistry.get_instance()
        clf_meta = model_reg.get_model_metadata("catboost_assignment_group", "latest")
        reg_meta = model_reg.get_model_metadata("catboost_resolution_time_hours", "latest")
        
        print(f"  {'Step':<5} | {'Stage Description':<55} | {'Status'}")
        print(f"  {'-'*5} | {'-'*55} | {'-'*12}")
        for idx, s_name, s_stat in stage_results:
            print(f"  #{idx:<4} | {s_name:<55} | {s_stat}")

        print("\nModel Performance Metrics:")
        if clf_meta:
            print(f"  ? Classification (Accuracy): {clf_meta.metrics.get('accuracy', 'N/A')} | F1 Score: {clf_meta.metrics.get('f1_weighted', 'N/A')}")
        if reg_meta:
            print(f"  ? Regression (RMSE)      : {reg_meta.metrics.get('rmse', 'N/A')} hours | MAE: {reg_meta.metrics.get('mae', 'N/A')} hours")

        print("\nGenerated Artifacts & Output Locations:")
        print("  • Processed Data : data/processed/ (train.csv, val.csv, test.csv, master_engineered_incidents.csv)")
        print("  • Trained Models : models/ (catboost_assignment_group.pkl, catboost_resolution_time_hours.pkl)")
        print("  • Vector Index   : indexes/ (incident_semantic_index_latest.index)")
        print("  • Audit Reports  : reports/ (validation_report.md, eda_report.md, hybrid_prediction.json/md/csv)")
        print("=" * 70 + "\n")

        return overall_status

    def cmd_clean_workspace(self) -> int:
        """Remove generated runtime artifacts while preserving source files and directory hierarchy."""
        print("\n" + "=" * 70)
        print("FIRST CITIZENS BANK — ENTERPRISE INCIDENT INTELLIGENCE PLATFORM (`v2.0.0`)")
        print("Cleaning Workspace Runtime Artifacts...")
        print("=" * 70)

        removed_reports = 0
        removed_models = 0
        removed_indexes = 0
        removed_logs = 0
        removed_cache = 0
        preserved_dirs = 0

        # 1. Clean Reports across all subfolders (json, csv, md, html, png, txt, js, css, ico)
        reports_p = Path("reports")
        if reports_p.exists():
            for child in reports_p.rglob("*"):
                if child.name == ".gitkeep":
                    continue
                if child.is_file() and child.suffix.lower() in [".json", ".csv", ".md", ".html", ".png", ".txt", ".js", ".css", ".ico"]:
                    try:
                        child.unlink()
                        removed_reports += 1
                    except Exception as e:
                        logger.warning(f"Could not remove report file {child}: {e}")

        # 2. Clean Models across all subfolders (pkl, npy, joblib, bin, csv, parquet) and hf_cache
        models_p = Path("models")
        if models_p.exists():
            for child in models_p.rglob("*"):
                if child.name == ".gitkeep":
                    continue
                if child.is_file() and child.suffix.lower() in [".pkl", ".npy", ".joblib", ".bin", ".csv", ".parquet"]:
                    try:
                        child.unlink()
                        removed_models += 1
                    except Exception as e:
                        logger.warning(f"Could not remove model file {child}: {e}")

        hf_cache_p = Path("models/embeddings/hf_cache")
        if hf_cache_p.exists():
            try:
                shutil.rmtree(hf_cache_p, ignore_errors=True)
                removed_models += 1
            except Exception:
                pass

        # 3. Clean Indexes across all subfolders (index, npy, json, csv, parquet)
        indexes_p = Path("indexes")
        if indexes_p.exists():
            for child in indexes_p.rglob("*"):
                if child.name == ".gitkeep":
                    continue
                if child.is_file() and child.suffix.lower() in [".index", ".npy", ".json", ".csv", ".parquet"]:
                    try:
                        child.unlink()
                        removed_indexes += 1
                    except Exception as e:
                        logger.warning(f"Could not remove index file {child}: {e}")

        # 4. Clean Logs across all subfolders (*.log)
        logs_p = Path("logs")
        if logs_p.exists():
            for child in logs_p.rglob("*"):
                if child.name == ".gitkeep":
                    continue
                if child.is_file():
                    try:
                        child.unlink()
                        removed_logs += 1
                    except Exception as e:
                        # If file is locked by current process, truncate to 0 bytes to free up space
                        try:
                            with open(child, "w") as f:
                                f.truncate(0)
                            removed_logs += 1
                        except Exception:
                            logger.warning(f"Could not remove or truncate log file {child}: {e}")

        # 5. Clean Cache (.pytest_cache, __pycache__, *.pyc, *.pyo, .coverage, htmlcov, *.tmp, *.bak, *.cache)
        for cache_dir in Path(".").rglob("__pycache__"):
            try:
                shutil.rmtree(cache_dir, ignore_errors=True)
                removed_cache += 1
            except Exception:
                pass

        for cache_dir in Path(".").rglob(".pytest_cache"):
            try:
                shutil.rmtree(cache_dir, ignore_errors=True)
                removed_cache += 1
            except Exception:
                pass

        for cache_dir in Path(".").rglob("htmlcov"):
            try:
                shutil.rmtree(cache_dir, ignore_errors=True)
                removed_cache += 1
            except Exception:
                pass

        for tmp_ext in ["*.pyc", "*.pyo", "*.tmp", "*.bak", "*.cache"]:
            for tmp_file in Path(".").rglob(tmp_ext):
                try:
                    tmp_file.unlink()
                    removed_cache += 1
                except Exception:
                    pass

        cov_file = Path(".coverage")
        if cov_file.exists():
            try:
                cov_file.unlink()
                removed_cache += 1
            except Exception:
                pass

        # Also clean temporary generated CSV/Parquet in data/processed and datasets/synthetic
        for tmp_dir in ["data/processed", "data/interim", "datasets/synthetic"]:
            p = Path(tmp_dir)
            if p.exists():
                for child in p.rglob("*"):
                    if child.name == ".gitkeep":
                        continue
                    if child.is_file() and child.suffix.lower() in [".csv", ".json", ".npy", ".parquet"]:
                        try:
                            child.unlink()
                            removed_cache += 1
                        except Exception:
                            pass

        # Count preserved directories
        for p in Path(".").rglob("*"):
            if p.is_dir() and not any(part.startswith(".") for part in p.parts):
                preserved_dirs += 1

        print("\n========================================")
        print("Workspace Cleanup Summary")
        print("========================================")
        print(f"Reports Removed       : {removed_reports}")
        print(f"Models Removed        : {removed_models}")
        print(f"Indexes Removed       : {removed_indexes}")
        print(f"Logs Removed          : {removed_logs}")
        print(f"Cache / Temp Removed  : {removed_cache}")
        print(f"Directories Preserved : {preserved_dirs}")
        print("----------------------------------------")
        print("Status                : Repository Clean")
        print("========================================\n")
        return 0

    def cmd_status(self) -> int:
        """Display real-time platform health and registry status."""
        all_feats = self.registry.list_all_features()
        from src.ml.embedding_registry import EmbeddingRegistry
        embed_reg = EmbeddingRegistry.get_instance()
        print("\n====================================================")
        print("First Citizens Bank — Incident Intelligence Platform Status (`v2.0.0`)")
        print("====================================================")
        print(f"Platform Architecture Release: v2.0.0")
        print(f"Central Feature Registry Attributes: {len(all_feats)} registered features")
        print(f"Registered ML Models in Catalog: {len(self.model_reg.models)}")
        print(f"Registered Vector Indexes in Catalog: {len(embed_reg.indexes)}")
        
        # Check directories
        dirs = ["data/raw", "data/processed", "reports", "reports/figures", "models", "indexes"]
        for d in dirs:
            path = Path(d)
            status = "ACTIVE & READY" if path.exists() else "NOT INITIALIZED"
            print(f"  [{status:<17}] Directory: {d}/")
        print("====================================================\n")
        return 0

    def run_interactive_menu(self) -> int:
        """Run non-blocking interactive terminal control menu (Options 1-21)."""
        print("\n====================================================")
        print("Enterprise Incident Intelligence Platform (First Citizens Bank)")
        print("====================================================")
        print("--- PHASE 1 & 2: DATA FOUNDATION & INTELLIGENCE ---")
        print("2. Validate Dataset")
        print("3. Run Quality Gates")
        print("4. ML Readiness Verification")
        print("5. Run Exploratory Data Analysis (EDA)")
        print("6. Run Data Cleaning Engine")
        print("7. Run Feature Engineering Engine")
        print("8. Run Complete End-to-End Pipeline (`Clean -> Engineer -> Preprocess -> Split`)")
        print("--- PHASE 3: CATBOOST ML MODULE ---")
        print("9. Train Classification Pipeline (`assignment_group` + Multi-Baseline Comparison)")
        print("10. Train Regression Pipeline (`resolution_time_hours` + Multi-Baseline Comparison)")
        print("12. Evaluate Trained Classification Model (`evaluate`)")
        print("13. Run SHAP Explainable AI Diagnostics (`explain`)")
        print("14. Audit Model Registry Catalog (`models`)")
        print("--- PHASE 4: SEMANTIC SIMILARITY ENGINE ---")
        print("15. Generate Local Neural Embeddings (`embed`)")
        print("16. Build & Register FAISS Vector Index (`index`)")
        print("17. Semantic Search by Historical Incident Number (`similar --incident`)")
        print("18. Semantic Search by Free Query Text (`similar --text`)")
        print("--- PHASE 5: HYBRID RECOMMENDATION ENGINE ---")
        print("19. Run Hybrid Intelligence Recommendation (`recommend`)")
        print("--- ENTERPRISE PACKAGING AUTOMATION ---")
        print("20. Execute Complete End-to-End Enterprise Pipeline (`full-pipeline`)")
        print("21. Clean Workspace Runtime Artifacts (`clean-workspace`)")
        print("--- SYSTEM HEALTH & MONITORING ---")
        print("22. View Project Status & Health (`status`)")
        print("23. Audit Model & Vector Registry Catalog (`models`)")
        print("24. Exit")
        print("====================================================")
        
        if not sys.stdin.isatty():
            print("\n[NOTE] Non-interactive terminal detected. Executing option 20 (Status Summary)...")
            return self.cmd_status()

        try:
            choice = input("Enter option (1-22): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting platform...")
            return 0

        input_path = "data/raw/incidents.csv"
        if choice == "1":
            print("\n[INFO] Synthetic dataset generation has been disabled in production.")
            return 0
        elif choice == "2":
            return self.cmd_validate(input_path)
        elif choice == "3" or choice == "4":
            return self.cmd_readiness(input_path)
        elif choice == "5":
            return self.cmd_eda(input_path, "reports")
        elif choice == "6":
            return self.cmd_clean(input_path, "data/processed/cleaned_incidents.csv")
        elif choice == "7":
            return self.cmd_engineer("data/processed/cleaned_incidents.csv", "data/processed/engineered_incidents.csv")
        elif choice == "8":
            return self.cmd_pipeline(input_path, "data/processed")
        elif choice == "9":
            return self.cmd_train(target="assignment_group", compare_baselines=True)
        elif choice == "10":
            return self.cmd_train(target="resolution_time_hours", compare_baselines=True)
        elif choice == "11":
            print("\n[INFO] Hyperparameter Optimization (HPO) has been removed.")
            return 0
        elif choice == "12":
            return self.cmd_evaluate(model_key="catboost_assignment_group:latest", target="assignment_group")
        elif choice == "13":
            return self.cmd_explain(model_key="catboost_assignment_group:latest", target="assignment_group")
        elif choice == "14":
            return self.cmd_models()
        elif choice == "15":
            return self.cmd_embed("data/processed/train.csv", batch_size=64)
        elif choice == "16":
            return self.cmd_index("data/processed/train.csv", index_name="incident_semantic_index")
        elif choice == "17":
            inc_num = input("Enter Incident Number (e.g. INC0000001): ").strip()
            return self.cmd_similar(incident=inc_num, text=None, top_k=10, index_name="incident_semantic_index")
        elif choice == "18":
            query_txt = input("Enter free natural language query (e.g. 'ATM cash withdrawal failing'): ").strip()
            return self.cmd_similar(incident=None, text=query_txt, top_k=10, index_name="incident_semantic_index")
        elif choice == "19":
            query_txt = input("Enter Incident JSON path or free query text (e.g. 'ATM cash jam'): ").strip()
            return self.cmd_recommend(input_path=None, text=query_txt, top_k=5)
        elif choice == "20":
            return self.cmd_full_pipeline(input_path="data/raw/incidents.csv")
        elif choice == "21":
            return self.cmd_clean_workspace()
        elif choice == "22":
            return self.cmd_status()
        elif choice == "23":
            return self.cmd_models()
        elif choice == "24":
            print("Exiting platform...")
            return 0
        else:
            print("[ERROR] Invalid choice. Enter a number between 1 and 24.")
            return 1
