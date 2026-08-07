"""
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
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd
from src.utils import robust_open

from src.ml.hybrid.confidence_engine import HybridConfidenceEngine
from src.ml.hybrid.decision_engine import HybridDecisionEngine
from src.ml.hybrid.reasoning_engine import HybridReasoningEngine
from src.ml.model_registry import ModelRegistry
from src.ml.random_forest.transformers import EnterpriseFeatureExtractor
from src.ml.semantic.similarity_engine import SemanticSimilarityEngine
from src.utils.config_manager import ConfigManager
from src.utils.logger import get_logger

logger = get_logger(__name__)


class HybridRecommendationEngine:
    """
    Master orchestration controller for the Enterprise Hybrid Incident Intelligence Engine.
    """

    def __init__(
        self,
        models_dir: Union[str, Path] = "models",
        reports_dir: Union[str, Path] = "reports",
        rf_classifier_pipeline: Optional[Any] = None,
        rf_regressor_pipeline: Optional[Any] = None,
        semantic_engine: Optional[SemanticSimilarityEngine] = None,
        decision_engine: Optional[HybridDecisionEngine] = None,
        config_manager: Optional[ConfigManager] = None
    ) -> None:
        """
        Initialize HybridRecommendationEngine and load required sub-engines and models.
        """
        self.config_mgr = config_manager or ConfigManager.get_instance()
        self.models_dir = Path(models_dir)
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        self.model_reg = ModelRegistry.get_instance(base_dir=str(self.models_dir))
        self.decision_engine = decision_engine or HybridDecisionEngine(config_manager=self.config_mgr)
        self.semantic_engine = semantic_engine or SemanticSimilarityEngine(reports_dir=self.reports_dir)

        # Load Random Forest pipelines from canonical Model Registry source or exact fallback path
        self.rf_classifier = rf_classifier_pipeline
        if self.rf_classifier is None:
            clf_path = self.model_reg.get_model_path("random_forest_assignment_group")
            if not clf_path:
                clf_path = self.models_dir / "random_forest_assignment_group.pkl"
            if clf_path.exists():
                try:
                    self.rf_classifier = joblib.load(clf_path)
                    logger.info(f"Loaded Random Forest classifier pipeline from {clf_path}")
                except Exception as e:
                    logger.warning(f"Could not load classifier pipeline from {clf_path}: {e}")
            else:
                logger.warning(f"Classifier pipeline file not found at {clf_path}; RF predictions will use defaults unless provided.")

        self.rf_regressor = rf_regressor_pipeline
        if self.rf_regressor is None:
            reg_path = self.model_reg.get_model_path("random_forest_resolution_time_hours")
            if not reg_path:
                reg_path = self.models_dir / "random_forest_resolution_time_hours.pkl"
            if reg_path.exists():
                try:
                    self.rf_regressor = joblib.load(reg_path)
                    logger.info(f"Loaded Random Forest regressor pipeline from {reg_path}")
                except Exception as e:
                    logger.warning(f"Could not load regressor pipeline from {reg_path}: {e}")
            else:
                logger.warning(f"Regressor pipeline file not found at {reg_path}; RF regression will use defaults unless provided.")

        self.feature_extractor = EnterpriseFeatureExtractor()

    def _parse_input_payload(self, input_payload: Union[str, Dict[str, Any], Path]) -> Dict[str, Any]:
        """Parse raw JSON path, dictionary, or free-text string into a structured ticket dict."""
        if isinstance(input_payload, dict):
            return input_payload.copy()

        if isinstance(input_payload, (str, Path)):
            path_obj = Path(input_payload)
            if path_obj.exists() and path_obj.is_file():
                try:
                    with robust_open(path_obj, "r") as f:
                        data = json.load(f)
                        return data if isinstance(data, dict) else {}
                except Exception as e:
                    logger.error(f"Failed to parse JSON file {path_obj}: {e}")
                    raise ValueError(f"Invalid JSON content in {path_obj}: {e}")

            # Treat as free natural language text
            text_str = str(input_payload).strip()
            now_ts = pd.Timestamp.now()
            return {
                "incident_number": "INC_QUERY_001",
                "short_description": text_str,
                "description": text_str,
                "category": "General",
                "subcategory": "General",
                "business_service": "Enterprise Core Service",
                "cmdb_ci": "ci-query-node",
                "priority": "P3 - Moderate",
                "impact": "3 - Low",
                "urgency": "3 - Low",
                "severity": "3 - Low",
                "contact_type": "Self service",
                "location": "UNKNOWN",
                "vendor": "UNKNOWN",
                "reassignment_count": 0,
                "reopen_count": 0,
                "is_business_hours": 1 if 8 <= now_ts.hour <= 17 else 0,
                "has_change_request": 0,
                "has_problem_record": 0,
                "is_duplicate": 0,
                "has_parent_incident": 0,
                "opened_at_dayofweek": now_ts.dayofweek,
                "opened_at_hour": now_ts.hour,
                "opened_at": now_ts.isoformat()
            }

        raise TypeError(f"Unsupported input payload type: {type(input_payload)}")

    def _sync_features_for_model(self, df: pd.DataFrame, model: Any) -> pd.DataFrame:
        """Ensure input DataFrame contains all expected features required by scikit-learn model."""
        df_synced = df.copy()
        expected_cols = []
        if hasattr(model, "feature_names_in_"):
            expected_cols = list(model.feature_names_in_)
        elif hasattr(model, "named_steps") and "preprocessing" in model.named_steps:
            if hasattr(model.named_steps["preprocessing"], "feature_names_in_"):
                expected_cols = list(model.named_steps["preprocessing"].feature_names_in_)
            elif hasattr(model.named_steps["preprocessing"], "extractor") and hasattr(model.named_steps["preprocessing"].extractor, "feature_names_in_"):
                expected_cols = list(model.named_steps["preprocessing"].extractor.feature_names_in_)

        for col in expected_cols:
            if col not in df_synced.columns:
                if col in ["category", "subcategory", "business_service", "location", "cmdb_ci", "vendor", "contact_type", "priority", "impact", "urgency", "severity"]:
                    df_synced[col] = "UNKNOWN"
                elif col in ["short_description", "description"]:
                    df_synced[col] = ""
                else:
                    df_synced[col] = 0

        return df_synced

    def _predict_rf(self, ticket_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Random Forest prediction on ticket dictionary."""
        df = pd.DataFrame([ticket_dict])
        df_extracted = self.feature_extractor.transform(df)

        pred_group = "General Support"
        rf_conf = 0.50
        if self.rf_classifier is not None:
            try:
                df_clf = self._sync_features_for_model(df_extracted, self.rf_classifier)
                pred_group = str(self.rf_classifier.predict(df_clf)[0])
                if hasattr(self.rf_classifier, "predict_proba"):
                    probs = self.rf_classifier.predict_proba(df_clf)[0]
                    rf_conf = float(np.max(probs))
            except Exception as e:
                logger.warning(f"RF classifier prediction failed during hybrid evaluation: {e}")

        pred_mttr = 4.0
        if self.rf_regressor is not None:
            try:
                df_reg = self._sync_features_for_model(df_extracted, self.rf_regressor)
                raw_pred = float(self.rf_regressor.predict(df_reg)[0])
                # Check if model predicted log-transformed target (`np.log1p`)
                if raw_pred < 6.0 and raw_pred > 0.0:
                    # In banking log1p models, typical log output is ~1.0-4.0
                    pred_mttr = float(np.expm1(raw_pred))
                else:
                    pred_mttr = raw_pred
            except Exception as e:
                logger.warning(f"RF regressor prediction failed during hybrid evaluation: {e}")

        return {
            "assignment_group": pred_group,
            "confidence_score": round(rf_conf, 4),
            "resolution_time_hours": round(pred_mttr, 2),
            "priority": str(ticket_dict.get("priority", "P3 - Moderate"))
        }

    def recommend(
        self,
        input_payload: Union[str, Dict[str, Any], Path],
        top_k: int = 5,
        export_reports: bool = True
    ) -> Dict[str, Any]:
        """
        Orchestrate end-to-end Hybrid Incident Intelligence recommendation.

        Args:
            input_payload: JSON file path (`sample.json`), dict, or free text (`"ATM cash jam"`).
            top_k: Number of semantic precedents to retrieve via FAISS.
            export_reports: Whether to save `reports/hybrid_prediction.json`, `.md`, `.csv`.

        Returns:
            Comprehensive recommendation dictionary containing fused outputs, Historical Evidence,
            and explainable reasoning justification.
        """
        logger.info("Starting Enterprise Hybrid Incident Intelligence recommendation flow...")

        # Step 1: Parse input
        ticket_dict = self._parse_input_payload(input_payload)
        inc_number = str(ticket_dict.get("incident_number", "INC_QUERY_001"))

        # Step 2: Random Forest Prediction
        rf_prediction = self._predict_rf(ticket_dict)
        logger.debug(f"Step 2 RF Output: Group={rf_prediction['assignment_group']}, Conf={rf_prediction['confidence_score']:.2%}, MTTR={rf_prediction['resolution_time_hours']}h")

        # Step 3: Semantic Similarity Search
        # Pass free text or composite text to FAISS index
        query_text = str(ticket_dict.get("short_description", "")) + ". " + str(ticket_dict.get("description", ""))
        query_text = query_text.strip(". ")
        semantic_matches = self.semantic_engine.find_similar_incidents(
            query=query_text if query_text else inc_number,
            top_k=top_k,
            export_reports=False
        )
        logger.debug(f"Step 3 Semantic Output: Retrieved {len(semantic_matches)} precedents via FAISS.")

        # Step 4: Hybrid Decision Engine Fusing
        decision = self.decision_engine.fuse_recommendation(rf_prediction, semantic_matches)

        # Step 5: Explainable Reasoning Generation
        reasoning_output = HybridReasoningEngine.generate_reasoning(decision, semantic_matches)

        # Step 6: Assemble Master Recommendation Payload
        recommendation = {
            "incident_number": inc_number,
            "short_description": str(ticket_dict.get("short_description", "")),
            "recommended_assignment_group": decision["recommended_assignment_group"],
            "confidence_score": decision["confidence_score"],
            "confidence_tier": decision["confidence_tier"],
            "estimated_resolution_time_hours": decision["estimated_resolution_time_hours"],
            "historical_success_rate": decision["historical_success_rate"],
            "decision_reason_code": decision["decision_reason_code"],
            "agreement": decision["agreement"],
            "rf_prediction": rf_prediction,
            "semantic_consensus": {
                "mode_group": decision["mode_semantic_group"],
                "consensus_count": decision["semantic_consensus_count"],
                "consensus_pct": decision["semantic_consensus_pct"],
                "average_similarity": decision["average_similarity"],
                "historical_mttr_average": decision["historical_mttr_average"]
            },
            "reasoning": reasoning_output["executive_summary"],
            "reasoning_bullet_breakdown": reasoning_output["bullet_breakdown"],
            "historical_evidence": reasoning_output["historical_evidence"]
        }

        # Step 7: Export Reports
        if export_reports:
            self.export_reports(recommendation)

        logger.info(
            f"Hybrid Recommendation Complete: {recommendation['recommended_assignment_group']} "
            f"({recommendation['confidence_tier']} - {recommendation['confidence_score']:.2%})"
        )
        return recommendation

    def export_reports(self, recommendation: Dict[str, Any]) -> None:
        """
        Export formal enterprise recommendation reports (`.json`, `.md`, `.csv`)
        with complete Windows file-lock resilience (`_latest`).
        """
        # 1. JSON Report
        json_path = self.reports_dir / "hybrid_prediction.json"
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(recommendation, f, indent=2, ensure_ascii=False)
        except PermissionError:
            json_path_latest = self.reports_dir / "hybrid_prediction_latest.json"
            logger.warning(f"File {json_path} locked by another process. Writing to {json_path_latest}")
            with open(json_path_latest, "w", encoding="utf-8") as f:
                json.dump(recommendation, f, indent=2, ensure_ascii=False)

        # 2. Markdown Report
        md_path = self.reports_dir / "hybrid_prediction.md"
        md_lines = [
            f"# Enterprise Hybrid Incident Intelligence Report (`v2.0.0-alpha`)",
            f"**Incident ID:** `{recommendation['incident_number']}`  ",
            f"**Query/Summary:** `{recommendation['short_description']}`  ",
            f"**Execution Timestamp:** `{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}`  \n",
            f"---",
            f"## Final Recommendation Summary",
            f"- **Recommended Assignment Group:** `{recommendation['recommended_assignment_group']}`",
            f"- **Confidence Rating:** **{recommendation['confidence_tier']}** (`{recommendation['confidence_score']:.2%}`)",
            f"- **Estimated Resolution Time (MTTR):** `{recommendation['estimated_resolution_time_hours']:.2f} hours`",
            f"- **Historical Success Rate:** `{recommendation['historical_success_rate']:.2f}%`",
            f"- **Cross-Engine Agreement:** `{'Yes (Verified Convergence)' if recommendation['agreement'] else 'No (Configuration Weighted Resolution)'}`\n",
            f"---",
            f"## Explainable Reasoning",
            f"{recommendation['reasoning']}\n",
            f"### Key Decision Justifications"
        ]
        for b in recommendation["reasoning_bullet_breakdown"]:
            md_lines.append(b)

        md_lines.extend([
            f"\n---",
            f"## Historical Evidence",
            f"| Rank | Incident Number | Similarity Score | Historical Assignment Group | Historical Resolution Time |",
            f"| :---: | :---: | :---: | :--- | :---: |"
        ])
        for row in recommendation["historical_evidence"]:
            md_lines.append(
                f"| #{row['rank']} | `{row['incident_number']}` | `{row['similarity_score']:.4f}` | "
                f"`{row['historical_assignment_group']}` | `{row['historical_resolution_time']}` |"
            )

        md_content = "\n".join(md_lines) + "\n"
        try:
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(md_content)
        except PermissionError:
            md_path_latest = self.reports_dir / "hybrid_prediction_latest.md"
            logger.warning(f"File {md_path} locked by another process. Writing to {md_path_latest}")
            with open(md_path_latest, "w", encoding="utf-8") as f:
                f.write(md_content)

        # 3. CSV Report (Historical Evidence + Top-Level Summary)
        csv_path = self.reports_dir / "hybrid_prediction.csv"
        df_evidence = pd.DataFrame(recommendation["historical_evidence"])
        if not df_evidence.empty:
            df_evidence["recommended_group"] = recommendation["recommended_assignment_group"]
            df_evidence["confidence_score"] = recommendation["confidence_score"]
            df_evidence["confidence_tier"] = recommendation["confidence_tier"]
            df_evidence["estimated_mttr_hours"] = recommendation["estimated_resolution_time_hours"]
            df_evidence["historical_success_rate"] = recommendation["historical_success_rate"]
        else:
            df_evidence = pd.DataFrame([{
                "incident_number": recommendation["incident_number"],
                "recommended_group": recommendation["recommended_assignment_group"],
                "confidence_score": recommendation["confidence_score"],
                "confidence_tier": recommendation["confidence_tier"],
                "estimated_mttr_hours": recommendation["estimated_resolution_time_hours"],
                "historical_success_rate": recommendation["historical_success_rate"]
            }])

        try:
            df_evidence.to_csv(csv_path, index=False, encoding="utf-8")
        except PermissionError:
            csv_path_latest = self.reports_dir / "hybrid_prediction_latest.csv"
            logger.warning(f"File {csv_path} locked by another process. Writing to {csv_path_latest}")
            df_evidence.to_csv(csv_path_latest, index=False, encoding="utf-8")

        logger.info(f"Exported Phase 5 Hybrid prediction reports to {self.reports_dir}")
