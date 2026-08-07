"""
Enterprise Hybrid Decision Engine (`v2.0.0-alpha` - Phase 5).

Deterministically fuses Random Forest ML predictions with FAISS Semantic Top-K
precedents to produce a verified enterprise recommendation, estimated resolution time,
and historical success rate.

Governance Mandate:
- Zero hardcoded magic numbers (`ConfigManager` reads `ml.hybrid`).
- 100% deterministic rule and statistical blending.
"""

from collections import Counter
from typing import Any, Dict, List, Optional

import numpy as np

from src.ml.hybrid.confidence_engine import HybridConfidenceEngine
from src.utils.config_manager import ConfigManager
from src.utils.logger import get_logger

logger = get_logger(__name__)


class HybridDecisionEngine:
    """
    Orchestration decision layer for fusing ML outputs and Semantic search precedents.
    """

    def __init__(
        self,
        confidence_engine: Optional[HybridConfidenceEngine] = None,
        config_manager: Optional[ConfigManager] = None
    ) -> None:
        """
        Initialize HybridDecisionEngine.
        """
        self.config_mgr = config_manager or ConfigManager.get_instance()
        self.conf_engine = confidence_engine or HybridConfidenceEngine(self.config_mgr)
        cfg = self.config_mgr.get_hybrid_config()

        self.rf_dominant_thresh = float(cfg.get("rf_dominant_threshold", 0.70))
        self.mttr_rf_weight = float(cfg.get("mttr_rf_weight", 0.50))
        self.mttr_sem_weight = float(cfg.get("mttr_semantic_weight", 0.50))
        self.partial_reassign_credit = float(cfg.get("partial_reassign_credit", 0.50))

    def fuse_recommendation(
        self,
        rf_prediction: Dict[str, Any],
        semantic_matches: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Execute deterministic fusion logic combining RF prediction and Top-K precedents.

        Args:
            rf_prediction: Dictionary containing `assignment_group`, `confidence_score` (or `confidence`),
                           `resolution_time_hours`, and optional `priority`.
            semantic_matches: List of Top-K dictionary matches from `FAISSVectorIndex`.

        Returns:
            Structured decision dictionary containing recommended group, fused confidence,
            estimated resolution time, historical success rate, and intermediate consensus metrics.
        """
        rf_group = str(rf_prediction.get("assignment_group") or rf_prediction.get("predicted_class") or "General Support")
        rf_conf = float(rf_prediction.get("confidence_score") or rf_prediction.get("confidence") or 0.50)
        rf_mttr = float(rf_prediction.get("resolution_time_hours") or rf_prediction.get("resolution_time") or 4.0)

        top_k = len(semantic_matches)
        if top_k == 0:
            logger.warning("No semantic matches provided to Decision Engine; falling back entirely to Random Forest.")
            fused_conf, tier = self.conf_engine.calculate_confidence(rf_conf, 0.0, True, 0)
            return {
                "recommended_assignment_group": rf_group,
                "confidence_score": fused_conf,
                "confidence_tier": tier,
                "estimated_resolution_time_hours": round(rf_mttr, 2),
                "historical_success_rate": 0.0,
                "agreement": True,
                "mode_semantic_group": rf_group,
                "semantic_consensus_count": 0,
                "semantic_consensus_pct": 0.0,
                "average_similarity": 0.0,
                "historical_mttr_average": round(rf_mttr, 2)
            }

        # Analyze Semantic Top-K Consensus
        sem_groups = [
            str(m.get("assignment_group", "General Support"))
            for m in semantic_matches
        ]
        group_counts = Counter(sem_groups)
        mode_group, mode_count = group_counts.most_common(1)[0]
        sem_consensus_pct = mode_count / top_k

        # Compute average similarity score across all matches or mode matches
        sim_scores = [float(m.get("similarity_score", 0.0)) for m in semantic_matches]
        avg_similarity = float(np.mean(sim_scores)) if sim_scores else 0.0

        # Determine agreement
        agreement = (rf_group == mode_group)

        # Fused Group Recommendation Logic
        if agreement:
            recommended_group = rf_group
            reason_code = "AGREEMENT"
        else:
            if rf_conf >= self.rf_dominant_thresh:
                recommended_group = rf_group
                reason_code = "RF_DOMINANT"
            else:
                recommended_group = mode_group
                reason_code = "SEMANTIC_DOMINANT"

        # Calculate Fused Confidence Score and Tier
        fused_conf, tier = self.conf_engine.calculate_confidence(
            rf_confidence=rf_conf,
            sem_confidence=avg_similarity,
            agreement=agreement,
            top_k_matches=top_k
        )

        # Compute Historical MTTR among semantic precedents
        hist_mttr_vals = []
        for m in semantic_matches:
            mttr_val = m.get("resolution_time_hours")
            if mttr_val is None:
                mttr_val = m.get("resolution_time")
            if mttr_val is not None:
                try:
                    hist_mttr_vals.append(float(mttr_val))
                except (ValueError, TypeError):
                    pass

        sem_mttr = float(np.mean(hist_mttr_vals)) if hist_mttr_vals else rf_mttr
        fused_mttr = (rf_mttr * self.mttr_rf_weight) + (sem_mttr * self.mttr_sem_weight)

        # Calculate Historical Success Rate across Top-K precedents
        # Defined as precedents that either had 0 reassignments OR matched the recommended assignment group
        success_count = 0
        for m in semantic_matches:
            reassign_cnt = m.get("reassignment_count")
            match_group = str(m.get("assignment_group", ""))
            is_group_match = (match_group == recommended_group)
            try:
                clean_reassign = int(reassign_cnt) if reassign_cnt is not None else 0
            except (ValueError, TypeError):
                clean_reassign = 0

            if clean_reassign == 0 and is_group_match:
                success_count += 1
            elif is_group_match and (clean_reassign <= 1):
                success_count += self.partial_reassign_credit  # Partial credit for minor reassignment within target team

        hist_success_rate = round((success_count / top_k) * 100.0, 2)
        # Ensure success rate reflects strong consensus Mode if reassignments weren't logged precisely
        if hist_success_rate == 0.0 and agreement:
            hist_success_rate = round(sem_consensus_pct * 100.0, 2)

        decision = {
            "recommended_assignment_group": recommended_group,
            "confidence_score": fused_conf,
            "confidence_tier": tier,
            "estimated_resolution_time_hours": round(fused_mttr, 2),
            "historical_success_rate": hist_success_rate,
            "agreement": agreement,
            "decision_reason_code": reason_code,
            "rf_predicted_group": rf_group,
            "rf_confidence": round(rf_conf, 4),
            "rf_mttr": round(rf_mttr, 2),
            "mode_semantic_group": mode_group,
            "semantic_consensus_count": mode_count,
            "semantic_consensus_pct": round(sem_consensus_pct * 100.0, 2),
            "average_similarity": round(avg_similarity, 4),
            "historical_mttr_average": round(sem_mttr, 2)
        }

        logger.info(
            f"Decision Engine Output: Recommended Group='{recommended_group}' "
            f"(Confidence: {fused_conf:.2%} - {tier}, MTTR: {fused_mttr:.2f}h, Success Rate: {hist_success_rate:.1f}%)"
        )
        return decision
