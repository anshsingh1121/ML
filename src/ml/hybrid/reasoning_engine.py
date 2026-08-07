"""
Enterprise Hybrid Reasoning Engine (`v2.0.0-alpha` - Phase 5).

Synthesizes quantitative output metrics from Random Forest predictions and FAISS
Top-K semantic precedents into audit-ready, deterministic natural language justifications.

Governance Mandate:
- Zero LLMs, Zero GenAI, Zero cloud APIs.
- 100% deterministic template and quantitative rule synthesis.
"""

from typing import Any, Dict, List

from src.utils.logger import get_logger

logger = get_logger(__name__)


class HybridReasoningEngine:
    """
    Synthesizes quantitative decision metrics and historical precedents into
    human-readable explanations and formatted audit summaries.
    """

    @classmethod
    def generate_reasoning(
        cls,
        decision: Dict[str, Any],
        semantic_matches: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate comprehensive reasoning explanation and formatted Historical Evidence tables.

        Args:
            decision: Output dictionary from `HybridDecisionEngine.fuse_recommendation()`.
            semantic_matches: List of Top-K dictionary matches from FAISS index.

        Returns:
            Dictionary containing `executive_summary`, `bullet_breakdown`, and `historical_evidence_table`.
        """
        rec_group = decision.get("recommended_assignment_group", "General Support")
        tier = decision.get("confidence_tier", "Moderate")
        fused_conf = float(decision.get("confidence_score", 0.0))
        fused_mttr = float(decision.get("estimated_resolution_time_hours", 0.0))
        success_rate = float(decision.get("historical_success_rate", 0.0))

        agreement = bool(decision.get("agreement", False))
        reason_code = decision.get("decision_reason_code", "AGREEMENT")
        rf_group = decision.get("rf_predicted_group", rec_group)
        rf_conf = float(decision.get("rf_confidence", 0.0))
        rf_mttr = float(decision.get("rf_mttr", 0.0))

        mode_group = decision.get("mode_semantic_group", rec_group)
        mode_count = int(decision.get("semantic_consensus_count", 0))
        consensus_pct = float(decision.get("semantic_consensus_pct", 0.0))
        avg_sim = float(decision.get("average_similarity", 0.0))
        sem_mttr = float(decision.get("historical_mttr_average", 0.0))

        top_k = len(semantic_matches)

        # 1. Executive Summary Text
        if agreement:
            exec_summary = (
                f"Random Forest classification predicted assignment to '{rf_group}' with {rf_conf:.1%} confidence. "
                f"Simultaneously, semantic similarity retrieval identified {mode_count} out of {top_k} historical "
                f"precedents ({consensus_pct:.1f}%) belonging to '{mode_group}' with an average vector similarity "
                f"of {avg_sim:.4f}. Because both machine learning prediction and historical semantic evidence "
                f"converge on '{rec_group}', the recommendation is assigned a '{tier}' confidence rating of "
                f"{fused_conf:.2%} and an estimated resolution time of {fused_mttr:.2f} hours."
            )
        else:
            if reason_code == "RF_DOMINANT":
                exec_summary = (
                    f"Random Forest classification predicted assignment to '{rf_group}' with high confidence ({rf_conf:.1%}), "
                    f"whereas historical semantic precedents favored '{mode_group}' across {mode_count} of {top_k} matches "
                    f"({consensus_pct:.1f}% mode consensus, {avg_sim:.4f} average similarity). Based on configuration-driven "
                    f"dominant threshold weighting, '{rec_group}' is recommended with '{tier}' confidence ({fused_conf:.2%}) "
                    f"and an estimated resolution window of {fused_mttr:.2f} hours."
                )
            else:
                exec_summary = (
                    f"Semantic similarity search retrieved strong consensus for '{mode_group}' across {mode_count} of {top_k} "
                    f"historical precedents ({consensus_pct:.1f}% consensus, {avg_sim:.4f} average similarity), superseding "
                    f"the Random Forest prediction of '{rf_group}' ({rf_conf:.1%} confidence). Consequently, '{rec_group}' "
                    f"is recommended with '{tier}' confidence ({fused_conf:.2%}) and an estimated resolution time of "
                    f"{fused_mttr:.2f} hours."
                )

        # 2. Bulleted Breakdown
        bullets = [
            f"• Machine Learning Prediction: {rf_group} ({rf_conf:.1%} confidence, {rf_mttr:.2f}h estimated MTTR)",
            f"• Semantic Precedent Consensus: {mode_count} of {top_k} similar incidents assigned to {mode_group} ({consensus_pct:.1f}% consensus, {avg_sim:.4f} avg similarity)",
            f"• Cross-Engine Agreement: {'Yes (Verified Convergence)' if agreement else 'No (Weighted Configuration Resolution)'}",
            f"• Fused Resolution Estimate: {fused_mttr:.2f} hours (blending ML regression and historical mean {sem_mttr:.2f}h)",
            f"• Historical Operational Success Rate: {success_rate:.2f}% across top similar tickets"
        ]

        # 3. Format Historical Evidence Table
        evidence_list = []
        for rank, match in enumerate(semantic_matches, start=1):
            inc_id = str(match.get("incident_number", f"INC{rank:05d}"))
            sim_val = float(match.get("similarity_score", 0.0))
            grp_val = str(match.get("assignment_group", "Unknown"))
            time_val = match.get("resolution_time_hours")
            if time_val is None:
                time_val = match.get("resolution_time", "N/A")
            try:
                formatted_time = f"{float(time_val):.2f}h"
            except (ValueError, TypeError):
                formatted_time = str(time_val)

            evidence_list.append({
                "rank": rank,
                "incident_number": inc_id,
                "similarity_score": round(sim_val, 4),
                "historical_assignment_group": grp_val,
                "historical_resolution_time": formatted_time
            })

        logger.debug(f"Generated deterministic reasoning summary for '{rec_group}'.")
        return {
            "executive_summary": exec_summary,
            "bullet_breakdown": bullets,
            "historical_evidence": evidence_list
        }
