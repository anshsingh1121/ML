"""
Enterprise Hybrid Confidence Engine (`v2.0.0-alpha` - Phase 5).

Calculates deterministic numerical confidence (`0.0` to `1.0`) and categorical confidence
tiers (`Very High`, `High`, `Moderate`, `Low`, `Review Required`) based on cross-engine
agreement between Random Forest classification confidence and Semantic FAISS consensus.

Governance Mandate:
- Zero hardcoded magic numbers (`ConfigManager` reads `ml.hybrid`).
- 100% deterministic mathematical evaluation.
"""

from typing import Any, Dict, Optional, Tuple

from src.utils.config_manager import ConfigManager
from src.utils.logger import get_logger

logger = get_logger(__name__)


class HybridConfidenceEngine:
    """
    Configuration-driven confidence fusion engine.

    Synthesizes numerical probability from Random Forest classifiers (`rf_confidence`)
    and average/consensus similarity score from Top-K FAISS retrieval (`sem_confidence`)
    into a calibrated overall recommendation confidence and category tier.
    """

    def __init__(self, config_manager: Optional[ConfigManager] = None) -> None:
        """
        Initialize HybridConfidenceEngine reading parameters from `ConfigManager`.
        """
        self.config_mgr = config_manager or ConfigManager.get_instance()
        cfg = self.config_mgr.get_hybrid_config()

        self.rf_weight = float(cfg.get("rf_weight", 0.60))
        self.semantic_weight = float(cfg.get("semantic_weight", 0.40))
        self.agreement_bonus = float(cfg.get("agreement_bonus", 0.10))
        self.disagreement_penalty = float(cfg.get("disagreement_penalty", 0.05))
        self.insufficient_matches_penalty = float(cfg.get("insufficient_matches_penalty", 0.10))

        self.very_high_thresh = float(cfg.get("very_high_threshold", 0.88))
        self.high_thresh = float(cfg.get("high_threshold", 0.75))
        self.moderate_thresh = float(cfg.get("moderate_threshold", 0.60))
        self.low_thresh = float(cfg.get("low_threshold", 0.45))

    def calculate_confidence(
        self,
        rf_confidence: float,
        sem_confidence: float,
        agreement: bool,
        top_k_matches: int = 5
    ) -> Tuple[float, str]:
        """
        Compute fused numerical confidence score and classification tier.

        Args:
            rf_confidence: Classification probability/confidence from Random Forest (`0.0 to 1.0`).
            sem_confidence: Consensus similarity or average similarity from FAISS Top-K (`0.0 to 1.0`).
            agreement: `True` if RF predicted group matches the mode/majority assignment group across Top-K precedents.
            top_k_matches: Number of semantic precedents retrieved.

        Returns:
            Tuple of `(numerical_confidence_score, categorical_tier)`.
        """
        # Base weighted blend
        base_score = (rf_confidence * self.rf_weight) + (sem_confidence * self.semantic_weight)

        # Apply agreement bonus / disagreement penalty
        if agreement:
            fused_score = base_score + self.agreement_bonus
        else:
            fused_score = base_score - self.disagreement_penalty

        # Penalty if insufficient semantic precedents were retrieved
        if top_k_matches < 2:
            fused_score -= self.insufficient_matches_penalty

        # Clamp cleanly between 0.0001 and 1.0000
        fused_score = max(0.0001, min(1.0, float(fused_score)))

        # Determine categorical confidence tier
        if fused_score >= self.very_high_thresh:
            tier = "Very High"
        elif fused_score >= self.high_thresh:
            tier = "High"
        elif fused_score >= self.moderate_thresh:
            tier = "Moderate"
        elif fused_score >= self.low_thresh:
            tier = "Low"
        else:
            tier = "Review Required"

        logger.debug(
            f"Confidence calculation: RF={rf_confidence:.4f}, Sem={sem_confidence:.4f}, "
            f"Agreement={agreement} -> Fused={fused_score:.4f} ({tier})"
        )
        return round(fused_score, 4), tier
