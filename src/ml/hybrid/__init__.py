"""
Enterprise Hybrid Incident Intelligence Engine (`v2.0.0-alpha` - Phase 5).

Orchestration layer that deterministically fuses Machine Learning predictions
(Random Forest classification & regression) with Semantic Similarity Search (FAISS Top-K precedents)
into a unified, explainable enterprise recommendation.

Governance Mandate:
- 100% deterministic (Zero LLMs, Zero GenAI, Zero cloud APIs).
- Configuration-driven parameters (`ConfigManager` reading `ml.hybrid`).
- Full Windows file-lock resilience (`_latest` fallbacks for reports).
"""

from src.ml.hybrid.confidence_engine import HybridConfidenceEngine
from src.ml.hybrid.decision_engine import HybridDecisionEngine
from src.ml.hybrid.reasoning_engine import HybridReasoningEngine
from src.ml.hybrid.recommendation_engine import HybridRecommendationEngine

__all__ = [
    "HybridConfidenceEngine",
    "HybridDecisionEngine",
    "HybridReasoningEngine",
    "HybridRecommendationEngine",
]
