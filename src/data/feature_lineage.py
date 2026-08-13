"""
Feature Lineage Tracker (`v1.5.0`).

Tracks parent-child derivation relationships and mathematical transformation formulas
for all engineered attributes in the AI-Powered Incident Intelligence Platform.
"""

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class LineageEdge:
    """Represents a direct transformation step from source attribute(s) to derived attribute."""
    derived_feature: str
    source_features: List[str]
    transformation_type: str
    formula: str
    business_rationale: str
    stage: str  # e.g., 'Phase 2 (EDA/Prep)', 'Phase 3 (Engineering)'

    def to_dict(self) -> Dict[str, Any]:
        """Convert lineage edge to dictionary."""
        return asdict(self)


class FeatureLineageTracker:
    """
    Centralized governance graph tracking exact ancestry and formulas for all derived features.
    Supports singleton access via get_instance() and direct edge creation.
    """
    _instance: Optional["FeatureLineageTracker"] = None

    @classmethod
    def get_instance(cls) -> "FeatureLineageTracker":
        """Retrieve or initialize the singleton FeatureLineageTracker instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        """Initialize FeatureLineageTracker and populate standard enterprise derivation rules."""
        self.edges: Dict[str, LineageEdge] = {}
        self._populate_default_lineage()

    def add_lineage(self, edge: LineageEdge) -> None:
        """Register a feature derivation step."""
        self.edges[edge.derived_feature] = edge
        logger.debug(f"Registered lineage edge: {edge.source_features} -> {edge.derived_feature}")

    def add_edge(
        self,
        source: Any,
        target: str,
        transformation: str = "Feature Engineering",
        formula: str = "",
        stage: str = "Phase 2 (Preprocessing)",
        rationale: str = "Engineered feature for ML modeling"
    ) -> None:
        """Helper wrapper to create or merge a LineageEdge from string or list arguments."""
        new_sources = [source] if isinstance(source, str) else list(source)
        existing = self.edges.get(target)
        if existing:
            for s in new_sources:
                if s not in existing.source_features:
                    existing.source_features.append(s)
            if formula and not existing.formula:
                existing.formula = formula
        else:
            edge = LineageEdge(
                derived_feature=target,
                source_features=new_sources,
                transformation_type=transformation,
                formula=formula,
                business_rationale=rationale,
                stage=stage
            )
            self.add_lineage(edge)

    def get_lineage(self, derived_feature: str) -> Optional[LineageEdge]:
        """Retrieve lineage definition for a specific derived feature."""
        return self.edges.get(derived_feature)

    def get_ancestry_chain(self, feature_name: str) -> List[str]:
        """Recursively trace back all ancestral raw source features."""
        chain = [feature_name]
        edge = self.edges.get(feature_name)
        if edge:
            for src in edge.source_features:
                chain.extend(self.get_ancestry_chain(src))
        # Deduplicate while preserving order
        seen = set()
        return [f for f in chain if not (f in seen or seen.add(f))]

    def export_json(self, output_path: Optional[str] = None) -> Path:
        """Export full feature lineage graph to feature_lineage.json."""
        out_file = Path(output_path or "reports/feature_lineage.json")
        out_file.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "lineage_version": "1.5.0",
            "last_updated": "2026-07-11T11:00:00Z",
            "total_derived_features": len(self.edges),
            "lineage_graph": {k: v.to_dict() for k, v in sorted(self.edges.items())}
        }

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        logger.info(f"Exported Feature Lineage JSON to: {out_file}")
        return out_file

    def export_markdown(self, output_path: Optional[str] = None) -> Path:
        """Export full feature lineage to feature_lineage.md specification table and tree."""
        out_file = Path(output_path or "reports/feature_lineage.md")
        out_file.parent.mkdir(parents=True, exist_ok=True)

        lines = [
            "# Enterprise Feature Lineage & Transformation Graph (`v1.5.0`)",
            "**Organization:** First Citizens Bank — Incident Intelligence Platform (IIP)  ",
            f"**Total Tracked Derived Attributes:** {len(self.edges)}  ",
            "**Governance Mandate:** All feature transformations must maintain documented ancestry and mathematical formulas.  \n",
            "---",
            "\n## Lineage Tree & Derivation Matrix\n",
            "| Derived Feature | Source Attribute(s) | Transformation Type | Mathematical / Logical Formula | Stage |",
            "|---|---|---|---|:---:|"
        ]

        for k, edge in sorted(self.edges.items()):
            srcs = ", ".join([f"`{s}`" for s in edge.source_features])
            lines.append(f"| `{edge.derived_feature}` | {srcs} | `{edge.transformation_type}` | `{edge.formula}` | {edge.stage} |")

        lines.extend([
            "\n---",
            "\n## Visual Derivation Chain Summary\n"
        ])

        # Group by root source
        root_groups: Dict[str, List[LineageEdge]] = {}
        for edge in self.edges.values():
            root = edge.source_features[0]
            if root not in root_groups:
                root_groups[root] = []
            root_groups[root].append(edge)

        for root, edges in sorted(root_groups.items()):
            lines.append(f"### Root Source: `{root}`")
            for e in edges:
                lines.append(f"- `{root}` $\\rightarrow$ `{e.derived_feature}` (`{e.transformation_type}`): {e.business_rationale}")
            lines.append("")

        with open(out_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        logger.info(f"Exported Feature Lineage Markdown to: {out_file}")
        return out_file

    def _populate_default_lineage(self) -> None:
        """Populate default derivation rules for temporal and relationship flags."""
        self.add_lineage(LineageEdge(
            derived_feature="opened_at_hour",
            source_features=["opened_at"],
            transformation_type="Temporal Extraction",
            formula="opened_at.dt.hour",
            business_rationale="Extracts integer hour (0-23) to capture daily triage volume peaks.",
            stage="Phase 2 (EDA/Prep)"
        ))

        self.add_lineage(LineageEdge(
            derived_feature="opened_at_hour_sin",
            source_features=["opened_at_hour"],
            transformation_type="Cyclic Sine Encoding",
            formula="sin(2 * pi * opened_at_hour / 24)",
            business_rationale="Preserves continuous shift proximity across midnight boundary (23:00 to 00:00).",
            stage="Phase 2 (EDA/Prep)"
        ))

        self.add_lineage(LineageEdge(
            derived_feature="opened_at_hour_cos",
            source_features=["opened_at_hour"],
            transformation_type="Cyclic Cosine Encoding",
            formula="cos(2 * pi * opened_at_hour / 24)",
            business_rationale="Preserves continuous shift proximity in tandem with sine component.",
            stage="Phase 2 (EDA/Prep)"
        ))

        self.add_lineage(LineageEdge(
            derived_feature="opened_at_dayofweek",
            source_features=["opened_at"],
            transformation_type="Temporal Extraction",
            formula="opened_at.dt.dayofweek",
            business_rationale="Extracts integer day of week (0=Mon to 6=Sun) to capture weekend staffing dynamics.",
            stage="Phase 2 (EDA/Prep)"
        ))

        self.add_lineage(LineageEdge(
            derived_feature="opened_at_dayofweek_sin",
            source_features=["opened_at_dayofweek"],
            transformation_type="Cyclic Sine Encoding",
            formula="sin(2 * pi * opened_at_dayofweek / 7)",
            business_rationale="Preserves continuous day proximity from Sunday to Monday.",
            stage="Phase 2 (EDA/Prep)"
        ))

        self.add_lineage(LineageEdge(
            derived_feature="opened_at_dayofweek_cos",
            source_features=["opened_at_dayofweek"],
            transformation_type="Cyclic Cosine Encoding",
            formula="cos(2 * pi * opened_at_dayofweek / 7)",
            business_rationale="Preserves continuous day proximity from Sunday to Monday.",
            stage="Phase 2 (EDA/Prep)"
        ))

        self.add_lineage(LineageEdge(
            derived_feature="is_business_hours",
            source_features=["opened_at"],
            transformation_type="Binary Thresholding",
            formula="1 if (dayofweek <= 4 and 8 <= hour < 18) else 0",
            business_rationale="Flags tickets opened during standard banking business hours vs after-hours.",
            stage="Phase 2 (EDA/Prep)"
        ))

        self.add_lineage(LineageEdge(
            derived_feature="resolution_time_hours",
            source_features=["opened_at", "resolved_at"],
            transformation_type="Timestamp Difference",
            formula="(resolved_at - opened_at).total_seconds() / 3600.0",
            business_rationale="Primary target outcome KPI measuring Mean Time To Resolution (MTTR).",
            stage="Phase 1 (Ingestion/Gen)"
        ))

        self.add_lineage(LineageEdge(
            derived_feature="has_parent_incident",
            source_features=["parent_incident"],
            transformation_type="Presence Indicator",
            formula="1 if (parent_incident != '' and not null) else 0",
            business_rationale="Identifies whether incident belongs to a major outage parent-child cluster.",
            stage="Phase 2 (EDA/Prep)"
        ))

        self.add_lineage(LineageEdge(
            derived_feature="has_change_request",
            source_features=["change_request"],
            transformation_type="Presence Indicator",
            formula="1 if (change_request != '' and not null) else 0",
            business_rationale="Strong signal routing towards release engineering and change management squads.",
            stage="Phase 2 (EDA/Prep)"
        ))

        self.add_lineage(LineageEdge(
            derived_feature="has_problem_id",
            source_features=["problem_id"],
            transformation_type="Presence Indicator",
            formula="1 if (problem_record != '' and not null) else 0",
            business_rationale="Flags systemic underlying defects requiring long-term architectural root cause analysis.",
            stage="Phase 2 (EDA/Prep)"
        ))

        self.add_lineage(LineageEdge(
            derived_feature="is_duplicate",
            source_features=["duplicate_incident"],
            transformation_type="Presence Indicator",
            formula="1 if (duplicate_incident != '' and not null) else 0",
            business_rationale="Flags secondary ticket storm duplicates linked to a master incident.",
            stage="Phase 2 (EDA/Prep)"
        ))
