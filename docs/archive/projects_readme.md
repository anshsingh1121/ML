# Pipeline Diagrams — Version History

This directory contains all versions of pipeline and architecture diagrams for the Incident Intelligence Platform.

**Rule:** Never overwrite previous versions. Always create a new versioned file.

## Version Log

| Version | Date | Description | Status |
|---|---|---|---|
| `pipeline_v1.md` | 2026-07-10 | Initial architecture — full end-to-end pipeline + ML pipeline detail | 🟡 Under Review |

## Naming Convention

- `pipeline_v{N}.md` — Mermaid source files
- `pipeline_v{N}.png` — Rendered PNG files (generated after approval)
- `architecture_v{N}.md` — System architecture diagrams
- `deployment_v{N}.md` — Deployment architecture diagrams
- `sequence_v{N}.md` — Sequence diagrams
- `component_v{N}.md` — Component diagrams
- `schema_v{N}.md` — Database schema diagrams

## Rendering

PNG versions will be generated from Mermaid sources using the Mermaid CLI or exported from the dashboard after implementation.
