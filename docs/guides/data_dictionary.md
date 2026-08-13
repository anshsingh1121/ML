# Enterprise Data Dictionary — ServiceNow Incident Schema (`v1.5.0`)

**Organization:** First Citizens Bank — Incident Intelligence Platform (IIP)  
**Schema Version:** `1.5.0`  
**Total Attributes:** 38  
**Data Classification:** Internal Confidential — Banking IT Operations  

---

## Attribute Specification Matrix

| Attribute Name | Datatype | Nullable | Example Value | Description | Business Meaning | ML Usage |
|---|:---:|:---:|---|---|---|:---:|
| `incident_number` | `String` | No | `INC0010042` | Unique system identifier for the incident record. | Primary key for ticket auditing and cross-system correlation. | **Metadata ID** (Excluded from modeling) |
| `opened_at` | `DateTime` | No | `2026-03-14 08:30:00` | UTC timestamp when the ticket was created in ServiceNow. | Establishes the baseline start time for SLA measurement and operational triage. | **Temporal Feature** (Extract hour, day of week, month) |
| `resolved_at` | `DateTime` | Yes | `2026-03-14 12:45:00` | UTC timestamp when support engineers restored service. | Marks the end of active disruption and stops operational SLA timers. | **Post-Resolution Target / Leakage** (Strictly excluded at triage) |
| `closed_at` | `DateTime` | Yes | `2026-03-19 12:45:00` | UTC timestamp when the ticket was administratively closed after verification. | Final lifecycle state after user acceptance or auto-close timeout (5 days post-resolution). | **Post-Resolution / Leakage** (Excluded from modeling) |
| `priority` | `Integer` | No | `2` | 1 (Critical) to 5 (Planning). Calculated from Impact and Urgency matrix. | Dictates escalation velocity, on-call paging, and SLA target windows (e.g., P1 = 4h target). | **Key Numerical / Categorical Predictor** |
| `impact` | `Integer` | No | `2` | 1 (High) to 3 (Low). Measures scope of business disruption. | Reflects financial risk, affected customer count, or regulatory reporting impact. | **Numerical Predictor** |
| `urgency` | `Integer` | No | `2` | 1 (High) to 3 (Low). Measures time criticality of restoration. | Reflects operational deadline pressure and payment processing cutoffs. | **Numerical Predictor** |
| `severity` | `Integer` | No | `2` | 1 (Critical) to 3 (Low). System-perceived fault intensity. | Correlates technical log alarms directly to ticket priority. | **Numerical Predictor** |
| `state` | `Integer` | No | `6` | 1 (New) to 8 (Canceled). Current operational phase of the ticket. | Tracks lifecycle transition (1=New, 2=In Progress, 3=On Hold, 6=Resolved, 7=Closed). | **Lifecycle Status / Filter** |
| `category` | `String` | No | `Core Banking` | Top-level IT domain taxonomy classification. | Segregates tickets across functional divisions (Payment Systems, Infrastructure, Security, Retail). | **Categorical Feature** (High importance predictor) |
| `subcategory` | `String` | No | `SWIFT Gateway` | Granular technical fault classification within Category. | Identifies specific sub-components or software modules experiencing failures. | **Categorical Feature** (Frequency/Target encoded) |
| `assignment_group` | `String` | No | `Core_Banking_L2` | Designated support team responsible for resolving the incident. | Enterprise routing destination where ticket ownership is assigned. | **Primary Target Label** (Multi-class classification target) |
| `assigned_to` | `String` | Yes | `engineer.42@fcb.com` | Specific support engineer assigned to the ticket within the group. | Tracks individual workload and engineering accountability. | **Metadata / Excluded** (High cardinality) |
| `business_service` | `String` | No | `SWIFT Payments` | Business-facing IT service catalog offering affected by the outage. | Links technical faults directly to banking revenue streams and customer channels. | **Categorical Feature** |
| `cmdb_ci` | `String` | No | `ci_swift_gw_prod_01` | Configuration Item (server, database, application) in CMDB. | Pinpoints exact hardware or software infrastructure asset where the defect resides. | **Categorical Feature / Graph Node** |
| `u_describe_customer_impact` | `String` | Yes | `IBM` | Third-party software or hardware u_describe_customer_impact associated with the CI. | Enables external SLA enforcement and u_describe_customer_impact escalations for proprietary hardware/software. | **Categorical Feature** |
| `caller` | `String` | No | `user.1042@fcb.com` | Bank employee, teller, or automated monitoring system that reported the issue. | Identifies origin channel and reporting persona. | **Metadata ID / Excluded** |
| `short_description` | `String` | No | `SWIFT Alliance Gateway payment processing failure` | Brief summary title of the incident reported by user or monitoring alert. | Primary text payload used by engineers for rapid triage and pattern matching. | **Text Feature** (SentenceTransformer embedding input) |
| `description` | `String` | No | `Users reporting SWIFT messages stuck in ACK_PENDING queue...` | Full diagnostic details, stack traces, and error logs describing the incident. | Deep technical context required to diagnose root cause and recommend specific fixes. | **Text Feature** (SentenceTransformer embedding & RAG input) |
| `close_notes` | `String` | Yes | `Restarted SWIFT messaging daemon and flushed stuck ACK queue.` | Engineer's documented resolution actions and technical fix details. | Historical knowledge base goldmine for recommending fixes to future identical incidents. | **Post-Resolution Target / RAG Knowledge Base** |
| `resolution_code` | `String` | Yes | `Solved (Permanently)` | Standardized taxonomy of how the incident was resolved. | Distinguishes permanent code fixes from temporary workarounds, user errors, or false alarms. | **Post-Resolution Leakage** (Excluded from triage prediction) |
| `resolution_time_hours` | `Float` | Yes | `2.45` | Total elapsed clock time in hours from `opened_at` to `resolved_at`. | Core operational KPI measuring Mean Time To Resolution (MTTR). | **Primary Regression Target** (Resolution Time Prediction) |
| `calendar_duration_hours` | `Float` | Yes | `2.45` | Elapsed calendar time in hours from opening to closure. | Measures overall lifecycle duration including administrative closure delays. | **Post-Resolution Leakage** |
| `business_duration_hours` | `Float` | Yes | `2.10` | Elapsed operational time in hours within standard banking business hours (8am-6pm). | Measures actual engineering effort during standard operating shifts. | **Post-Resolution Leakage** |
| `made_sla` | `Boolean` | No | `True` | Binary flag indicating whether the incident was resolved within authorized SLA targets. | Executive SLA compliance metric reporting service quality to banking regulators. | **Post-Resolution Target / Leakage** |
| `sla_status` | `String` | No | `Met` | Categorical SLA state (`Met` or `Breached`). | Human-readable SLA status shown on engineering dashboards. | **Post-Resolution Target / Leakage** |
| `sla_due` | `DateTime` | No | `2026-03-14 20:30:00` | Exact UTC deadline by which the incident must be resolved to avoid SLA breach. | Operational cutoff timestamp driving priority queue ordering and paging alerts. | **Temporal Feature / Target Calculation** |
| `u_caused_by` | `Integer` | No | `1` | Number of times the ticket was reassigned across different assignment groups. | Measures routing friction and initial misassignment waste across support tiers. | **Operational Efficiency KPI / Post-Resolution Leakage** |
| `reopen_count` | `Integer` | No | `0` | Number of times the ticket was reopened after initial resolution. | Indicates incomplete fixes or recurring underlying instability. | **Post-Resolution Leakage** |
| `problem_flag` | `Boolean` | No | `False` | Binary flag indicating if this incident triggered or linked to a formal Problem investigation (`PRB`). | Identifies systemic defects requiring architectural root cause analysis. | **Post-Resolution Outcome** |
| `problem_record` | `String` | Yes | `PRB001042` | Associated Problem investigation number if `problem_flag` is `True`. | Cross-references long-term engineering bug tracking and remediation tickets. | **Metadata Reference** |
| `change_request` | `String` | Yes | `CHG002011` | Associated Change Request number (`CHG`) if the incident was caused by or resolved via a change. | Audits whether infrastructure deployments or code releases caused production outages. | **Metadata Reference / Causal Feature** |
| `knowledge_linked` | `Boolean` | No | `True` | Binary flag indicating whether an existing Knowledge Base article (`KB`) was used to resolve the issue. | Quantifies knowledge base efficiency and self-service deflection rates. | **Post-Resolution Outcome** |
| `knowledge_base` | `String` | Yes | `KB0010042` | Specific Knowledge Base article ID linked by the resolving engineer. | Direct link to remediation procedures and standard work instructions. | **RAG Retrieval Citation Reference** |
| `contact_type` | `String` | No | `Alert` | Origin reporting channel (`Alert`, `Phone`, `Self-service`, `Email`). | Distinguishes automated monitoring alerts from human-reported tickets. | **Categorical Feature** |
| `location` | `String` | No | `HQ - New York` | Geographic banking facility or data center where the incident originated. | Isolates regional data center network outages or branch-specific hardware failures. | **Categorical Feature** |
| `duplicate_incident` | `String` | Yes | `INC0010039` | Parent ticket ID if this incident is marked as a duplicate (`State=8`). | Identifies ticket storms caused by a single underlying infrastructure failure. | **Metadata Relationship / Graph Link** |
| `parent_incident` | `String` | Yes | `INC0010012` | Parent ticket ID if this incident is part of a parent-child hierarchy. | Clusters related alerts under a single master incident for unified management. | **Metadata Relationship / Graph Link** |

---

## Domain Catalog Constraints

All categorical attributes are rigorously bounded by First Citizens Bank enterprise taxonomies (`BankingIncidentCatalog`):
- **Categories:** `Core Banking`, `Payment Systems`, `Infrastructure`, `Security`, `Application Services`, `Database Services`, `Network Services`, `Retail Banking`
- **Assignment Groups:** Segregated cleanly across L1, L2, L3, and specialized domain engineering squads (`Core_Banking_L2`, `SWIFT_Operations_L3`, `Database_DBA_L3`, `SOC_Security_L3`, `Network_Engineering_L3`, `Digital_Banking_L2`).
