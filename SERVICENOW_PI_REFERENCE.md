# ServiceNow Predictive Intelligence (PI) — Compact Reference

## 1. Core Idea

Predictive Intelligence (PI) = ServiceNow's Machine Learning capability that learns from historical records and predicts outcomes for new records.

Core lifecycle:

Data → Train → Predict → Integrate → Monitor → Tune → Retrain

Primary goals:
- Reduce manual classification/routing
- Reduce reassignments and handling time
- Improve resolution speed and consistency
- Surface similar/relevant information
- Continuously improve from actual outcomes

## 2. Main PI Frameworks

| Framework | Purpose | Example |
|---|---|---|
| Classification | Predict a categorical value | Category / Assignment Group |
| Similarity | Find similar records | Similar resolved incidents |
| Regression | Predict a numeric value | MTTR |
| Clustering | Group records by common patterns | Incident topic clusters |

For Incident Classification, Classification is the primary framework.

## 3. Classification Terminology

### Input
Information/features used by the model to make a prediction.

Examples:
- Short Description
- Description
- Caller
- Location
- Product / CI / Service

### Output / Target
The field/attribute that the model is trying to predict.

Examples:
- Assignment Group
- Category
- Subcategory
- Priority

### Class
A specific possible value of the output field.

Example:

Output = Category
Classes = Network, Software, Hardware, Database

Or:

Output = Assignment Group
Classes = Network Support, Service Desk, Application Support

### Historical Training Record

For a resolved incident:

Inputs (X) → Short Description, Description, etc.
Output (Y)  → Actual Assignment Group

During training, the actual output is the answer the model learns to predict.

For a new incident, the output is unknown and PI predicts it.

## 4. Data Requirements

### Qualified Dataset

Records suitable for learning:
- Correct/validated values
- Preferably closed/resolved records
- Relevant to the use case
- Required input fields populated
- Minimal bad/duplicate/noisy data

### Diverse Dataset

Data should represent the different real-world cases the model will see.

Example:

Network   = 2,500
Software  = 3,500
Hardware  = 1,500
Database  = 1,000

Diversity means:
- Different classes are represented
- Each important class has enough examples
- There is variation within each class

### Recent + Relevant + Diverse

Do not blindly use all historical records. Training data should represent the current process and environment.

### Training Volume

Referenced ServiceNow material:
- 10,000 = minimum mentioned for classification training
- ~30,000 = recommended volume discussed
- Exact limits can vary by ServiceNow release/product/configuration

### Minimum Records Per Class

This is counted from the OUTPUT/TARGET field, not the input fields.

Example:

Target = Assignment Group

Network Support = 500
Software Support = 300
Rare Group = 18

The class with too few qualifying records may be excluded from training/prediction.

Important:
- Webinar/classic PI material uses 30 records/class
- Newer ServiceNow releases can expose the minimum as a configurable setting
- Always verify the exact threshold on the target corporate instance

## 5. Classification Configuration

Configure:

1. Table
2. Input fields
3. Output field
4. Training filter/conditions
5. Language, when applicable
6. Stop words, when applicable
7. Retraining frequency
8. Advanced settings

Example:

Table = Incident
Inputs = Short Description, Description
Output = Assignment Group
Training filter = Closed/Resolved incidents with valid Assignment Group
Retraining = Scheduled/as required

Important:
Only use information genuinely available at prediction time.

Never use target/future/resolution information as an input.

## 6. Training Flow

Historical incidents
    ↓
Training filter
    ↓
Qualified + Relevant + Diverse records
    ↓
Input fields (X) + Actual Output (Y)
    ↓
Classification training
    ↓
Trained solution
    ↓
Solution statistics

## 7. Key Metrics

### Precision = Correct / Predicted

Question:
"When PI predicts this class, how often is it correct?"

Example:

PI predicts Network = 100
Correct Network = 80

Precision = 80 / 100 = 80%

Memory word: CORRECTNESS

### Recall = Correct / Actual

Question:
"Of all incidents that actually belong to this class, how many did PI detect?"

Example:

Actual Network = 100
Correctly detected Network = 80

Recall = 80 / 100 = 80%

Memory word: DETECTION

Recall is generally evaluated per class.

### Coverage

Question:
"Out of the records presented to the model, how many receive a prediction?"

Example:

100 incidents
PI predicts for 80

Coverage = 80%

Memory word: REACH

Coverage ≠ Confidence.

### Net Automation

Referenced approximately as:

Net Automation ≈ Precision × Coverage

## 8. Precision vs Coverage

### Higher Precision
- More selective
- Fewer predictions accepted
- Predictions tend to be more accurate
- Coverage generally decreases

### Higher Coverage
- More predictions
- More records receive automation
- Lower-confidence predictions may be accepted
- Precision may decrease

General trade-off:

Higher threshold/selectivity → Precision ↑, Coverage ↓

Lower threshold/selectivity → Coverage ↑, Precision ↓

There is no universal best value. Business risk determines the preferred operating point.

## 9. Confidence

Confidence = confidence signal for an individual prediction.

Example:

Incident = "VPN is not working"
Prediction = Network
Confidence = 92%

Confidence is NOT:

Precision × Coverage

Conceptual flow:

New incident
    ↓
Model prediction
    ↓
Confidence
    ↓
Compare with class threshold / prediction policy
    ↓
Accept prediction OR do not use prediction

Definitions:

Confidence = per-record signal
Precision  = correctness across accepted predictions
Recall     = detection of actual class instances
Coverage   = proportion of records receiving predictions

## 10. Class-Level Tuning

PI supports class-level Precision/Coverage operating points.

Example:

Network Support
→ Higher precision
→ Lower coverage acceptable

Software Support
→ More coverage
→ Some lower precision acceptable

A class can be excluded from prediction using the supported 100 Precision / 0 Coverage operating point where applicable.

Why class-level tuning matters:

Different classes have different business risks.

Example:
A small Network team may prefer fewer but highly accurate predictions because incorrect routing causes costly reassignment.

## 11. Testing

Before production:
- Test individual records
- Use batch testing where supported
- Review predicted class
- Review confidence
- Check Precision/Recall/Coverage
- Validate using realistic incidents

Do not judge the model using overall accuracy alone.

## 12. Deployment / Activation

A trained model must be connected to the business process.

Depending on the ServiceNow product/release:
- Business Rules
- Flow Designer / Workflow Studio
- Task Intelligence for ITSM
- Workspace recommendations
- Monitoring/background mode

Operational flow:

New Incident
    ↓
Prediction triggered
    ↓
Classification model
    ↓
Prediction + Confidence
    ↓
Threshold / Business Policy
    ↓
Recommendation / Autofill / Automation / Human fallback

## 13. Human-in-the-Loop Strategy

### Shadow / Monitoring Mode
Model predicts but does not affect the workflow.

### Assist / Recommendation Mode
Model recommends and agent validates/changes it.

### Automation Mode
High-confidence predictions are automatically applied.

Recommended approach:
Shadow → Assist → Controlled Automation

## 14. Predictor Results / Feedback Loop

Prediction
    ↓
Agent accepts/changes it
    ↓
Incident resolved/closed
    ↓
Final value recorded
    ↓
Prediction vs Final Value
    ↓
Performance updated

Continuous loop:

Predict → Observe → Measure → Tune → Retrain

## 15. Monitoring

Monitor:
- Precision
- Recall
- Coverage
- Confidence
- Class distribution
- Correct vs incorrect predictions
- Accepted vs changed predictions
- Reassignment rate
- Resolution/handling time
- Data drift
- Process changes

Also measure business impact:
- Reassignment ↓
- Manual effort ↓
- Resolution time ↓
- First-time correct assignment ↑

## 16. Retraining

Retraining should consider:
- New incidents
- Process changes
- New classes/assignment groups
- New terminology
- Performance degradation
- Data drift

Retraining is continuous, not one-time.

## 17. PI vs Assignment Rules

Assignment Rule:

Condition X → Fixed Action Y

Deterministic and rigid.

Predictive Intelligence:

Historical patterns + Current Inputs
        ↓
Predicted Outcome

Data-driven and adaptive.

Use rules for explicit business policies/exceptions.
Use ML for pattern-based prediction.

## 18. Data Leakage

Never use information that becomes available after the prediction point.

Bad:

Input = Final Assignment Group
Output = Assignment Group

Bad:

Input = Resolution Code entered after closure
Output = Assignment Group at creation

Good:

Input = Short Description + Description + available context
Output = Assignment Group

## 19. Recommended Incident Classification Architecture

Historical ServiceNow Incidents
            ↓
       DATA HEALTH CHECK
            ↓
Qualified + Recent + Relevant + Diverse
            ↓
        Input / Feature Set
            ↓
     Classification Solution
            ↓
          Train Model
            ↓
 Precision / Recall / Coverage
            ↓
     Class-Level Tuning
            ↓
        Test Solution
            ↓
           Deploy
            ↓
       New Incident
            ↓
Prediction + Confidence
            ↓
    Threshold / Policy
        ↙          ↘
    Accept       Reject
      ↓             ↓
Assignment       Human
 Group           Handling
      ↓
Incident Resolution
      ↓
Final Actual Value
      ↓
Predictor Results
      ↓
Performance + Business Metrics
      ↓
Tune / Retrain
      ↺

## 20. Recommended Initial Configuration

Use Case:
Incident Assignment Group Classification

Table:
Incident

Output:
Assignment Group

Possible secondary output:
Category

Initial inputs:
- Short Description
- Description
- Other fields available at prediction time

Training data:
- Closed/resolved
- Validated
- Representative
- Relevant
- Diverse

Minimum records/class:
Use the threshold supported by the target ServiceNow release.
30 was the webinar/classic PI reference.

Primary metric:
Precision

Secondary metrics:
Recall + Coverage

Prediction:
Confidence + native class threshold

Low confidence:
Human fallback

Monitoring:
Predictor Results + model metrics + business metrics

Retraining:
Periodic and/or triggered by performance/data/process changes

## 21. Recommended Decision Logic

Is the training data trustworthy?
    ↓ No → Fix data
    ↓ Yes

Does every important class have enough representative records?
    ↓ No → Improve/review data
    ↓ Yes

Is Precision sufficient for the business risk?
    ↓ No → Tune data/model/class thresholds
    ↓ Yes

Is Coverage useful?
    ↓ No → Gradually increase coverage
    ↓ Yes

Deploy in monitoring/assist mode
    ↓
Validate real outcomes
    ↓
Automate only where risk is acceptable
    ↓
Monitor continuously
    ↓
Retrain/tune when required

## 22. Key Words — Memory Map

PI              = Machine Learning in ServiceNow
Classification  = Predict a category/value
Input           = Information used to predict
Output/Target   = Field being predicted
Class           = Specific value of output
Qualified Data  = Trustworthy training records
Diverse Data    = Representative variety
Precision       = Correctness
Recall          = Detection
Coverage        = Reach
Confidence      = Per-prediction confidence signal
Threshold       = Required confidence/operating point
Training        = Learn X → Y from historical data
Prediction      = Apply model to new X
Predictor Result= Prediction vs Final Actual
Tuning          = Adjust operating point/model/data
Retraining      = Learn again from newer/better data
Data Leakage    = Future/target information used incorrectly
Drift           = Real-world data/process changes
Human Fallback  = Manual handling when prediction is unsuitable

## 23. Current ServiceNow Version Note

The webinar material reflects the older Predictive Intelligence / Workbench experience.

Newer ServiceNow ITSM releases have moved toward Task Intelligence for ITSM for newer guided ITSM prediction workflows.

Therefore, before configuring the corporate instance:

1. Check ServiceNow release/version.
2. Check installed/licensed AI/ML applications.
3. Use the corresponding current configuration path.
4. Do not assume the webinar UI is identical to the corporate instance.

## 24. Final Mental Model

GOOD DATA
    ↓
RIGHT INPUTS
    ↓
RIGHT OUTPUT / TARGET
    ↓
ENOUGH EXAMPLES PER CLASS
    ↓
TRAIN
    ↓
PRECISION / RECALL / COVERAGE
    ↓
CONFIDENCE + THRESHOLD
    ↓
PREDICT
    ↓
HUMAN OR AUTOMATION
    ↓
FINAL OUTCOME
    ↓
MONITOR
    ↓
TUNE
    ↓
RETRAIN

## One-Line Takeaway

ServiceNow Predictive Intelligence learns from qualified historical data to predict an output class for new records, uses confidence/thresholds to control predictions, balances Precision and Coverage, and continuously improves through real outcomes and retraining.
