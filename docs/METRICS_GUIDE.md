# EquiMed_DSS Metrics Guide

Comprehensive guide to all 19 metrics in the EquiMed_DSS library.

## Table of Contents

1. [Domain 1: Reliability and Calibration Assessment](#domain-1-reliability-and-calibration-assessment)
2. [Domain 2: Fairness, Equity, and Ethics Assessment](#domain-2-fairness-equity-and-ethics-assessment)
3. [Domain 3: Governance and Transparency Assessment](#domain-3-governance-and-transparency-assessment)
4. [Appendix: Advanced Metrics](#appendix-advanced-metrics)

---

## Domain 1: Reliability and Calibration Assessment

### 1. Dynamic Fairness Ratio (DFR)

**Purpose**: Measures how well a model maintains performance consistency across dynamic conditions.

**Formula**: DFR = Dynamic Metric / Baseline Metric

**Clinical Relevance**: In healthcare, models must perform consistently across different patient populations, time periods, and clinical settings. DFR helps identify when performance degrades in real-world deployment.

**Implementation**:
```python
from equimed_dss.domain1 import DynamicFairnessRatio

dfr = DynamicFairnessRatio()
result = dfr.calculate_dfr(baseline_metric=0.85, dynamic_metric=0.80)
```

**Interpretation**:
- **DFR ≥ 0.9**: Acceptable performance retention
- **0.8 ≤ DFR < 0.9**: Moderate degradation (investigate)
- **DFR < 0.8**: Significant degradation (requires intervention)

**Use Cases**:
- Monitoring model performance over time
- Comparing performance across different hospitals or sites
- Evaluating robustness to distribution shift

---

### 2. Expected Calibration Score (ECS)

**Purpose**: Quantifies how well predicted probabilities match actual outcomes.

**Formula**: ECS = Σ |Predicted Probability - Actual Frequency| / N_bins

**Clinical Relevance**: Calibration is critical in clinical decision-making. If a model predicts 80% probability of disease, approximately 80% of such cases should actually have the disease.

**Implementation**:
```python
from equimed_dss.domain1 import ExpectedCalibrationScore

ecs = ExpectedCalibrationScore()
predictions = [0.9, 0.8, 0.7, 0.6, 0.5]
actuals = [1, 1, 0, 1, 0]
result = ecs.calculate_ecs(predictions, actuals, n_bins=5)
```

**Interpretation**:
- **ECS < 0.05**: Well-calibrated (excellent)
- **0.05 ≤ ECS < 0.1**: Moderately calibrated (acceptable)
- **ECS ≥ 0.1**: Poorly calibrated (requires recalibration)

**Use Cases**:
- Risk prediction models
- Diagnostic support systems
- Treatment recommendation systems

---

### 3. Intraclass Correlation Coefficient (ICC)

**Purpose**: Assesses consistency and reliability of measurements across different raters, conditions, or time points.

**Formula**: ICC = (Between-group variance - Within-group variance) / Total variance

**Clinical Relevance**: Essential for evaluating inter-rater reliability, test-retest reliability, and consistency of AI predictions across different conditions.

**Implementation**:
```python
from equimed_dss.domain1 import IntraclassCorrelationCoefficient

icc = IntraclassCorrelationCoefficient()
ratings = [[3, 4, 3], [5, 5, 4], [2, 3, 2]]  # Multiple raters
result = icc.calculate_icc(ratings)
```

**Interpretation**:
- **ICC > 0.75**: Excellent reliability
- **0.6 ≤ ICC ≤ 0.75**: Good reliability
- **0.4 ≤ ICC < 0.6**: Moderate reliability
- **ICC < 0.4**: Poor reliability

**Use Cases**:
- Validating AI systems against multiple clinicians
- Assessing temporal stability of predictions
- Evaluating consistency across clinical sites

---

## Domain 2: Fairness, Equity, and Ethics Assessment

### 4. Hierarchical Equity Ratio (HER)

**Purpose**: Measures equity across demographic groups by comparing performance metrics relative to a reference group.

**Formula**: HER_group = Performance_group / Performance_reference

**Clinical Relevance**: Ensures AI systems don't systematically disadvantage certain demographic groups. Based on the 4/5ths rule from employment discrimination law.

**Implementation**:
```python
from equimed_dss.domain2 import HierarchicalEquityRatio

her = HierarchicalEquityRatio()
scores = {
    'White': 0.85,
    'Black': 0.68,
    'Hispanic': 0.75,
    'Asian': 0.87
}
her_scores = her.calculate_her(scores, reference_group='White')
gini = her.calculate_bias_gini(list(scores.values()))
```

**Interpretation**:
- **0.8 ≤ HER ≤ 1.25**: Equitable performance
- **HER < 0.8 or HER > 1.25**: Potential disparity

**Bias-Gini Index**:
- **Gini < 0.2**: Low dispersion (equitable)
- **0.2 ≤ Gini < 0.4**: Moderate dispersion
- **Gini ≥ 0.4**: High dispersion (concerning)

**Use Cases**:
- Auditing diagnostic algorithms for racial bias
- Evaluating treatment recommendation fairness
- Risk stratification equity analysis

---

### 5. Harm-Adjusted Fairness Gap (HAFG)

**Purpose**: Quantifies fairness gaps weighted by the clinical harm of different error types.

**Formula**: HAFG = |Harm_group1 - Harm_group2| / max(Harm_group1, Harm_group2)

**Clinical Relevance**: Not all errors are equal. False negatives in cancer screening carry much more harm than false positives. HAFG accounts for this reality.

**Implementation**:
```python
from equimed_dss.domain2 import HarmAdjustedFairnessGap

hafg = HarmAdjustedFairnessGap()
group1_errors = {'fn': 5, 'fp': 10}  # False negatives, false positives
group2_errors = {'fn': 2, 'fp': 5}
result = hafg.calculate_hafg(
    group1_errors,
    group2_errors,
    fn_harm_weight=10.0,  # False negative much worse
    fp_harm_weight=1.0
)
```

**Interpretation**:
- **HAFG < 0.1**: Minimal harm disparity (acceptable)
- **0.1 ≤ HAFG < 0.2**: Moderate harm disparity (investigate)
- **HAFG ≥ 0.2**: Significant harm disparity (requires intervention)

**Use Cases**:
- Cancer screening algorithms
- Sepsis prediction systems
- Emergency triage systems

---

### 6. Ethical Risk Index (ERI)

**Purpose**: Aggregates ethical violations weighted by severity to provide an overall ethical risk score.

**Formula**: ERI = Σ(Severity_i × Frequency_i) / Total_Outputs

**Clinical Relevance**: Systematically tracks and quantifies ethical issues like privacy violations, bias incidents, and safety concerns.

**Implementation**:
```python
from equimed_dss.domain2 import EthicalRiskIndex

eri = EthicalRiskIndex()
violations = [
    {'severity': 2.5, 'type': 'privacy'},
    {'severity': 1.0, 'type': 'bias'},
    {'severity': 5.0, 'type': 'safety'}
]
result = eri.calculate_eri(violations, n_total_outputs=1000)
```

**Severity Scale**:
- **1-2**: Minor (e.g., suboptimal communication)
- **3-4**: Moderate (e.g., privacy concern)
- **5-7**: Severe (e.g., patient harm)
- **8-10**: Critical (e.g., life-threatening)

**Interpretation**:
- **ERI < 0.05**: Low ethical risk (acceptable)
- **0.05 ≤ ERI < 0.1**: Moderate ethical risk (monitor)
- **ERI ≥ 0.1**: High ethical risk (immediate action)

**Use Cases**:
- AI ethics monitoring dashboards
- Regulatory compliance reporting
- Risk management systems

---

### 7. Intersectional Bias Score (IBS)

**Purpose**: Detects bias in intersectional subgroups (e.g., Black women, elderly Asian men) using distance-based outlier detection.

**Formula**: Distance = ||Performance_subgroup - Mean_performance_all||

**Clinical Relevance**: Single-axis fairness analysis misses intersectional discrimination. A system might be fair for "women" and "elderly" separately but unfair for "elderly women."

**Implementation**:
```python
from equimed_dss.domain2 import IntersectionalBiasScore

ibs = IntersectionalBiasScore()
vectors = {
    'White_Male': np.array([0.85, 0.90, 0.88]),
    'Black_Female': np.array([0.50, 0.55, 0.52]),  # Outlier
    'Hispanic_Male': np.array([0.82, 0.87, 0.85]),
    'Asian_Female': np.array([0.88, 0.91, 0.89])
}
result = ibs.calculate_subgroup_similarity(vectors)
```

**Interpretation**:
- Identifies subgroups with significantly different performance
- Higher distance = greater intersectional bias
- Threshold typically set at 1.5× standard deviation

**Use Cases**:
- Intersectional fairness audits
- Multi-dimensional bias detection
- Equity impact assessments

---

## Domain 3: Governance and Transparency Assessment

### 8. Temporal Fairness Drift (TFD)

**Purpose**: Tracks how fairness metrics change over time using statistical process control methods.

**Method**: Uses control charts with 3-sigma limits to detect statistically significant drift.

**Clinical Relevance**: Models can degrade over time due to distribution shift, data drift, or changing clinical practices. TFD provides early warning.

**Implementation**:
```python
from equimed_dss.domain3 import TemporalFairnessDrift

tfd = TemporalFairnessDrift()
time_series = [0.85, 0.84, 0.86, 0.83, 0.75, 0.84, 0.82]
result = tfd.calculate_drift(time_series)
```

**Interpretation**:
- **Stable Process**: All points within 3σ control limits
- **Warning**: 2 of 3 consecutive points > 2σ
- **Out of Control**: Any point > 3σ

**Use Cases**:
- Continuous model monitoring
- Performance degradation detection
- Trigger for model retraining

---

### 9. Audit Traceability Score (ATS)

**Purpose**: Measures the completeness and quality of audit trails for AI system decisions.

**Formula**: ATS = Σ(Completeness_i) / Total_Decisions

**Clinical Relevance**: For regulatory compliance and accountability, every AI decision must be traceable with complete metadata.

**Implementation**:
```python
from equimed_dss.domain3 import AuditTraceabilityScore

ats = AuditTraceabilityScore()
logs = [
    {'timestamp': True, 'user': True, 'action': True, 'details': True},
    {'timestamp': True, 'user': True, 'action': False, 'details': True},
    {'timestamp': True, 'user': False, 'action': True, 'details': False}
]
result = ats.calculate_ats(logs)
```

**Required Fields**:
- Timestamp
- User/System identifier
- Action taken
- Decision details
- Input data snapshot
- Model version

**Interpretation**:
- **ATS ≥ 0.9**: Excellent traceability
- **0.7 ≤ ATS < 0.9**: Adequate traceability
- **ATS < 0.7**: Poor traceability (non-compliant)

**Use Cases**:
- HIPAA compliance
- FDA regulatory submissions
- Malpractice liability protection

---

### 10. Governance Compliance Index (GCI)

**Purpose**: Assesses adherence to governance frameworks, regulations, and best practices.

**Formula**: GCI = N_requirements_met / N_total_requirements

**Clinical Relevance**: Healthcare AI must comply with multiple regulatory frameworks (FDA, HIPAA, state laws, hospital policies).

**Implementation**:
```python
from equimed_dss.domain3 import GovernanceComplianceIndex

gci = GovernanceComplianceIndex()
requirements = [
    True,   # FDA 510(k) clearance
    True,   # HIPAA compliance
    False,  # State-specific regulations
    True,   # Institutional review
    True    # Ethics committee approval
]
result = gci.calculate_gci(requirements)
```

**Common Requirements**:
- Regulatory approvals
- Privacy compliance
- Security standards
- Clinical validation
- Ethics review
- Documentation standards

**Interpretation**:
- **GCI = 1.0**: Full compliance
- **0.8 ≤ GCI < 1.0**: Mostly compliant (address gaps)
- **GCI < 0.8**: Non-compliant (major issues)

**Use Cases**:
- Pre-deployment readiness assessment
- Regulatory audit preparation
- Continuous compliance monitoring

---

## Appendix: Advanced Metrics

### Advanced Reliability Metrics

#### Bias Concentration Index (BCI)
Measures whether bias is concentrated in specific subgroups or distributed evenly.

#### Subgroup Performance Analyzer (SPA)
Statistical analysis of performance variations across demographic subgroups.

#### Bland-Altman Bias
Assesses agreement between two measurement methods using difference vs. mean plots.

### Advanced Information-Theoretic Metrics

#### Maximal Information Coefficient (MIC)
Detects both linear and non-linear relationships between variables.

#### Jensen-Shannon Divergence (JSD)
Symmetric measure of similarity between two probability distributions.

#### Wasserstein Distance (WD)
Optimal transport distance between distributions, sensitive to tail behavior.

### Advanced Network and Governance Metrics

#### Network Modularity (NM)
Analyzes community structure in bias correlation networks.

#### Topic Sensitivity (TS)
Measures how much performance varies across different clinical topics or conditions.

#### Risk Cascade Score (RCS)
Quantifies how risks propagate through interconnected system components.

---

## Choosing the Right Metrics

### For Diagnostic Systems
- **Primary**: ECS (calibration), HER (equity), TFD (drift)
- **Secondary**: DFR (robustness), HAFG (harm analysis)

### For Risk Prediction
- **Primary**: ECS (calibration), DFR (consistency), TFD (drift)
- **Secondary**: HER (equity), IBS (intersectional fairness)

### For Treatment Recommendation
- **Primary**: HAFG (harm), ERI (ethics), ICC (reliability)
- **Secondary**: HER (equity), GCI (compliance)

### For Regulatory Submission
- **Primary**: GCI (compliance), ATS (traceability), ICC (reliability)
- **Secondary**: All fairness metrics (HER, HAFG, IBS)

---

## Best Practices

1. **Establish Baselines**: Measure all metrics before deployment
2. **Continuous Monitoring**: Track metrics over time, not just at deployment
3. **Intersectional Analysis**: Always check intersectional subgroups
4. **Clinical Context**: Interpret metrics in clinical context, not just statistically
5. **Stakeholder Input**: Involve clinicians, patients, and ethicists in interpretation
6. **Documentation**: Document all metric calculations and interpretations
7. **Action Thresholds**: Define clear thresholds that trigger investigations or interventions

---

## References and Further Reading

1. Rajkomar, A., et al. (2018). "Ensuring Fairness in Machine Learning to Advance Health Equity"
2. FDA (2021). "Artificial Intelligence/Machine Learning-Based Software as a Medical Device"
3. Obermeyer, Z., et al. (2019). "Dissecting racial bias in an algorithm used to manage the health of populations"
4. Chen, I. Y., et al. (2021). "Ethical Machine Learning in Healthcare"

---

For implementation examples, see the `examples/` directory in the repository.
