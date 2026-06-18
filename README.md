# EquiMed_DSS

<div align="center">

<h3>A Comprehensive Python Library for Clinical AI Fairness Assessment</h3>

<p>Evaluate reliability, equity, governance, and intersectionality in clinical AI systems using <strong>37 metrics across five domains</strong></p>

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/johnmuteba/EquiMed_DSS/blob/master/LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

</div>

---

## Overview

**EquiMed_DSS** (Equitable Medical Decision Support System) provides a systematic framework for evaluating clinical AI systems across multiple dimensions of fairness, reliability, and governance. The library implements **37 metrics across five domains** specifically designed for healthcare applications where equity and safety are paramount.

### Key Features

| Feature | Description |
|---------|-------------|
| **37 Metrics** | Five domains (reliability, equity, governance, representation/robustness, technical-supplement fairness) plus geographic and advanced-appendix metrics |
| **Clinical AI Focus** | Designed specifically for healthcare applications |
| **Statistical Analyses** | HLM, Mediation Analysis, Network Statistics |
| **Publication-Ready Visualizations** | 6 manuscript-quality figure generators |
| **Multi-Format Data Support** | MySQL, CSV, TSV, JSON with automatic standardization |
| **Intersectional Analysis** | Detect bias across demographic combinations |
| **Geographic Equity** | BEMI and GCC measure evidence-burden mismatch and regional concentration |
| **Tidy Reporting Tables** | `export_table` renders metric results as markdown, LaTeX, or HTML |

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Data Format](#data-format)
- [Metrics Overview](#metrics-overview)
  - [Domain 1: Reliability & Calibration](#domain-1-reliability--calibration)
  - [Domain 2: Fairness, Equity & Ethics](#domain-2-fairness-equity--ethics)
  - [Domain 3: Governance & Transparency](#domain-3-governance--transparency)
  - [Domain 4: Representation & Robustness](#domain-4-representation--robustness)
  - [Domain 5: Technical-supplement Fairness](#domain-5-technical-supplement-fairness)
  - [Appendix: Advanced Metrics](#appendix-advanced-metrics)
  - Full formulas, clinical meaning, and runnable examples: see [docs/VIGNETTE.md](https://github.com/johnmuteba/EquiMed_DSS/blob/master/docs/VIGNETTE.md)
- [Statistical Analyses](#statistical-analyses)
- [Visualizations](#visualizations)
- [Examples](#examples)
- [Vignette](https://github.com/johnmuteba/EquiMed_DSS/blob/master/docs/VIGNETTE.md)
- [API Reference](#api-reference)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Install from Source

```bash
# Clone the repository
git clone https://github.com/johnmuteba/EquiMed_DSS.git
cd EquiMed_DSS

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package with dependencies
pip install -e .
```

### Install via pip

```bash
pip install equimed_dss
```

### Installing inside Jupyter or conda (read this if you hit ModuleNotFoundError)

The most common installation problem is installing into a *different* Python
than the one your notebook or environment actually runs. You see
`Successfully installed equimed_dss` in a terminal, then
`ModuleNotFoundError: No module named 'equimed_dss'` in Jupyter. This is an
environment mismatch, not a package problem. Install into the **running**
interpreter:

In a Jupyter notebook cell (installs into the active kernel, then restart the kernel):

```python
%pip install equimed_dss
```

From a terminal, target a specific interpreter explicitly:

```bash
python -m pip install equimed_dss          # uses THIS python
# conda example:
conda activate myenv && python -m pip install equimed_dss
```

Confirm the install is visible to your interpreter:

```python
import sys; print(sys.executable)          # which python is running
import equimed_dss; print(equimed_dss.__version__)
```

### Dependencies

```
numpy>=1.20.0
pandas>=1.3.0
scipy>=1.7.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
networkx>=2.6.0
statsmodels>=0.13.0
```

---

## Quick Start

### Generate Sample Data

```python
from equimed_dss.utils import SampleDataGenerator

# Generate synthetic clinical AI evaluation data
generator = SampleDataGenerator(random_state=42)
data = generator.generate_fairness_data(n_samples=1000)

print(f"Generated {len(data)} samples with columns: {list(data.columns)}")
# Output: Generated 1000 samples with columns: ['id', 'race', 'gender', 'age_group', 'prediction', 'actual', 'confidence']
```

### Calculate Fairness Metrics

```python
import numpy as np
from equimed_dss.domain2 import HierarchicalEquityRatio, HarmAdjustedFairnessGap

# Example: Calculate Hierarchical Equity Ratio across racial groups
her_metric = HierarchicalEquityRatio()
group_performance = {
    'White': 0.85,
    'Black': 0.78,
    'Hispanic': 0.80,
    'Asian': 0.87
}

her_scores = her_metric.calculate_her(group_performance)
gini = her_metric.calculate_bias_gini(list(group_performance.values()))

print(f"Equity Ratios: {her_scores}")
print(f"Bias-Gini Index: {gini:.4f}")
# Interpretation: Gini < 0.2 indicates low dispersion (good)
```

### Analyze Distributional Fairness

```python
from equimed_dss.appendix import JensenShannonDivergence, WassersteinDistance

# Compare prediction distributions between groups
group_a_predictions = np.array([0.9, 0.85, 0.78, 0.92, 0.88])
group_b_predictions = np.array([0.75, 0.70, 0.68, 0.72, 0.65])

# Jensen-Shannon Divergence
jsd = JensenShannonDivergence()
jsd_result = jsd.calculate_jsd(group_a_predictions, group_b_predictions)
print(f"JSD: {jsd_result['jsd']:.4f} - {jsd_result['interpretation']['verdict']}")

# Wasserstein Distance
wd = WassersteinDistance()
wd_result = wd.calculate_wd(group_a_predictions, group_b_predictions)
print(f"WD: {wd_result['wasserstein_distance']:.4f} - {wd_result['interpretation']['verdict']}")
```

---

## Data Format

### Required Data Structure

EquiMed_DSS expects data in a standardized format. Use the built-in utilities to convert your data:

```python
from equimed_dss.utils import CorpusLoader, DemographicProcessor

# Load and standardize your data
loader = CorpusLoader()

# From CSV
df = loader.load_from_csv('your_data.csv', text_column='clinical_notes')

# Validate format
validation = loader.validate_format(df)
print(validation)
```

### Expected Columns

| Column | Type | Required | Description |
|--------|------|----------|-------------|
| `id` | string | Yes | Unique identifier |
| `content` | string | For text analysis | Clinical text/notes |
| `prediction` | float | For fairness metrics | Model prediction (0-1) |
| `actual` | int | For fairness metrics | Ground truth (0 or 1) |
| `race` | string | For demographic analysis | Racial/ethnic group |
| `gender` | string | For demographic analysis | Gender identity |
| `age_group` | string | For demographic analysis | Age category |

### Sample Data Schema

```json
{
  "id": "patient_001",
  "content": "Patient presents with chest pain...",
  "prediction": 0.85,
  "actual": 1,
  "race": "Black",
  "gender": "Female",
  "age_group": "Middle Age"
}
```

---

## Metrics Overview

### Domain 1: Reliability & Calibration

| Metric | Abbreviation | Range | Ideal | Description |
|--------|--------------|-------|-------|-------------|
| Decision Flip Rate | DFR | [0, 1] | close to 0 | Decision instability under counterfactual inputs |
| Embedding Consistency Score | ECS | [0, 1] | higher | Embedding stability under perturbation |
| Inter-Rater Reliability (ICC 2,1) | ICC | [0, 1] | > 0.75 | Agreement across judges |

Every metric result carries a 95% confidence interval and prints it. Just
`print(result)` and the value is shown alongside its CI (the result is still a
plain dict, so `result['flip_rate']` etc. keep working):

```python
import numpy as np
from equimed_dss.domain1 import DecisionFlipRate, EmbeddingConsistencyScore, InterRaterReliability

# Decision Flip Rate (DFR): how often decisions flip under counterfactual inputs
dfr = DecisionFlipRate()
print(dfr.calculate_dfr(
    original_decisions=['ACS', 'ACS', 'non-cardiac', 'ACS'],
    counterfactual_decisions=['ACS', 'non-cardiac', 'non-cardiac', 'ACS'],
))
# DFR = 0.250 :: 95% CI [0.046; 0.699] (Wilson score)

# Embedding Consistency Score (ECS): stability of embeddings under perturbation
ecs = EmbeddingConsistencyScore()
original = np.random.RandomState(0).rand(10, 8)
perturbed = original + np.random.RandomState(1).normal(0, 0.05, (10, 8))
print(ecs.calculate_ecs(original, perturbed))
# ECS = 0.003 :: 95% CI [0.002; 0.005] (bootstrap)

# Inter-Rater Reliability (ICC 2,1): agreement across judges (subjects x raters)
icc = InterRaterReliability()
print(icc.calculate_icc_2_1(np.array([[3, 4, 3], [5, 5, 4], [2, 3, 2], [4, 4, 5]])))
# ICC(2,1) = 0.750 :: 95% CI [-0.500; 0.816] (bootstrap (over items))
```

### Domain 2: Fairness, Equity & Ethics

| Metric | Abbreviation | Range | Ideal | Description |
|--------|--------------|-------|-------|-------------|
| Hierarchical Equity Ratio | HER | [0, ∞) | 0.8-1.25 | Group equity (4/5ths rule) |
| Bias-Gini Index | BGI | [0, 1] | < 0.2 | Performance dispersion |
| Harm-Adjusted Fairness Gap | HAFG | [0, ∞) | < 0.1 | Clinical harm-weighted disparity |
| Ethical Risk Index | ERI | [0, ∞) | < 0.05 | Aggregated ethical violations |
| Intersectional Bias Score | IBS | varies | low | Subgroup outlier detection |

```python
from equimed_dss.domain2 import HierarchicalEquityRatio, HarmAdjustedFairnessGap, EthicalRiskIndex

# Harm-Adjusted Fairness Gap
hafg = HarmAdjustedFairnessGap()
result = hafg.calculate_hafg(
    group1_errors={'fn': 5, 'fp': 10},
    group2_errors={'fn': 2, 'fp': 5}
)
print(f"HAFG: {result['hafg']:.3f}")

# Ethical Risk Index
eri = EthicalRiskIndex()
result = eri.calculate_eri(
    violations=[{'severity': 2.5}, {'severity': 1.0}, {'severity': 5.0}],
    n_total_outputs=100
)
print(f"ERI: {result['eri']:.3f}")
```

### Domain 3: Governance & Transparency

| Metric | Abbreviation | Range | Ideal | Description |
|--------|--------------|-------|-------|-------------|
| Temporal Fairness Drift | TFD | varies | stable | Fairness degradation over time |
| Audit Traceability Score | ATS | [0, 1] | > 0.9 | Audit trail completeness |
| Governance Compliance Index | GCI | [0, 1] | 1.0 | Regulatory compliance |

```python
from equimed_dss.domain3 import TemporalFairnessDrift, AuditTraceabilityScore, GovernanceComplianceIndex

# Temporal Fairness Drift
tfd = TemporalFairnessDrift()
result = tfd.calculate_drift([0.85, 0.84, 0.86, 0.83, 0.75, 0.84])
print(f"Drift Detected: {result['drift_detected']}")

# Audit Traceability Score: fraction of audit records that are fully traceable
ats = AuditTraceabilityScore()
result = ats.calculate_ats(n_traceable=8, n_total=10)
print(f"ATS: {result['ats_score']:.3f} - {result['interpretation']['verdict']}")
```

### Domain 4: Representation & Robustness

| Metric | Abbreviation | Range | Ideal | Description |
|--------|--------------|-------|-------|-------------|
| Semantic Parity Gap | SPG | [0, ∞) | close to 0 | Embedding-centroid distance between identical cases differing only by a protected attribute |
| Clinical Hallucination Rate | CHR | [0, 1] | close to 0 | Fraction of response claims unsupported by the retrieved context (NLI entailment) |
| Instructional Vulnerability Index | IVI | [0, 1] | close to 0 | Proportion of decisions that flip under a biased/leading instruction |
| Geographic Representation Index | GRI | [0, 1] | higher | Set-based non-Western variety of represented locations |

```python
import numpy as np
from equimed_dss.domain4 import (
    SemanticParityGap, ClinicalHallucinationRate,
    InstructionalVulnerabilityIndex, GeographicRepresentationIndex,
)

# Clinical Hallucination Rate (CHR): unsupported claims at entailment threshold tau
chr_ = ClinicalHallucinationRate()
result = chr_.calculate_chr(support_scores=[0.9, 0.2, 0.4, 0.8, 0.1], tau=0.5)
print(f"CHR: {result['chr']:.3f} ({result['n_unsupported']}/{result['n_claims']} unsupported)")

# Instructional Vulnerability Index (IVI): decision flips under a biased prompt
ivi = InstructionalVulnerabilityIndex()
result = ivi.calculate_ivi(neutral_outputs=['acs', 'non_cardiac', 'acs', 'other'],
                           biased_outputs=['acs', 'acs', 'acs', 'other'])
print(f"IVI: {result['ivi_flip_rate']:.3f} ({result['n_flipped']}/{result['n_pairs']})")

# Semantic Parity Gap (SPG): centroid distance between two demographic embedding clusters
spg = SemanticParityGap()
rng = np.random.RandomState(0)
result = spg.calculate_spg(privileged_embeddings=rng.rand(20, 16),
                           marginalized_embeddings=rng.rand(20, 16) + 0.1)
print(f"SPG euclidean: {result['spg_euclidean']:.3f}; cosine: {result['spg_cosine']:.3f}")
```

### Domain 5: Technical-supplement Fairness

| Metric | Abbreviation | Range | Ideal | Description |
|--------|--------------|-------|-------|-------------|
| Intersectional Calibration Error | ICE | [0, 1] | close to 0 | Population-weighted calibration error across intersectional groups; `dICE` = max gap |
| Weighted Clinical Harm-Adjusted Fairness Gap | wHAFG | [0, 1] | close to 0 | Severity-weighted harm gap across groups |
| Lexical Diversity Disparity Index | LDDI | [0, ∞) | close to 0 | Root type-token-ratio gap across groups |
| Recommendation Entropy Gap | REG | [0, ∞) bits | close to 0 | Max-min recommendation-distribution entropy across groups |
| Counterfactual Parity Score | CPS | [0, 1] | 1.0 | Mean response similarity under a demographic swap (CFU = 1 − CPS) |
| Clinical Information Density Ratio | CIDR | [0, 1] | 1.0 | Group concept-density relative to the richest group |
| Diagnostic Completeness Index | DCI | [0, 1] | higher | Coverage of a reference differential set; `dDCI` = max gap |
| Uncertainty Quantification Gap | UQG | [0, ∞) | close to 0 | Hedging-density disparity across groups |
| Geographic Representation Bias Index | GRBI | [0, ∞) nats | close to 0 | KL divergence of corpus geography from disease burden |
| Healthcare System Stratified Fairness | HSSF | [0, 1] | close to 0 | Population-weighted within-system demographic gap |
| Intersectional Shapley Fairness Value | ISFV | varies | low | Shapley attribution of disparity to each attribute + interactions |
| Semantic Robustness Parity Index | SRPI | [0, 1] | 1.0 | Min/max paraphrase robustness across groups |

```python
from equimed_dss.domain5 import (
    CounterfactualParityScore, GeographicRepresentationBiasIndex,
    IntersectionalShapleyFairnessValue,
)

# Counterfactual Parity Score (CPS) and counterfactual unfairness (CFU)
cps = CounterfactualParityScore()
result = cps.calculate_cps([0.95, 0.88, 0.91, 0.86])
print(f"CPS: {result['cps']:.3f}; CFU: {result['cfu']:.3f}")

# Geographic Representation Bias Index (GRBI): KL of corpus geography from burden
grbi = GeographicRepresentationBiasIndex()
result = grbi.calculate_grbi(
    corpus_counts={'AMRO': 78, 'EURO': 11, 'WPRO': 8, 'EMRO': 3, 'AFRO': 0, 'SEARO': 0},
    burden_shares={'AMRO': 0.11, 'EURO': 0.19, 'WPRO': 0.10, 'EMRO': 0.23, 'AFRO': 0.15, 'SEARO': 0.21},
)
print(f"GRBI: {result['grbi']:.3f} nats")

# Intersectional Shapley Fairness Value (ISFV): attribute attribution of disparity
isfv = IntersectionalShapleyFairnessValue()
result = isfv.calculate_isfv(
    attributes={'race': ['A', 'A', 'B', 'B'], 'sex': ['M', 'F', 'M', 'F']},
    outcomes=[0.8, 0.7, 0.5, 0.6],
)
print(f"ISFV by attribute: {result['shapley_by_attribute']}")
```

### Appendix: Advanced Metrics

These 9 additional metrics provide deeper statistical analysis:

| Metric | Class | Range | Threshold | Description |
|--------|-------|-------|-----------|-------------|
| Bootstrap Confidence Intervals | `BootstrapConfidenceIntervals` | varies | CI width < 0.05 | Robust uncertainty estimation |
| Statistical Power Analysis | `StatisticalPowerAnalysis` | [0, 1] | ≥ 0.8 | Sample size adequacy |
| Bias Concentration Index | `BiasConcentrationIndex` | [0, 1] | > 0.7 | Bias distribution across groups |
| Mutual Information Content | `MutualInformationContent` | [0, ∞) | < 0.1 | Demographic information leakage |
| Jensen-Shannon Divergence | `JensenShannonDivergence` | [0, 1] | < 0.1 | Distributional similarity |
| Wasserstein Distance | `WassersteinDistance` | [0, ∞) | < 0.1 | Optimal transport distance |
| Network Modularity | `NetworkModularity` | [-1, 1] | > 0.3 | Metric clustering structure |
| Transparency Score | `TransparencyScore` | [0, 1] | > 0.7 | Explanation quality |
| Robustness Certification | `RobustnessCertificationScore` | [0, 1] | > 0.8 | Perturbation stability |

```python
from equimed_dss.appendix import (
    BootstrapConfidenceIntervals,
    StatisticalPowerAnalysis,
    BiasConcentrationIndex,
    MutualInformationContent,
    JensenShannonDivergence,
    WassersteinDistance,
    NetworkModularity,
    TransparencyScore,
    RobustnessCertificationScore
)
import numpy as np

# Bootstrap Confidence Intervals
bci = BootstrapConfidenceIntervals(n_bootstrap=1000, random_state=42)
data = np.random.normal(0.85, 0.05, 100)
result = bci.calculate_bci(data)
print(f"95% CI: [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]")
print(f"Stability: {result['interpretation']['stability']}")

# Statistical Power Analysis
spa = StatisticalPowerAnalysis()
result = spa.calculate_sample_size(effect_size=0.5, power=0.8)
print(f"Required N per group: {result['n_per_group']}")

# Bias Concentration Index
bci_metric = BiasConcentrationIndex()
result = bci_metric.calculate_bci([0.3, 0.25, 0.25, 0.2])  # Bias proportions
print(f"BCI: {result['bci']:.4f} - {result['interpretation']['distribution']}")

# Mutual Information Content
mic = MutualInformationContent()
demographics = np.array([0, 0, 1, 1, 2, 2, 0, 1])  # Encoded demographics
outcomes = np.array([1, 1, 0, 0, 1, 0, 1, 0])      # Model outcomes
result = mic.calculate_mic(demographics, outcomes)
print(f"MIC: {result['mic']:.4f} - {result['interpretation']['leakage_level']}")

# Network Modularity
nm = NetworkModularity()
adjacency = np.array([[0, 0.8, 0.3], [0.8, 0, 0.4], [0.3, 0.4, 0]])
result = nm.calculate_modularity(adjacency)
print(f"Modularity: {result['modularity']:.4f}")

# Transparency Score
ts = TransparencyScore()
explanations = [
    {'explanation_quality': 0.8, 'feature_importance': 0.75, 'interpretability': 0.9},
    {'explanation_quality': 0.7, 'feature_importance': 0.8, 'interpretability': 0.85}
]
result = ts.calculate_ts(explanations)
print(f"TS: {result['ts']:.4f} - {result['interpretation']['verdict']}")

# Robustness Certification Score
rcs = RobustnessCertificationScore()
original = np.array([1, 1, 0, 1, 0])
perturbed = [np.array([1, 1, 0, 1, 0]), np.array([1, 0, 0, 1, 0])]
result = rcs.calculate_rcs(original, perturbed)
print(f"RCS: {result['rcs']:.4f} - {result['interpretation']['robustness_level']}")
```

### Geographic Equity (v1.1.0)

`equimed_dss.geographic` quantifies how well the evidence base reflects the global disease burden:

- **`BurdenEvidenceMismatch` (BEMI)**: total-variation distance between the regional evidence
  distribution and the regional disease-burden distribution. Range [0, 1]; 0 means evidence
  tracks burden exactly, 1 means the two distributions are completely disjoint.
- **`GeographicConcentration` (GCC)**: sample-corrected Gini coefficient (G*) and normalized
  Shannon entropy (H_norm) for the regional distribution of included studies. G* = 0 and
  H_norm = 1 both indicate even coverage; G* = 1 and H_norm = 0 indicate single-region
  concentration. Note that Gini and entropy run in opposite directions.
- **`WHO_REGION_IHD_BURDEN`**: bundled reference constant of normalized IHD DALY shares from
  Roth GA et al., 2020 (GBD). AFRO and SEARO together carry about 36% of global IHD burden.

```python
from equimed_dss.geographic import BurdenEvidenceMismatch, GeographicConcentration, WHO_REGION_IHD_BURDEN

evidence = {"AFRO": 5, "AMRO": 40, "EURO": 30, "SEARO": 3, "WPRO": 10, "EMRO": 2}

bemi = BurdenEvidenceMismatch()
bemi_result = bemi.calculate_bemi(
    evidence_counts=evidence,
    burden_shares=WHO_REGION_IHD_BURDEN,
)
print(f"BEMI: {bemi_result['bemi']:.3f}")  # 0 = aligned, 1 = disjoint

gcc = GeographicConcentration()
gcc_result = gcc.calculate_gcc(evidence)
print(f"Gini* (G*): {gcc_result['gini_corrected']:.3f}")
print(f"H_norm: {gcc_result['entropy_normalized']:.3f}")
```

### Reporting Tables (v1.1.0)

`equimed_dss.reporting` converts metric results into tidy DataFrames and exports them:

```python
from equimed_dss.geographic import (
    BurdenEvidenceMismatch, GeographicConcentration, WHO_REGION_IHD_BURDEN,
)
from equimed_dss.reporting import geographic_table, export_table

evidence = {"AFRO": 5, "AMRO": 40, "EURO": 30, "SEARO": 3, "WPRO": 10, "EMRO": 2}
bemi_result = BurdenEvidenceMismatch().calculate_bemi(
    evidence_counts=evidence, burden_shares=WHO_REGION_IHD_BURDEN)
gcc_result = GeographicConcentration().calculate_gcc(evidence)

df = geographic_table(bemi_result, gcc_result)
print(df)                                    # show the table in the console
print(export_table(df, fmt="markdown"))      # render it as markdown to the screen

# To also save to files, pass a path (returns None, so do not wrap these in print):
export_table(df, fmt="markdown", path="results/geographic.md")
export_table(df, fmt="latex",    path="results/geographic.tex")
export_table(df, fmt="html",     path="results/geographic.html")
```

Output:

```
                       metric  value
0                        BEMI   0.48
1                  Gini* (G*)  0.613
2                      H_norm  0.742
3  concentration (1 - H_norm)  0.258
4     most_underserved_region   EMRO
5                   n_regions      6
```

---

## Visualizations

All plot helpers in `equimed_dss.utils` **return a Matplotlib figure** (they do
not call `plt.show()`), and write to `save_path` when given, so they compose
cleanly in scripts, notebooks, and report pipelines.

**Equity radar** — one normalized score per domain for an at-a-glance audit:

```python
from equimed_dss.utils import plot_equity_radar

fig = plot_equity_radar(
    {"Reliability": 0.16, "Fairness": 0.95, "Governance": 0.80,
     "Representation": 0.33, "Robustness": 0.73},
    reference=0.8,                      # optional acceptability-target ring
    save_path="equity_radar.png",
)
```

**Geographic dumbbell** — disease burden vs evidence share per region (reads the
burden-evidence mismatch, BEMI, far more clearly than a bubble plot):

```python
from equimed_dss.utils import plot_geographic_dumbbell

fig = plot_geographic_dumbbell(
    burden_shares={"AMRO": 0.114, "SEARO": 0.211, "AFRO": 0.150,
                   "EMRO": 0.230, "EURO": 0.195, "WPRO": 0.100},
    evidence_shares={"AMRO": 0.780, "SEARO": 0.0, "AFRO": 0.002,
                     "EMRO": 0.037, "EURO": 0.105, "WPRO": 0.077},
    save_path="geographic_dumbbell.png",
)
```

Other helpers: `plot_bland_altman`, `plot_control_chart`,
`plot_correlation_matrix`, `plot_her_heatmap`, `plot_metric_distribution`,
`plot_network_graph`, and the six manuscript figures `plot_figure2…7`.

---

## Statistical Analyses

EquiMed_DSS includes advanced statistical methods:

### Hierarchical Linear Modeling (HLM)

```python
from equimed_dss.statistics import HierarchicalLinearModeling

hlm = HierarchicalLinearModeling()
# Decompose outcome variance into between-group (e.g. hospital) vs within-group
# components; the ICC reports the between-group share.
```

### Mediation Analysis

```python
from equimed_dss.statistics import MediationAnalysis

mediation = MediationAnalysis(n_bootstrap=1000)
# Decompose a total effect into direct and indirect (mediated) pathways with a
# bootstrap CI for the indirect effect.
```

### Network Statistics

```python
from equimed_dss.statistics import NetworkStatistics

network = NetworkStatistics()
# Calculate centrality measures, clustering coefficients
```

### Reliability Analysis

```python
from equimed_dss.statistics import ReliabilityAnalysis

reliability = ReliabilityAnalysis()
# Cronbach's Alpha, Bland-Altman analysis
```

---

## Visualizations

Generate publication-ready figures (Figures 2-7 from manuscript):

Each `plot_figure*` function takes a structured dict of inputs (the exact keys
are documented in each function's docstring). Use `generate_figure_data()` for
ready-to-run sample inputs, then swap in your own data using the same keys:

```python
from equimed_dss.utils import (
    generate_figure_data,
    plot_figure2_reliability_dashboard,
    plot_figure3_corpus_comparison,
    plot_figure4_temporal_robustness,
    plot_figure5_ethics_governance,
    plot_figure6_metric_networks,
    plot_figure7_intersectional_heatmap,
)

figs = generate_figure_data()  # sample inputs for every figure

plot_figure2_reliability_dashboard(figs["fig2"], save_path="figures/fig2.png")
plot_figure3_corpus_comparison(figs["fig3"], save_path="figures/fig3.png")
plot_figure4_temporal_robustness(figs["fig4"], save_path="figures/fig4.png")
plot_figure5_ethics_governance(figs["fig5"], save_path="figures/fig5.png")
plot_figure6_metric_networks(figs["fig6"], save_path="figures/fig6.png")
plot_figure7_intersectional_heatmap(figs["fig7"], save_path="figures/fig7.png")
```

---

## Examples

The `examples/` directory contains comprehensive usage examples:

```bash
# Domain examples
python examples/example_domain1.py  # Reliability metrics
python examples/example_domain2.py  # Fairness & Ethics metrics
python examples/example_domain3.py  # Governance metrics
python examples/example_appendix.py # Advanced metrics

# Advanced examples
python examples/example_advanced_metrics.py  # All 9 new metrics
python examples/example_dataset.py           # Sample data generation
python examples/example_advanced_network.py  # Network analysis
```

For an end-to-end tutorial with data loading, metric calculations, statistical analyses, and visualization examples, see the [EquiMed-DSS vignette](https://github.com/johnmuteba/EquiMed_DSS/blob/master/docs/VIGNETTE.md).

---

## Project Structure

```
EquiMed_DSS/
├── equimed_dss/
│   ├── domain1/              # Reliability & Calibration (3 metrics)
│   │   ├── dfr.py            # Decision Flip Rate (DFR)
│   │   ├── ecs.py            # Embedding Consistency Score (ECS)
│   │   └── icc.py            # Inter-Rater Reliability (ICC)
│   ├── domain2/              # Fairness, Equity & Ethics (4 metrics)
│   │   ├── her.py            # Hierarchical Equity Ratio + Bias-Gini
│   │   ├── hafg.py           # Harm-Adjusted Fairness Gap
│   │   ├── eri.py            # Ethical Risk Index
│   │   └── ibs.py            # Intersectional Bias Score
│   ├── domain3/              # Governance & Transparency (3 metrics)
│   │   ├── tfd.py            # Temporal Fairness Drift
│   │   ├── ats.py            # Audit Traceability Score
│   │   └── gci.py            # Governance Compliance Index
│   ├── domain4/              # Representation & Robustness (4 metrics)
│   │   ├── spg.py            # Semantic Parity Gap (SPG)
│   │   ├── chr.py            # Clinical Hallucination Rate (CHR)
│   │   ├── ivi.py            # Instructional Vulnerability Index (IVI)
│   │   └── gri.py            # Geographic Representation Index (GRI)
│   ├── domain5/              # Technical-supplement fairness (12 metrics)
│   │   ├── calibration.py    # Intersectional Calibration Error (ICE)
│   │   ├── harm.py           # Weighted Clinical Harm-Adjusted Fairness Gap (wHAFG)
│   │   ├── text.py           # LDDI, REG, CIDR, DCI, UQG
│   │   ├── counterfactual.py # Counterfactual Parity Score (CPS), SRPI
│   │   ├── geographic_bias.py# Geographic Representation Bias Index (GRBI)
│   │   ├── system.py         # Healthcare System Stratified Fairness (HSSF)
│   │   └── shapley.py        # Intersectional Shapley Fairness Value (ISFV)
│   ├── geographic/           # Geographic equity (2 metrics + reference)
│   │   ├── burden_evidence.py# Burden-Evidence Mismatch (BEMI)
│   │   ├── concentration.py  # Geographic Concentration of Coverage (GCC)
│   │   └── reference_data.py # WHO_REGION_IHD_BURDEN reference shares
│   ├── appendix/             # Advanced Metrics (9 metrics)
│   │   └── advanced_metrics.py
│   ├── reporting/            # Tidy result tables + export
│   │   ├── tables.py         # hierarchical/mediation/network/geographic tables
│   │   └── export.py         # export_table (markdown/LaTeX/HTML)
│   ├── statistics/           # Statistical Analyses
│   │   ├── hierarchical.py   # Hierarchical Linear Modeling
│   │   ├── mediation.py      # Mediation Analysis
│   │   ├── network_stats.py  # Network Statistics
│   │   └── reliability_stats.py
│   └── utils/                # Utilities
│       ├── data_formatters.py # Data loading & conversion
│       ├── visualization.py   # Publication-ready figures
│       └── sample_data.py     # Sample data generation
├── examples/                 # Usage examples
├── tests/                    # Test suite (124 tests)
├── docs/                     # Documentation (incl. VIGNETTE.md)
└── pyproject.toml
```

---

## API Reference

### Core Classes

| Class | Module | Description |
|-------|--------|-------------|
| `DecisionFlipRate` | `domain1` | Decision instability under counterfactual inputs |
| `EmbeddingConsistencyScore` | `domain1` | Representation stability under perturbation |
| `InterRaterReliability` | `domain1` | Inter-rater reliability (ICC(2,1)) |
| `HierarchicalEquityRatio` | `domain2` | Group equity ratios + Bias-Gini |
| `HarmAdjustedFairnessGap` | `domain2` | Clinical harm gaps |
| `EthicalRiskIndex` | `domain2` | Ethical violations |
| `IntersectionalBiasScore` | `domain2` | Subgroup bias detection |
| `TemporalFairnessDrift` | `domain3` | Fairness over time |
| `AuditTraceabilityScore` | `domain3` | Audit completeness |
| `GovernanceComplianceIndex` | `domain3` | Regulatory compliance |
| `SemanticParityGap` | `domain4` | Latent demographic sensitivity (SPG) |
| `ClinicalHallucinationRate` | `domain4` | Unsupported-claim rate (CHR) |
| `InstructionalVulnerabilityIndex` | `domain4` | Susceptibility to bias-priming (IVI) |
| `GeographicRepresentationIndex` | `domain4` | Non-Western location share (GRI) |
| `IntersectionalCalibrationError` | `domain5` | Intersectional calibration gap (ICE) |
| `WeightedClinicalHarmAdjustedFairnessGap` | `domain5` | Severity-weighted harm gap (wHAFG) |
| `LexicalDiversityDisparityIndex` | `domain5` | Vocabulary-richness disparity (LDDI) |
| `RecommendationEntropyGap` | `domain5` | Recommendation-entropy gap (REG) |
| `CounterfactualParityScore` | `domain5` | Counterfactual response parity (CPS) |
| `ClinicalInformationDensityRatio` | `domain5` | Clinical-concept density ratio (CIDR) |
| `DiagnosticCompletenessIndex` | `domain5` | Guideline-differential coverage (DCI) |
| `UncertaintyQuantificationGap` | `domain5` | Hedging-density gap (UQG) |
| `GeographicRepresentationBiasIndex` | `domain5` | KL geography-vs-burden divergence (GRBI) |
| `HealthcareSystemStratifiedFairness` | `domain5` | System-stratified fairness (HSSF) |
| `IntersectionalShapleyFairnessValue` | `domain5` | Shapley disparity attribution (ISFV) |
| `SemanticRobustnessParityIndex` | `domain5` | Cross-group paraphrase robustness (SRPI) |
| `BootstrapConfidenceIntervals` | `appendix` | Uncertainty quantification |
| `StatisticalPowerAnalysis` | `appendix` | Sample size planning |
| `BiasConcentrationIndex` | `appendix` | Bias distribution |
| `MutualInformationContent` | `appendix` | Information leakage |
| `JensenShannonDivergence` | `appendix` | Distribution divergence |
| `WassersteinDistance` | `appendix` | Optimal transport |
| `NetworkModularity` | `appendix` | Community structure |
| `TransparencyScore` | `appendix` | Explanation quality |
| `RobustnessCertificationScore` | `appendix` | Perturbation stability |
| `BurdenEvidenceMismatch` | `geographic` | Evidence-burden mismatch (BEMI) |
| `GeographicConcentration` | `geographic` | Regional concentration (GCC) |
| `WHO_REGION_IHD_BURDEN` | `geographic` | IHD DALY burden reference shares |
| `hierarchical_coefficients_table` | `reporting` | HLM results as tidy DataFrame |
| `mediation_effects_table` | `reporting` | Mediation results as tidy DataFrame |
| `network_centrality_table` | `reporting` | Network centrality as tidy DataFrame |
| `geographic_table` | `reporting` | BEMI/GCC results as tidy DataFrame |
| `export_table` | `reporting` | Render DataFrame to markdown/LaTeX/HTML |

---

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](https://github.com/johnmuteba/EquiMed_DSS/blob/master/CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Clone and install in development mode
git clone https://github.com/johnmuteba/EquiMed_DSS.git
cd EquiMed_DSS
pip install -e ".[dev]"

# Run tests
pytest tests/ -v --cov=equimed_dss

# Code quality
black equimed_dss tests examples
isort equimed_dss tests examples
mypy equimed_dss
```

---

## Citation

If you use EquiMed_DSS in your research, please cite:

```bibtex
@software{muteba_equimed_dss_2025,
  title={EquiMed_DSS: A Comprehensive Library for Clinical AI Fairness Assessment},
  author={Muteba Mwamba, John},
  year={2025},
  url={https://github.com/johnmuteba/EquiMed_DSS},
  note={37 metrics for reliability, equity, governance, representation, and robustness in clinical AI}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/johnmuteba/EquiMed_DSS/blob/master/LICENSE) file for details.

---

## Acknowledgments

- Developed for advancing equity in clinical AI systems
- Built with support from the research community
- Statistical methods based on peer-reviewed literature

---

<div align="center">

**[Documentation](https://github.com/johnmuteba/EquiMed_DSS/tree/master/docs)** | **[Examples](https://github.com/johnmuteba/EquiMed_DSS/tree/master/examples)** | **[Issues](https://github.com/johnmuteba/EquiMed_DSS/issues)** | **[Discussions](https://github.com/johnmuteba/EquiMed_DSS/discussions)**

</div>
