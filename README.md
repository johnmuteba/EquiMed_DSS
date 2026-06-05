# EquiMed_DSS

<div align="center">

<h3>A Comprehensive Python Library for Clinical AI Fairness Assessment</h3>

<p>Evaluate reliability, equity, governance, and intersectionality in clinical AI systems using <strong>19 novel metrics</strong></p>

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Build Status](https://github.com/johnmuteba/EquiMed_DSS/workflows/CI/badge.svg)](https://github.com/johnmuteba/EquiMed_DSS/actions)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

</div>

---

## Overview

**EquiMed_DSS** (Equitable Medical Decision Support System) provides a systematic framework for evaluating clinical AI systems across multiple dimensions of fairness, reliability, and governance. The library implements **19 novel metrics** specifically designed for healthcare applications where equity and safety are paramount.

### Key Features

| Feature | Description |
|---------|-------------|
| **19 Novel Metrics** | Comprehensive coverage across 3 domains + advanced appendix |
| **Clinical AI Focus** | Designed specifically for healthcare applications |
| **Statistical Analyses** | HLM, Mediation Analysis, Network Statistics |
| **Publication-Ready Visualizations** | 6 manuscript-quality figure generators |
| **Multi-Format Data Support** | MySQL, CSV, TSV, JSON with automatic standardization |
| **Intersectional Analysis** | Detect bias across demographic combinations |

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Data Format](#data-format)
- [Metrics Overview](#metrics-overview)
  - [Domain 1: Reliability & Calibration](#domain-1-reliability--calibration)
  - [Domain 2: Fairness, Equity & Ethics](#domain-2-fairness-equity--ethics)
  - [Domain 3: Governance & Transparency](#domain-3-governance--transparency)
  - [Appendix: Advanced Metrics](#appendix-advanced-metrics)
- [Statistical Analyses](#statistical-analyses)
- [Visualizations](#visualizations)
- [Examples](#examples)
- [Vignette](docs/VIGNETTE.md)
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
| Dynamic Fairness Ratio | DFR | [0, ∞) | ≥ 0.9 | Performance consistency across conditions |
| Expected Calibration Score | ECS | [0, 1] | < 0.05 | Prediction calibration quality |
| Intraclass Correlation | ICC | [0, 1] | > 0.75 | Inter-rater reliability |

```python
from equimed_dss.domain1 import DynamicFairnessRatio, ExpectedCalibrationScore, IntraclassCorrelationCoefficient

# Dynamic Fairness Ratio
dfr = DynamicFairnessRatio()
result = dfr.calculate_dfr(baseline_metric=0.85, dynamic_metric=0.80)
print(f"DFR: {result['dfr']:.3f} - {result['interpretation']}")

# Expected Calibration Score
ecs = ExpectedCalibrationScore()
result = ecs.calculate_ecs([0.9, 0.8, 0.7, 0.6], [1, 1, 0, 1], n_bins=4)
print(f"ECS: {result['ecs']:.3f}")

# Intraclass Correlation Coefficient
icc = IntraclassCorrelationCoefficient()
result = icc.calculate_icc([[3, 4, 3], [5, 5, 4], [2, 3, 2]])
print(f"ICC: {result['icc']:.3f}")
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

# Audit Traceability Score
ats = AuditTraceabilityScore()
result = ats.calculate_ats([
    {'timestamp': True, 'user': True, 'action': True, 'details': True},
    {'timestamp': True, 'user': True, 'action': False, 'details': True}
])
print(f"ATS: {result['ats']:.3f}")
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

---

## Statistical Analyses

EquiMed_DSS includes advanced statistical methods:

### Hierarchical Linear Modeling (HLM)

```python
from equimed_dss.statistics import HierarchicalLinearModeling

hlm = HierarchicalLinearModeling()
# Analyze variance decomposition across hospital levels
# Key finding: 55.8% of variance at hospital level
```

### Mediation Analysis

```python
from equimed_dss.statistics import MediationAnalysis

mediation = MediationAnalysis(n_bootstrap=1000)
# Analyze bias pathways
# Key finding: 72.1% of bias through indirect pathways
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

```python
from equimed_dss.utils import (
    plot_figure2_reliability_dashboard,
    plot_figure3_corpus_comparison,
    plot_figure4_temporal_robustness,
    plot_figure5_ethics_governance,
    plot_figure6_metric_networks,
    plot_figure7_intersectional_heatmap
)

# Generate all manuscript figures
plot_figure2_reliability_dashboard(data, save_path='figures/fig2.png')
plot_figure3_corpus_comparison(corpus_data, save_path='figures/fig3.png')
plot_figure4_temporal_robustness(temporal_data, save_path='figures/fig4.png')
plot_figure5_ethics_governance(ethics_data, save_path='figures/fig5.png')
plot_figure6_metric_networks(network_data, save_path='figures/fig6.png')
plot_figure7_intersectional_heatmap(intersectional_data, save_path='figures/fig7.png')
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

For an end-to-end tutorial with data loading, metric calculations, statistical analyses, and visualization examples, see the [EquiMed-DSS vignette](docs/VIGNETTE.md).

---

## Project Structure

```
EquiMed_DSS/
├── equimed_dss/
│   ├── domain1/              # Reliability & Calibration (3 metrics)
│   │   ├── dfr.py            # Dynamic Fairness Ratio
│   │   ├── ecs.py            # Expected Calibration Score
│   │   └── icc.py            # Intraclass Correlation Coefficient
│   ├── domain2/              # Fairness, Equity & Ethics (4 metrics)
│   │   ├── her.py            # Hierarchical Equity Ratio + Bias-Gini
│   │   ├── hafg.py           # Harm-Adjusted Fairness Gap
│   │   ├── eri.py            # Ethical Risk Index
│   │   └── ibs.py            # Intersectional Bias Score
│   ├── domain3/              # Governance & Transparency (3 metrics)
│   │   ├── tfd.py            # Temporal Fairness Drift
│   │   ├── ats.py            # Audit Traceability Score
│   │   └── gci.py            # Governance Compliance Index
│   ├── appendix/             # Advanced Metrics (9 metrics)
│   │   └── advanced_metrics.py
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
├── tests/                    # Test suite (68 tests)
├── docs/                     # Documentation
└── setup.py
```

---

## API Reference

### Core Classes

| Class | Module | Description |
|-------|--------|-------------|
| `DynamicFairnessRatio` | `domain1` | Performance consistency |
| `ExpectedCalibrationScore` | `domain1` | Calibration quality |
| `IntraclassCorrelationCoefficient` | `domain1` | Inter-rater reliability |
| `HierarchicalEquityRatio` | `domain2` | Group equity ratios |
| `HarmAdjustedFairnessGap` | `domain2` | Clinical harm gaps |
| `EthicalRiskIndex` | `domain2` | Ethical violations |
| `IntersectionalBiasScore` | `domain2` | Subgroup bias detection |
| `TemporalFairnessDrift` | `domain3` | Fairness over time |
| `AuditTraceabilityScore` | `domain3` | Audit completeness |
| `GovernanceComplianceIndex` | `domain3` | Regulatory compliance |
| `BootstrapConfidenceIntervals` | `appendix` | Uncertainty quantification |
| `StatisticalPowerAnalysis` | `appendix` | Sample size planning |
| `BiasConcentrationIndex` | `appendix` | Bias distribution |
| `MutualInformationContent` | `appendix` | Information leakage |
| `JensenShannonDivergence` | `appendix` | Distribution divergence |
| `WassersteinDistance` | `appendix` | Optimal transport |
| `NetworkModularity` | `appendix` | Community structure |
| `TransparencyScore` | `appendix` | Explanation quality |
| `RobustnessCertificationScore` | `appendix` | Perturbation stability |

---

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

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
  note={19 novel metrics for reliability, equity, and governance in clinical AI}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- Developed for advancing equity in clinical AI systems
- Built with support from the research community
- Statistical methods based on peer-reviewed literature

---

<div align="center">

**[Documentation](docs/)** | **[Examples](examples/)** | **[Issues](https://github.com/johnmuteba/EquiMed_DSS/issues)** | **[Discussions](https://github.com/johnmuteba/EquiMed_DSS/discussions)**

</div>
