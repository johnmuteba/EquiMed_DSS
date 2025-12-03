# EquiMed_DSS

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Build Status](https://github.com/johnmuteba/EquiMed_DSS/workflows/CI/badge.svg)](https://github.com/johnmuteba/EquiMed_DSS/actions)

**EquiMed_DSS** is a comprehensive Python library for assessing reliability, equity, governance, and intersectionality in clinical AI systems using **19 novel metrics** across multiple domains.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Metrics Documentation](#metrics-documentation)
  - [Domain 1: Reliability and Calibration](#domain-1-reliability-and-calibration)
  - [Domain 2: Fairness, Equity, and Ethics](#domain-2-fairness-equity-and-ethics)
  - [Domain 3: Governance and Transparency](#domain-3-governance-and-transparency)
  - [Appendix: Advanced Metrics](#appendix-advanced-metrics)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)

## Overview

EquiMed_DSS provides a systematic framework for evaluating clinical AI systems across multiple dimensions of fairness, reliability, and governance. The library implements 19 novel metrics designed specifically for healthcare applications where equity and safety are paramount.

## Features

- **19 Novel Metrics** across 3 main domains plus advanced appendix metrics
- **Clinical AI Focus**: Designed specifically for healthcare applications
- **Comprehensive Coverage**: Reliability, fairness, ethics, governance, and transparency
- **Easy Integration**: Simple Python API with clear documentation
- **Visualization Tools**: Built-in plotting functions for metric analysis
- **Synthetic Data Generation**: Test metrics without real patient data

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Install from Source

```bash
# Clone the repository
git clone https://github.com/johnmuteba/EquiMed_DSS.git
cd EquiMed_DSS

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Install via pip (Coming Soon)

```bash
pip install equimed-dss
```

### Dependencies

The library requires the following packages:
- numpy
- pandas
- scipy
- scikit-learn
- matplotlib
- seaborn
- networkx
- statsmodels

## Quick Start

```python
import numpy as np
from equimed_dss.domain2 import HierarchicalEquityRatio

# Calculate Hierarchical Equity Ratio
her_metric = HierarchicalEquityRatio()
group_scores = {
    'White': 0.85,
    'Black': 0.78,
    'Hispanic': 0.80,
    'Asian': 0.87
}

her_scores = her_metric.calculate_her(group_scores)
print(f"HER Scores: {her_scores}")

# Calculate Bias-Gini Index
gini = her_metric.calculate_bias_gini(list(group_scores.values()))
print(f"Bias-Gini Index: {gini:.4f}")
```

## Metrics Documentation

### Domain 1: Reliability and Calibration

#### 1. Dynamic Fairness Ratio (DFR)
Measures performance consistency across dynamic conditions.

**Usage:**
```python
from equimed_dss.domain1 import DynamicFairnessRatio

dfr_metric = DynamicFairnessRatio()
result = dfr_metric.calculate_dfr(baseline_metric=0.85, dynamic_metric=0.80)
print(f"DFR: {result['dfr']:.3f} - {result['interpretation']}")
```

**Interpretation:**
- **Range**: [0, ∞)
- **Ideal**: ≥ 0.9 (minimal degradation)
- **Warning**: < 0.9 (significant performance drop)

#### 2. Expected Calibration Score (ECS)
Measures calibration quality of model predictions.

**Usage:**
```python
from equimed_dss.domain1 import ExpectedCalibrationScore

ecs_metric = ExpectedCalibrationScore()
predictions = [0.9, 0.8, 0.7, 0.6]
actuals = [1, 1, 0, 1]
result = ecs_metric.calculate_ecs(predictions, actuals, n_bins=4)
print(f"ECS: {result['ecs']:.3f}")
```

**Interpretation:**
- **Range**: [0, 1]
- **Ideal**: < 0.05 (well-calibrated)
- **Warning**: > 0.1 (miscalibrated)

#### 3. Intraclass Correlation Coefficient (ICC)
Assesses consistency of measurements across raters or conditions.

**Usage:**
```python
from equimed_dss.domain1 import IntraclassCorrelationCoefficient

icc_metric = IntraclassCorrelationCoefficient()
ratings = [[3, 4, 3], [5, 5, 4], [2, 3, 2]]
result = icc_metric.calculate_icc(ratings)
print(f"ICC: {result['icc']:.3f}")
```

**Interpretation:**
- **Range**: [0, 1]
- **Ideal**: > 0.75 (excellent reliability)
- **Good**: 0.6 - 0.75
- **Moderate**: 0.4 - 0.6
- **Poor**: < 0.4

---

### Domain 2: Fairness, Equity, and Ethics

#### 4. Hierarchical Equity Ratio (HER)
Calculates equity ratios across demographic groups relative to a reference group.

**Usage:**
```python
from equimed_dss.domain2 import HierarchicalEquityRatio

her_metric = HierarchicalEquityRatio()
scores = {
    'White': 0.85,
    'Black': 0.68,
    'Hispanic': 0.75
}
her_scores = her_metric.calculate_her(scores)
gini = her_metric.calculate_bias_gini(list(scores.values()))
```

**Interpretation:**
- **Range**: [0, ∞)
- **Ideal**: 0.8 - 1.25 (equitable, based on 4/5ths rule)
- **Disparity**: Outside this range

**Bias-Gini Index:**
- **Range**: [0, 1]
- **Ideal**: < 0.2 (low dispersion)

#### 5. Harm-Adjusted Fairness Gap (HAFG)
Quantifies fairness gaps weighted by clinical harm severity.

**Usage:**
```python
from equimed_dss.domain2 import HarmAdjustedFairnessGap

hafg_metric = HarmAdjustedFairnessGap()
group1_errors = {'fn': 5, 'fp': 10}
group2_errors = {'fn': 2, 'fp': 5}
result = hafg_metric.calculate_hafg(group1_errors, group2_errors)
print(f"HAFG: {result['hafg']:.3f}")
```

**Interpretation:**
- **Range**: [0, ∞)
- **Ideal**: < 0.1 (minimal harm gap)
- **Warning**: > 0.2 (significant harm disparity)

#### 6. Ethical Risk Index (ERI)
Aggregates ethical violations weighted by severity.

**Usage:**
```python
from equimed_dss.domain2 import EthicalRiskIndex

eri_metric = EthicalRiskIndex()
violations = [
    {'severity': 2.5},
    {'severity': 1.0},
    {'severity': 5.0}
]
result = eri_metric.calculate_eri(violations, n_total_outputs=100)
print(f"ERI: {result['eri']:.3f}")
```

**Interpretation:**
- **Range**: [0, ∞)
- **Ideal**: < 0.05 (low ethical risk)
- **Warning**: > 0.1 (high ethical risk)

#### 7. Intersectional Bias Score (IBS)
Detects outlier subgroups using distance-based analysis.

**Usage:**
```python
from equimed_dss.domain2 import IntersectionalBiasScore

ibs_metric = IntersectionalBiasScore()
vectors = {
    'GroupA': np.array([0.8, 0.7, 0.9]),
    'GroupB': np.array([0.75, 0.65, 0.85]),
    'GroupC': np.array([0.5, 0.4, 0.6])
}
result = ibs_metric.calculate_subgroup_similarity(vectors)
print(f"Outlier: {result['outlier_subgroup']}")
```

**Interpretation:**
- Identifies subgroups with significantly different outcomes
- Higher distance = greater intersectional bias

---

### Domain 3: Governance and Transparency

#### 8. Temporal Fairness Drift (TFD)
Tracks fairness metric degradation over time using statistical process control.

**Usage:**
```python
from equimed_dss.domain3 import TemporalFairnessDrift

tfd_metric = TemporalFairnessDrift()
time_series = [0.85, 0.84, 0.86, 0.83, 0.75, 0.84]
result = tfd_metric.calculate_drift(time_series)
print(f"Drift Detected: {result['drift_detected']}")
```

**Interpretation:**
- Uses 3-sigma control limits
- **Stable**: No out-of-control points
- **Unstable**: Points outside control limits indicate drift

#### 9. Audit Traceability Score (ATS)
Measures completeness and quality of audit trails.

**Usage:**
```python
from equimed_dss.domain3 import AuditTraceabilityScore

ats_metric = AuditTraceabilityScore()
logs = [
    {'timestamp': True, 'user': True, 'action': True, 'details': True},
    {'timestamp': True, 'user': True, 'action': False, 'details': True},
]
result = ats_metric.calculate_ats(logs)
print(f"ATS: {result['ats']:.3f}")
```

**Interpretation:**
- **Range**: [0, 1]
- **Ideal**: > 0.9 (excellent traceability)
- **Warning**: < 0.7 (poor audit trail)

#### 10. Governance Compliance Index (GCI)
Assesses adherence to governance frameworks and regulations.

**Usage:**
```python
from equimed_dss.domain3 import GovernanceComplianceIndex

gci_metric = GovernanceComplianceIndex()
requirements = [True, True, False, True, True]
result = gci_metric.calculate_gci(requirements)
print(f"GCI: {result['gci']:.3f}")
```

**Interpretation:**
- **Range**: [0, 1]
- **Ideal**: 1.0 (full compliance)
- **Warning**: < 0.8 (compliance issues)

---

### Appendix: Advanced Metrics

#### Advanced Reliability
- **Bias Concentration Index (BCI)**: Measures concentration of bias in specific subgroups
- **Subgroup Performance Analyzer (SPA)**: Analyzes performance variations across subgroups
- **Bland-Altman Bias**: Assesses agreement between measurement methods

#### Advanced Information-Theoretic
- **Maximal Information Coefficient (MIC)**: Detects non-linear relationships
- **Jensen-Shannon Divergence (JSD)**: Measures distribution similarity
- **Wasserstein Distance (WD)**: Optimal transport distance between distributions

#### Advanced Network and Governance
- **Network Modularity (NM)**: Analyzes community structure in bias networks
- **Topic Sensitivity (TS)**: Measures performance variation across topics
- **Risk Cascade Score (RCS)**: Quantifies propagation of risks through system

## Examples

The `examples/` directory contains comprehensive examples for each domain:

```bash
# Run Domain 1 examples (Reliability)
python examples/example_domain1.py

# Run Domain 2 examples (Fairness & Ethics)
python examples/example_domain2.py

# Run Domain 3 examples (Governance)
python examples/example_domain3.py

# Run Appendix examples (Advanced Metrics)
python examples/example_appendix.py
```

## Project Structure

```
EquiMed_DSS/
├── equimed_dss/
│   ├── domain1/          # Reliability & Calibration metrics
│   ├── domain2/          # Fairness, Equity & Ethics metrics
│   ├── domain3/          # Governance & Transparency metrics
│   ├── appendix/         # Advanced metrics
│   └── utils/            # Utilities and visualizations
├── examples/             # Usage examples
├── tests/                # Unit tests
├── docs/                 # Documentation
├── setup.py              # Package setup
└── README.md             # This file
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Clone the repository
git clone https://github.com/johnmuteba/EquiMed_DSS.git
cd EquiMed_DSS

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use EquiMed_DSS in your research, please cite:

```bibtex
@software{equimed_dss2024,
  title={EquiMed_DSS: A Comprehensive Library for Clinical AI Fairness Assessment},
  author={EquiMed Team},
  year={2024},
  url={https://github.com/johnmuteba/EquiMed_DSS}
}
```

## Acknowledgments

Built with Gemini3 and developed for advancing equity in clinical AI systems.

## Contact

For questions, issues, or collaborations, please:
- Open an issue on [GitHub](https://github.com/johnmuteba/EquiMed_DSS/issues)
- Contact the maintainers

---

**Note**: This library is under active development. Features and APIs may change. Feedback and contributions are welcome!
