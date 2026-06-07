# Changelog

All notable changes to EquiMed-DSS will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-06-07

### Added
- `equimed_dss.geographic`: Burden-Evidence Mismatch Index (BEMI, the
  total-variation distance between evidence and disease-burden distributions)
  and Geographic Concentration of Coverage (GCC, sample-corrected Gini plus
  normalized Shannon entropy). Bundled `WHO_REGION_IHD_BURDEN` reference.
- `equimed_dss.reporting`: tidy-DataFrame tables for hierarchical, mediation,
  and network results, plus `export_table` (markdown, LaTeX, HTML).
- Examples: `example_geographic.py`, `example_statistics_tables.py`.

### Notes
- `WHO_REGION_IHD_BURDEN` uses Roth GA et al., 2020 GBD IHD DALY shares; AFRO
  and SEARO together carry about 36% of global IHD burden.
- `proportion_mediated` in the mediation table is reported unclamped and
  flagged when it falls outside [0, 1] (competitive or unstable mediation).

---

## [1.0.0] - 2025-12-05

### Added

#### Core Metrics (19 Total)

**Domain 1: Reliability & Calibration**
- `DynamicFairnessRatio` (DFR) - Performance consistency across conditions
- `ExpectedCalibrationScore` (ECS) - Prediction calibration quality
- `IntraclassCorrelationCoefficient` (ICC) - Inter-rater reliability

**Domain 2: Fairness, Equity & Ethics**
- `HierarchicalEquityRatio` (HER) - Group equity ratios with Bias-Gini Index
- `HarmAdjustedFairnessGap` (HAFG) - Clinical harm-weighted disparity
- `EthicalRiskIndex` (ERI) - Aggregated ethical violations
- `IntersectionalBiasScore` (IBS) - Subgroup outlier detection

**Domain 3: Governance & Transparency**
- `TemporalFairnessDrift` (TFD) - Fairness degradation over time
- `AuditTraceabilityScore` (ATS) - Audit trail completeness
- `GovernanceComplianceIndex` (GCI) - Regulatory compliance

**Appendix: Advanced Metrics**
- `BootstrapConfidenceIntervals` (BCI) - Robust uncertainty estimation
- `StatisticalPowerAnalysis` (SPA) - Sample size adequacy
- `BiasConcentrationIndex` - Bias distribution across groups
- `MutualInformationContent` (MIC) - Demographic information leakage
- `JensenShannonDivergence` (JSD) - Distributional similarity
- `WassersteinDistance` (WD) - Optimal transport distance
- `NetworkModularity` (NM) - Metric clustering structure
- `TransparencyScore` (TS) - Explanation quality
- `RobustnessCertificationScore` (RCS) - Perturbation stability

#### Statistical Analyses
- `HierarchicalLinearModeling` - Mixed effects models with ICC calculation
- `MediationAnalysis` - Causal mediation with bootstrap confidence intervals
- `NetworkStatistics` - Centrality measures and clustering coefficients
- `ReliabilityAnalysis` - Cronbach's Alpha and Bland-Altman analysis

#### Visualizations (Figures 2-7)
- `plot_figure2_reliability_dashboard` - 4-panel reliability dashboard
- `plot_figure3_corpus_comparison` - Corpus comparison analysis
- `plot_figure4_temporal_robustness` - Temporal robustness analysis
- `plot_figure5_ethics_governance` - Ethics and governance dashboard
- `plot_figure6_metric_networks` - Network visualization
- `plot_figure7_intersectional_heatmap` - Intersectional analysis heatmap

#### Data Utilities
- `SampleDataGenerator` - 12 methods for generating synthetic test data
- `CorpusLoader` - Load data from MySQL, CSV, TSV, JSON
- `DemographicProcessor` - Intersectional analysis and demographic processing
- `convert_to_standard_format` - Data format standardization

#### Documentation
- Comprehensive README.md with usage examples
- Data format schema documentation
- API reference for all 19 metrics
- Example scripts for each domain

### Technical
- Python 3.8-3.13 support
- Full CI/CD pipeline with GitHub Actions
- 68 passing tests with 90%+ coverage
- Black, isort, flake8, mypy, bandit code quality checks
- Security scanning with bandit and safety

---

## [0.1.0] - 2024-12-01

### Added
- Initial release with 10 basic metrics
- Basic project structure
- Preliminary documentation

---

## Future Releases

### Planned for v1.1.0
- Interactive Plotly visualizations
- Propensity Score Matching (PSM)
- Instrumental Variable (IV) regression
- Additional demographic categories

### Planned for v1.2.0
- Command-line interface (CLI)
- Jupyter notebook integration
- Automated report generation
- PDF/HTML export

---

[1.0.0]: https://github.com/johnmuteba/EquiMed_DSS/releases/tag/v1.0.0
[0.1.0]: https://github.com/johnmuteba/EquiMed_DSS/releases/tag/v0.1.0
