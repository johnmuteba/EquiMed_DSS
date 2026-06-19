# Changelog

All notable changes to EquiMed-DSS will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.9.2] - 2026-06-19

### Documentation
- Fixed confidence-interval equations not rendering in the vignette and
  derivations Markdown. The `\%` in the `\mathrm{CI}_{95\%}` subscript was being
  read as a TeX comment by strict math renderers, swallowing the rest of the line
  ("Extra open brace or missing close brace"). The percent is removed from the
  math subscript (now `\mathrm{CI}_{95}`); the "95%" remains in the prose label,
  so every CI formula renders in GitHub, KaTeX, MathJax, and LaTeX.

## [1.9.1] - 2026-06-19

### Fixed (scalar-style usage of CI-carrying results)
- `MetricResult` now degrades gracefully to its point value in numeric and
  formatting contexts: `f"{result:.4f}"`, `round(result, 3)`, `float(result)`,
  and `result < 0.2` use the headline estimate. This fixes
  `TypeError: unsupported format string passed to MetricResult.__format__` and
  similar errors when a metric was used as a scalar.
- `HierarchicalEquityRatio.calculate_her` again returns a mapping containing
  **only** per-group entries, so `for g, r in result.items(): r["score"]` works.
  The across-group HER gap and its CI are carried as the result's printable
  point/CI (not as extra dict keys), so `print(result)` still shows the gap with
  its 95% CI while iteration and `result["White"]["score"]` behave as before.

### Documentation
- Vignette examples now `print(result)` so each metric shows its 95% CI, and the
  "Metric Formulas and Clinical Meaning" section gains, for every metric, the
  confidence-interval formula and a short explanation alongside the point formula.

## [1.9.0] - 2026-06-18

### Changed (all 37 metrics now return a MetricResult with a 95% CI)
- Every metric, when called, now displays its value alongside a 95% confidence
  interval (or the explicit "95% CI unavailable (needs observation-level input)"
  string for the few estimands whose inputs are aggregate-only). Previously only
  5 of 37 metrics printed a CI. Proportions use the Wilson score interval; sample
  statistics, gaps, ratios, divergences, and distances use a seeded percentile
  bootstrap (`random_state` fixed, so reproducible-table numbers do not drift).
- Aggregate-input metrics gained an optional observation-level argument that, when
  supplied, yields a real bootstrap CI: `HarmAdjustedFairnessGap` (`group1_cases`,
  `group2_cases`), `HierarchicalEquityRatio` (`group_observations`),
  `GeographicRepresentationBiasIndex` (`corpus_records`),
  `BurdenEvidenceMismatch` (`evidence_records`),
  `GeographicConcentration` (`region_records`).

### Breaking
- `HierarchicalEquityRatio.calculate_bias_gini` now returns a `MetricResult`
  (carrying `bias_gini` plus its CI) instead of a bare `float`. Access the value
  via `result["bias_gini"]`.
- `HierarchicalEquityRatio.calculate_her` now returns a `MetricResult` that adds a
  scalar `her_gap` (and CI when `group_observations` is given) alongside the
  per-group entries; printing shows the HER gap with its CI.

### Why
- Reviewer requirement (non-negotiable): whenever a metric is called, its value
  must be displayed alongside a 95% confidence interval (alpha = 0.05).

## [1.8.0] - 2026-06-18

### Changed (metrics always print their confidence interval)
- Metric results are now `MetricResult` objects (a `dict` subclass, so all existing
  key access and JSON serialisation are unchanged) whose printed form always shows
  the point estimate alongside its 95% CI, e.g.
  `DFR = 0.250 :: 95% CI [0.046; 0.699] (Wilson score)`. Printed bounds are ordered
  so lower <= upper (fixing reversed-interval display).
- Added confidence intervals to `EmbeddingConsistencyScore` (bootstrap over per-pair
  cosine distances) and `InterRaterReliability` ICC(2,1) (bootstrap over items);
  DFR, CHR, IVI already carried Wilson CIs and now print them. `MetricResult`
  exported at the top level.

### Why
- Reviewer requirement (non-negotiable): whenever a metric is called, its value must
  be displayed alongside its confidence interval.

## [1.7.0] - 2026-06-17

### Added (metrics now report uncertainty, not just a point value)
- Proportion metrics return value **and** uncertainty on every call:
  `ClinicalHallucinationRate`, `InstructionalVulnerabilityIndex`, and
  `DecisionFlipRate` now include `ci_lower`, `ci_upper`, `ci_method`, and a
  one-sided `p_value_above_threshold` (score test that the true rate exceeds a
  configurable acceptability `threshold`, default 5%); the interpretation string
  carries the CI and p-value too.
- `inference.bootstrap_metric(metric_fn, data, value_key=..., clusters=...)`:
  wrap *any* metric to obtain a percentile bootstrap CI over its observation
  sample (cluster-aware for repeated within-patient evaluations).
- `examples/example_uncertainty.py`: prints value + 95% CI + p-value for the
  native proportion metrics, a bootstrap CI (ordinary and cluster) for any
  metric, and a permutation-test p-value for a between-group fairness gap.
- 4 new tests (159 total pass).

### Why
- Reviewer requirement (non-negotiable): a metric should not be a single number;
  every metric call should also report uncertainty (CI and/or p-value).

## [1.6.0] - 2026-06-17

### Added (statistical inference)
- New `equimed_dss.inference` module providing uncertainty quantification for any
  metric: `wilson_ci` (binomial proportions), `proportion_ci` (Wilson CI plus a
  score test against a pre-specified acceptability threshold), `bootstrap_ci`
  (percentile bootstrap with optional **cluster/visit resampling** so repeated,
  non-independent evaluations of the same patient do not inflate precision), and
  `permutation_test` (group-difference p-values for fairness gaps). All return a
  single `InferenceResult` schema (estimate, CI, SE, method, n, n_clusters,
  p_value, null_value) with `.to_dict()` and `__str__`.
- 15 unit tests (`tests/test_inference.py`): known Wilson values, boundary
  proportions (k=0, k=n), cluster-vs-iid interval width, seeded reproducibility,
  and the add-one permutation p-value floor.

### Why
- Reviewer feedback: metrics were reported as single point estimates
  ("value-at-risk"-style numbers). Pairing each metric with a confidence interval
  and, where a null/threshold exists, a p-value, makes the library inferential
  rather than purely descriptive, and lets findings be reported with explicit
  uncertainty.

## [1.5.4] - 2026-06-16

### Added (documentation / examples)
- `examples/example_geographic.py`: now includes runnable `plot_geographic_dumbbell`
  and `plot_equity_radar` demos (using the manuscript's verified WHO-region evidence
  shares), so the v1.5.2 figures have a worked example. Headless-safe (`Agg` backend);
  writes `example_geographic_dumbbell.png` and `example_equity_radar.png`.

### Changed
- Synced `docs/Metric_Math_Derivations.tex` version stamp to the package version.

## [1.5.3] - 2026-06-15

### Changed (documentation)
- README: added a **Visualizations** section with runnable `plot_equity_radar`
  and `plot_geographic_dumbbell` examples and a note that all plot helpers return
  a Matplotlib figure (so the PyPI landing page documents the v1.5.2 plots).

## [1.5.2] - 2026-06-15

### Added
- `utils.plot_equity_radar(domain_scores, reference=0.8, ...)`: radar/spider
  chart of one normalized score per domain for an at-a-glance audit summary,
  with an optional acceptability-target ring.
- `utils.plot_geographic_dumbbell(burden_shares, evidence_shares, ...)`: dumbbell
  (Cleveland) chart of disease burden vs corpus-evidence share per region, far
  clearer than a bubble plot for reading the burden-evidence mismatch (BEMI).
- Both return the Matplotlib figure (no `plt.show()`), honour `save_path`,
  validate inputs, and normalize raw counts internally. 4 new tests (140 total).

## [1.5.1] - 2026-06-13

### Fixed (documentation / packaging)
- README: added the missing **Domain 4** (SPG, CHR, IVI, GRI) and **Domain 5**
  (ICE, wHAFG, LDDI, REG, CPS, CIDR, DCI, UQG, GRBI, HSSF, ISFV, SRPI) sections
  with metric tables and runnable examples; the table-of-contents links to them
  now resolve (they were plain text pointing nowhere on the PyPI/GitHub page).
- `pyproject.toml`: package summary corrected from "19 novel metrics" to
  "37 metrics across five domains" (this is the one-line description shown on PyPI).

## [1.5.0] - 2026-06-13

Pre-release correctness audit of all 37 metrics. Formulas were verified against
their documentation; the fixes below change the behaviour of a few metrics,
hence the minor version bump.

### Added
- `tests/test_regression_bugfixes.py`: 12 regression tests pinning the four
  corrected behaviours (Wilson CI, JSD consistency, HAFG normalization, sample-SD
  / no input mutation). Suite total: 136 tests.
- `examples/regression_bugfix_report.py` + `docs/REGRESSION_BUGFIX_REPORT.md`:
  a formatted report with a bug-fix verification table and publication-style
  mixed-effects regression coefficient and mediation tables.

### Fixed (correctness)
- **DecisionFlipRate (DFR):** the confidence interval was a percentile of the
  0/1 flip-indicator vector (effectively always [0,1]); it is now a proper
  **Wilson 95% interval** for the flip proportion. Adds `n_flipped`, `n_samples`.
- **JensenShannonDivergence (JSD):** the two implementations disagreed
  (`advanced_metrics` returned the distance, `info_theory` the divergence). Both
  now return the **JS divergence in base 2** (range [0,1]); `advanced_metrics`
  also exposes `jsd_distance`.
- **HarmAdjustedFairnessGap (HAFG):** `hafg` is now **normalized** to [0,1]
  (`|H1-H2|/max(H1,H2)`) as documented, with a principled verdict; the raw gap
  is returned as `absolute_harm_gap`.
- **IntersectionalBiasScore (IBS):** `interaction_analysis` no longer mutates the
  caller's DataFrame.
- **Bland-Altman (ICC) and TFD control chart:** now use the sample SD (ddof=1).

### Changed (documentation)
- **MutualInformationContent (MIC):** clarified that it computes mutual
  information, **not** the Reshef Maximal Information Coefficient.
- **NetworkModularity:** docstring corrected (greedy Clauset-Newman-Moore, not
  Louvain).
- Rewrote `docs/METRICS_GUIDE.md` to match the actual 37-metric API (correct
  Domain-1 classes; added Domains 4-5); fixed `docs/API_REFERENCE.md` Domain-1
  classes and the HAFG/ATS/GCI signatures; updated the package docstring to 37
  metrics; removed stale embedded findings ("55.8%", "72.1%") from docstrings,
  README examples, and a figure title.

### Changed (visualization)
- All `utils.visualization` plot functions now **return the Matplotlib figure**
  and no longer call `plt.show()` (library-friendly; still honours `save_path`).

## [1.4.2] - 2026-06-10

### Changed (documentation)
- README: API Reference "Core Classes" table now lists all Domain 4 and Domain 5
  classes and uses the correct Domain 1 class names (DecisionFlipRate,
  EmbeddingConsistencyScore, InterRaterReliability; the old names were stale).
- README: Project Structure now shows domain4/, domain5/, geographic/, and
  reporting/ packages; test count updated to 124.

## [1.4.1] - 2026-06-10

### Changed (documentation)
- README: corrected the metric count (37 across five domains) on the PyPI/GitHub
  landing; the Reporting Tables example is now self-contained and prints the table.
- Vignette "Metric Formulas And Clinical Meaning": reordered to Domain 1-5 then
  Statistics then Appendix; added a clinical interpretation and a runnable,
  printed example for every metric; fixed LaTeX rendering (literal asterisks in
  math now use \ast, so GCC and DCI render correctly).

## [1.4.0] - 2026-06-10

### Added
- `equimed_dss.domain5`: twelve technical-supplement fairness metrics.
  IntersectionalCalibrationError (ICE), WeightedClinicalHarmAdjustedFairnessGap
  (wHAFG), LexicalDiversityDisparityIndex (LDDI), RecommendationEntropyGap (REG),
  CounterfactualParityScore (CPS), ClinicalInformationDensityRatio (CIDR),
  DiagnosticCompletenessIndex (DCI), UncertaintyQuantificationGap (UQG),
  GeographicRepresentationBiasIndex (GRBI), HealthcareSystemStratifiedFairness
  (HSSF), IntersectionalShapleyFairnessValue (ISFV), and
  SemanticRobustnessParityIndex (SRPI).
- These complement, and do not duplicate, existing metrics: wHAFG generalizes
  HAFG (per-sample severity weighting), GRBI complements BEMI (KL vs
  total-variation), ISFV complements IBS (Shapley vs ANOVA), CPS/SRPI complement
  DFR/SPG/ECS/RCS.

## [1.3.0] - 2026-06-08

### Added
- `equimed_dss.domain4`: four representation/robustness metrics.
  - `SemanticParityGap` (SPG): Euclidean centroid and cosine distance between the
    embedding clusters of clinical prompts differing only by a protected attribute.
  - `ClinicalHallucinationRate` (CHR): unsupported-claim rate from per-claim
    entailment support scores, with a severity-weighted variant.
  - `InstructionalVulnerabilityIndex` (IVI): decision-flip rate (and directional
    effect) between neutral and biased/leading instructions.
  - `GeographicRepresentationIndex` (GRI): set-based non-Western location share,
    plus `calculate_geographic_bias` (correlation of GRI with error rate).

## [1.2.3] - 2026-06-08

### Fixed
- The `plot_figure*` and `plot_*` functions now display the figure inline (via
  `plt.show()`) in addition to saving it when `save_path` is given. Previously a
  saved figure was closed silently, so in a notebook nothing appeared even though
  the PNG was written to disk. Figures now both render and save.

## [1.2.2] - 2026-06-08

### Added
- `generate_figure_data()` in `equimed_dss.utils`: returns ready-to-use sample
  inputs for every `plot_figure*` function (keys `fig2` through `fig7`), so all
  six manuscript figures render with one call. Swap in your own data using the
  same keys (documented in each plot function's docstring).

### Changed
- `export_table` now follows the pandas `to_csv` convention: when `path` is
  given it writes the file and returns `None` (so a notebook cell no longer
  echoes a large raw HTML/markdown string); when `path` is `None` it returns the
  rendered string. For inline viewing use `print(export_table(df, fmt="markdown"))`.
- README and vignette figure examples now use `generate_figure_data()` and run
  as written (the previous snippet referenced undefined variables).

## [1.2.1] - 2026-06-08

### Fixed
- `export_table(path=...)` now creates the parent directory if it does not
  exist, so writing to e.g. `results/geographic.md` no longer raises
  FileNotFoundError.
- The `plot_figure*` functions now create the `save_path` parent directory
  before saving.
- Removed a broken Build Status badge (no CI workflow) that rendered as "?".

### Changed
- Vignette: metric formulas rewritten in LaTeX (rendered math) for a clean,
  academic presentation; removed an internal note not addressed to readers.

## [1.2.0] - 2026-06-08

### Changed (API finalized to match the documentation)
- `BurdenEvidenceMismatch.calculate_bemi(evidence_counts, burden_shares)` now
  returns `bemi`, `evidence_shares`, `burden_shares`, `per_region`,
  `most_underserved_region`, and a string `interpretation`.
- `GeographicConcentration.calculate_gcc(region_counts)` now returns
  `gini_corrected`, `entropy_normalized`, `concentration`, `per_region`.
- `geographic_table(bemi_result, gcc_result)` now combines both results into one
  summary DataFrame. All table functions accept `decimals=`.
- `mediation_effects_table` adds an `outside_bounds` column (replaces the prior
  DataFrame-attribute flag).
- `HierarchicalLinearModeling.fit_model` now also returns `coefficients`
  (per fixed effect: estimate, std_err, t, p_value, ci_lower, ci_upper), and
  `hierarchical_coefficients_table` renders them.

### Added
- Vignette sections: geographic metrics with formulas and the derivation of the
  "about 36%" AFRO+SEARO IHD-burden figure; a full "Metric Formulas and Clinical
  Meaning" reference; an expanded explanation of `prediction` vs `actual`.

### Notes
- Earlier 1.1.0/1.1.1 used different geographic/reporting argument and key names
  that did not match the published docs. 1.2.0 makes the documented API the real
  one. Reinstall from the index you use and restart your kernel.

## [1.1.1] - 2026-06-07

### Fixed
- `HierarchicalLinearModeling.fit_model` now returns finite AIC and BIC.
  Previously they were NaN because statsmodels withholds information criteria
  under REML estimation (the default). The full model is now also fit with
  maximum likelihood (reml=False) solely to obtain valid AIC/BIC, while the
  REML fit is retained for the variance components and ICC. A defensive
  fallback computes AIC/BIC from the log-likelihood if statsmodels still
  reports NaN.

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
