# Changelog

All notable changes to EquiMed-DSS will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.10.0] - 2026-10-08

A metric-by-metric review of the library. Values of BEMI, GCC, the GRI point
estimate, ICE, wHAFG and DFR are unchanged for valid inputs, so results computed
with 1.9.5 for those metrics stand. The corrections below change other results;
the reasons are given so that earlier analyses can be checked.

### Fixed (values change)
- `appendix.AdvancedReliabilityMetrics.calculate_power_analysis`: the per-group
  sample size lacked the factor 2 of the two-sample formula
  n = 2 (z_{1-alpha/2} + z_{power})^2 / d^2, so it was half what is needed (for
  d = 0.5, alpha = 0.05, power 0.8: 32 before, 63 now; the t-test solution in
  `StatisticalPowerAnalysis` gives 64).
- `appendix.AdvancedReliabilityMetrics.calculate_bias_concentration`: the
  `population_share` argument was ignored and the sample (1/(n-1)) covariance was
  used. It is now the Kakwani, Wagstaff and van Doorslaer concentration index,
  weighted by `population_share` for grouped data; with equal shares the value is
  (n-1)/n times the old one.
- `domain3.AuditTraceabilityScore`: the interval was labelled "Wilson score" but
  was the Agresti-Coull interval (same centre, slightly wider). It is now the
  Wilson interval, as documented.
- `appendix.RobustnessCertificationScore`: with list inputs, `==` compared whole
  lists, so any single difference gave an agreement of 0 for that perturbation.
  Inputs are now compared element-wise (and shapes are checked).
- `appendix.MutualInformationContent`: the normalization failed for string
  categories (it used `np.bincount`).
- `appendix.BiasConcentrationIndex`: the maximum of 1 - sum(p^2)/(sum p)^2 is
  1 - 1/n, so with few groups the verdict thresholds (0.3, 0.7) could not be
  reached (two equal groups scored 0.5, "moderate concentration"). The new
  `bci_normalized` divides by 1 - 1/n and the verdict uses it; `bci` is unchanged.
- `appendix.NetworkModularity` and `appendix.AdvancedNetworkMetrics.calculate_modularity`:
  the diagonal of a correlation matrix became self-loops, and the community search
  ignored the edge weights while the score used them. The diagonal is now ignored
  and the same weights are used for both. A failure now returns NaN with a warning
  (or raises, in `calculate_modularity`) instead of a silent 0.0. For a
  correlation matrix with two blocks the value moves from an artefact (0.417 in
  the review's example) to the correct 0.30.
- `statistics.NetworkStatistics.analyze_network`: the same self-loops pushed degree
  centrality above 1. The diagonal is ignored and a `threshold` option drops weak
  edges (unweighted centralities on a dense correlation matrix are uninformative).
- `appendix.AdvancedNetworkAnalysis`: `metric_correlation_network` left out
  metrics without an edge above the threshold (changing every centrality) and
  used a bare `except`; `concept_cooccurrence_network` matched substrings ("MI"
  was found in "family"), now whole words or phrases; `temporal_fairness_dynamics`
  treated (A, B) and (B, A) as different edges of an undirected graph.
- `domain5.UncertaintyQuantificationGap`: overlapping hedging terms were counted
  twice ("cannot rule out" also counted as "rule out"), and decimals ended a
  sentence ("0.04" made two sentences). Terms are now matched once, longest first,
  and sentences end only at '.', '!' or '?' followed by white space or the end.
- `domain1.InterRaterReliability.interpret_score` used strict inequalities while
  `calculate_icc_2_1` used inclusive ones, so 0.75 got two verdicts. Both now use
  Cicchetti's (1994) bands (>=0.75, >=0.60, >=0.40).
- `domain2.HarmAdjustedFairnessGap`: the case-level bootstrap pooled the two groups,
  so resampled group sizes varied; it now resamples within each group.
- `statistics.HierarchicalLinearModeling.fit_model`: `level2_predictors` were
  accepted but ignored (now added as fixed effects); a failed mixed model fell back
  to the ANOVA decomposition silently (now with a warning and `method` = "anova");
  in that fallback, `variance_between_groups` held the mean square rather than the
  variance component, and the group size ignored imbalance (now
  sigma_u^2 = (MSB - MSW)/n0 with the effective size n0; the mean squares are
  reported as `ms_between`, `ms_within`).
- `statistics.MediationAnalysis`: "Complete mediation" required a direct effect of
  exactly 0 and so was practically never returned. The direct effect now has a
  bootstrap CI (`direct_ci_lower`, `direct_ci_upper`) and the type follows Zhao,
  Lynch and Chen (2010).
- `inference.wilson_ci`: the bounds are now exactly 0 at k = 0 and 1 at k = n
  (previously about 1e-17 off).
- `utils.plot_figure3_corpus_comparison` failed with matplotlib 3.11
  (`boxplot()` no longer accepts `labels`); it now passes `tick_labels` when
  available and `labels` on older matplotlib.

### Changed (bundled reference data)
- `WHO_REGION_IHD_BURDEN` now holds each WHO region's share of ischaemic heart
  disease DALYs in 2023, from the WHO Global Health Estimates 2023 (Geneva: WHO;
  2026). Up to 1.9.5 it held age-standardised DALY-rate shares attributed to the
  GBD 2019 study whose regional values could not be traced to a published table.
  New: `WHO_REGION_IHD_BURDEN_RATE` (crude-rate shares),
  `WHO_GHE2023_IHD_DALYS_THOUSANDS`, `WHO_GHE2023_POPULATION_THOUSANDS` (the
  published figures) and `WHO_REGION_CODES` (WHO GHO codes such as "AFR" to the
  "AFRO" keys).

### Changed (errors instead of silently wrong results)
- `BurdenEvidenceMismatch`: a region with evidence but no entry in
  `burden_shares` raises `ValueError` (mismatched codes such as "AFR" against
  "AFRO" previously gave BEMI = 1). Record labels that are not among the regions
  of the point estimate raise `ValueError` in BEMI, GCC and GRBI, and a warning is
  given when the records imply shares different from the point estimate.
- `DecisionFlipRate`, `InstructionalVulnerabilityIndex`: missing outputs (None or
  NaN) raise `ValueError`; NaN never equals itself and was counted as a flip.
  An empty input to DFR raises `ValueError`.
- `ClinicalHallucinationRate`: a NaN support score was counted as a supported
  claim; NaN or out-of-range scores, and negative weights, now raise `ValueError`.
- `HierarchicalEquityRatio`: a reference score of 0 gave every group, the
  reference included, HER = 0; it now raises `ValueError`.
- `EthicalRiskIndex`: more violations than outputs now raises `ValueError`.
- `InterRaterReliability.calculate_icc_2_1` needs at least 2 items and 2 judges;
  `EmbeddingConsistencyScore` checks that the two arrays have the same shape.
- `HarmAdjustedFairnessGap` warns when the case lists disagree with the error
  counts, or when the groups differ in size by more than 10% (HAFG compares total
  harm; use counts per 1,000 patients or wHAFG).
- `inference.permutation_test` and the score test of `proportion_ci` raise on an
  unknown `alternative` (a typo such as "two_sided" was treated as "less" or as
  "two-sided").

### Added
- `SemanticParityGap`: `p_value_permutation` (1000 label permutations), because
  a centroid distance is positive even without a true difference and its
  bootstrap interval never contains 0.
- `TemporalFairnessDrift.calculate_drift(..., sigma_method="moving_range")`: the
  individuals-chart sigma (mean moving range / 1.128), which a drift does not
  inflate. The default ("sd") is unchanged.
- `WassersteinDistance.calculate_wd(..., support=...)` compares two histograms on
  shared bin locations; without it the inputs are, as before, samples.
- `IntersectionalCalibrationError`: `n_by_group`, to read dICE alongside group
  sizes (binned ECE is biased upward in small groups).
- `random_state` for `AdvancedReliabilityMetrics.calculate_bootstrap_ci` (it used
  the global NumPy generator) and for the synthetic generators in
  `utils.data_loader`.

### Documentation
- `Metric_Math_Derivations.md`: the worked instances now use invented,
  illustrative counts instead of quoting results of a specific study; ATS shows
  the Wilson formula; the modularity bootstrap uses B = 200, as implemented.
- Jensen-Shannon examples in the README and the vignette passed raw samples,
  which are compared position by position; they now histogram both samples on
  common bins.
- The README, metrics guide, API reference and vignette describe every change
  above; example outputs in the API reference now match what the code returns;
  verdict cut-offs are described as heuristics, not validated thresholds.
- The README citation, `CITATION.cff` and `.zenodo.json` give the author as
  John Weirstrass Muteba Mwamba, with ORCID and affiliation.

### Maintenance
- Code formatted with black and isort (the project's CI checks), in a separate
  commit with no functional change.
- `.claude/` (local editor settings) is no longer tracked.

## [1.9.5] - 2026-06-19

### Documentation
- Fixed the last non-rendering inline math in `docs/VIGNETTE.md` (Statistics
  module and Appendix sections) and `docs/Metric_Math_Derivations.md`. The cause
  was inline `$...$` expressions containing two or more `_{...}` subscripts, whose
  underscores markdown pairs as emphasis and deletes (e.g.
  `\sigma^{2}_{\text{between}}` rendered as `\sigma^{2}{\text{between}}`, and
  `[\mathrm{BCI}^{\ast}_{(0.025)}, \mathrm{BCI}^{\ast}_{(0.975)}]` lost its
  subscripts). The HLM ICC formula is now a display block, the appendix bootstrap
  intervals are described in words ("2.5th and 97.5th percentiles of the bootstrap
  replicates"), and remaining multi-subscript inline fragments were split so no
  inline expression carries more than one `_{...}`.

## [1.9.4] - 2026-06-19

### Documentation
- Fixed the remaining confidence-interval expressions that did not render in
  `docs/VIGNETTE.md` and `docs/Metric_Math_Derivations.md`. Inline `$...$` math
  containing a literal `*` (the bootstrap-replicate star, `\hat\theta^{*}`) had
  the asterisks consumed by markdown emphasis; all such stars are now written as
  `\ast` (identical glyph, no markdown-special character). Also fixed
  hyphen-adjacent inline math that was not recognised (`length-$n$`, `Fisher-$z$`,
  `pseudo-$R^2$`, `$1-$CPS`) by adding a space or using plain text in table cells.
- Replaced the stray `99` (a leaked `\begin{thebibliography}{99}` argument) before
  the derivations reference list with a proper `# References` heading. All
  references were verified as real, foundational sources.

### Changed
- Cite the Zenodo **concept DOI** 10.5281/zenodo.20766188 ("all versions",
  resolves to the latest) instead of the per-version DOI, in `CITATION.cff` and
  the manuscript.

## [1.9.3] - 2026-06-19

### Documentation
- Confidence-interval equations now render correctly in `docs/VIGNETTE.md` and
  `docs/Metric_Math_Derivations.md`. Two markdown-structure bugs were fixed:
  (1) display `$$...$$` equations wedged between prose lines without blank-line
  separation, which let markdown emphasis consume the subscript underscores
  (`\mathrm{HER}_g` rendered as `\mathrm{HER}g`); every display equation is now
  blank-line isolated. (2) a `math` code fence whose closing ``` had trailing
  note text on the same line, so the block never closed; notes now sit on their
  own line after the fence.

### Added
- Citable archived release: Zenodo DOI 10.5281/zenodo.20766189 recorded in
  `CITATION.cff` and the manuscript (references and Data-sharing statement).

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
