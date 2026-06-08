# EquiMed-DSS v1.1.0: Geographic Metrics + Statistical Table Outputs

Date: 2026-06-07
Status: Approved (design), pending spec review
Target version: 1.0.2 -> 1.1.0 (backward-compatible feature release)

## Goal

Extend the published `equimed_dss` library with (a) a geographic-equity metric
module, and (b) a reporting layer that turns the existing statistics results
into publication-ready tables. Examples must be compelling enough to demo the
library at conferences. Backward compatibility with 1.0.2 is required.

Scope confirmed with the user (2026-06-07): the original "item 1" was a typo;
real scope is the four items below.

## Non-goals (YAGNI)

- No new statistical estimators beyond the geographic metrics. The table layer
  wraps existing `statistics/` results; it does not recompute them.
- No bundling of MIMIC or any DUA-restricted patient data in the package.
- No CLI, no web UI, no plotting overhaul (existing `utils/visualization.py`
  stays as is).

## Components

### 1. New package: `equimed_dss/geographic/`

Two metrics, one module (user chose "both"). Both get conference-citable names.

- **`BurdenEvidenceMismatch` (BEMI)** in `burden_evidence.py`
  - Input: per-region evidence share and per-region disease-burden share
    (default reference = bundled GBD WHO-region IHD burden constant; user may
    pass their own).
  - Output:
    - per-region table: `region`, `evidence_share`, `burden_share`,
      `mismatch` (evidence - burden), `ratio` (evidence / burden)
    - scalar `bemi_index`: index of dissimilarity
      `0.5 * sum_r |evidence_share_r - burden_share_r|`, range 0-1
      (0 = evidence perfectly tracks burden; 1 = maximal divergence). Both
      share vectors are normalized to sum to 1 before the calculation. This is
      exactly the total-variation distance between the two distributions
      (verified 2026-06-07: bounds [0,1] proven, disjoint-support = 1,
      identical = 0; equals total excess evidence mass to reallocate).
  - Formalizes the manuscript finding: AFRO + SEARO ~36% IHD burden, ~0%
    retrieved evidence.

- **`GeographicConcentration` (GCC)** in `concentration.py`
  - Input: per-region evidence counts or shares (no burden reference needed).
  - Output: normalized Gini and normalized Shannon entropy of geographic
    coverage, plus a per-region contribution table.
  - Formula precision (verified 2026-06-07):
    - **Gini MUST be sample-corrected**: `G* = (R/(R-1)) * G_raw` where
      `G_raw = sum_i sum_j |x_i - x_j| / (2 R sum_k x_k)` and R = number of
      regions. Without the `R/(R-1)` factor the maximum (single-region) Gini is
      only `(R-1)/R` (e.g. 0.833 for R=6), so the "single region -> 1" property
      fails. Use `G*` so the range is a clean [0,1]. High G* = concentrated.
    - **Normalized Shannon entropy**: `H_norm = -sum_r p_r ln(p_r) / ln(R)`
      with `0 ln 0 := 0`, range [0,1]. Uniform -> 1, single-region -> 0.
    - DIRECTION CAVEAT: Gini and entropy run opposite (uniform gives Gini 0 but
      entropy 1). Both are reported; also expose `concentration = 1 - H_norm`
      so a single "higher = more concentrated" reading is available alongside
      Gini.

- Bundled reference data: a small `WHO_REGION_IHD_BURDEN` constant (published
  GBD aggregate shares, citable, not patient-level). Lives in the geographic
  package, documented with its source citation.

`geographic/__init__.py` exports `BurdenEvidenceMismatch`, `GeographicConcentration`,
and the reference constant. Top-level `equimed_dss/__init__.py` re-exports them.

### 2. New package: `equimed_dss/reporting/`

`tables.py` converts existing statistics result dicts into tidy
`pandas.DataFrame`s (one row per coefficient / effect / node / region):

- `hierarchical_coefficients_table(results)` -> columns: term, estimate,
  variance_component, icc, aic, bic, n (drawn from
  `HierarchicalLinearModeling.fit_model` dict: `icc`,
  `variance_between_groups`, `variance_within_groups`, `total_variance`,
  `r_squared_marginal`, `aic`, `bic`, `n_groups`, `n_observations`).
- `mediation_effects_table(results)` -> rows for total / direct / indirect
  effect; columns: effect, estimate, ci_lower, ci_upper, proportion_mediated,
  classification (from `MediationAnalysis.analyze_mediation`: `total_effect`,
  `direct_effect`, `indirect_effect`, `indirect_ci_lower`, `indirect_ci_upper`,
  `proportion_mediated`, and the classification string).
  CAVEAT (verified 2026-06-07): proportion_mediated = indirect/total is
  numerically unstable when total ~ 0 and can fall outside [0,1] under
  competitive/suppression mediation (ab and direct effect opposite signs). The
  table must show it as-is with a flag, never clamp it silently; the existing
  `_classify_mediation` already labels the competitive case.
- `network_centrality_table(results)` -> one row per node; columns: node,
  degree, betweenness, closeness, clustering (from
  `NetworkStatistics.analyze_network`).
- `geographic_table(result)` -> per-region rows for BEMI / GCC outputs.

`export_table(df, fmt="markdown"|"latex"|"html", path=None, decimals=3)`:
applies consistent rounding/formatting, then delegates to pandas'
`to_markdown` / `to_latex` / `to_html`. Returns the rendered string and
optionally writes to `path`. This is the user's chosen "DataFrame + exporters"
shape: tidy DataFrame returns plus thin export helpers.

`reporting/__init__.py` exports the four table functions and `export_table`.

### 3. Examples (more examples)

- New `examples/example_statistics_tables.py`: runs hierarchical, mediation,
  and network analyses on bundled sample data, renders each as a table, and
  exports markdown + LaTeX to files.
- New `examples/example_geographic.py`: BEMI + GCC demo using the bundled
  WHO-region burden reference; renders the geographic table.
- Data stance (user-approved): examples use **bundled illustrative sample
  data**, clearly labelled as synthetic, because the package is public and
  MIMIC is under DUA. Each new example also includes a documented
  **real-data hook**: a commented block / optional path argument showing how to
  point the same code at the user's real result JSON locally to render the
  actual manuscript tables for slides.
- Any example file touched gets `np.random.seed(...)` replaced with
  `np.random.RandomState(seed)` to match the user's reproducibility rule.

### 4. Tests, docs, packaging

- `tests/test_geographic.py`: known-value checks for BEMI (identical -> 0,
  disjoint support -> 1, plus a hand-computed mismatch case) and GCC using the
  SAMPLE-CORRECTED Gini G* = (R/(R-1)) G_raw (uniform -> Gini 0, entropy 1;
  single-region -> Gini 1, entropy 0). Asserting single-region Gini == 1
  validates the R/(R-1) correction is present.
- `tests/test_reporting.py`: each table function returns a DataFrame with the
  expected columns and row count; `export_table` produces non-empty markdown
  and LaTeX strings.
- Docs updated: `CHANGELOG.md` (1.1.0 entry), `docs/METRICS_GUIDE.md`
  (geographic section), `docs/API_REFERENCE.md` (new classes/functions),
  `docs/VIGNETTE.md` (tables + geographic walkthrough), `README.md`
  (feature list).
- `__version__.py` bumped to `1.1.0`.

### 5. Verification + delivery

1. `pytest` full suite passes (new + existing).
2. Build sdist + wheel; `twine check dist/*` passes.
3. With tokens the user provides at delivery (handled transiently, never
   written to disk or committed):
   - upload 1.1.0 to **TestPyPI**;
   - merge `feature/v1.1.0-geographic-and-tables` into `master` and push to
     GitHub `johnmuteba/EquiMed_DSS` (matches the existing master-based
     workflow; the repo stays private).
4. User tests 1.1.0 locally from TestPyPI, then decides whether to publish to
   real PyPI. Manuscript revision is downstream of that and is out of scope
   here.

## Data flow

```
existing statistics/*.py  --dict-->  reporting/tables.py  --DataFrame-->  export_table  --str/file-->  slides/manuscript
bundled burden constant + user shares  -->  geographic/*.py  --dict/DataFrame-->  geographic_table --> export_table
```

## Risks / decisions

- Illustrative sample data means conference example tables show synthetic
  numbers by default; the real-data hook lets the user swap in real result
  files on their own machine (DUA-safe, nothing real is shipped).
- BEMI index definition (L1/2 dissimilarity, 0-1) is a modelling choice; it
  must match whatever the manuscript reports for the geographic gap so the
  library and paper agree. Verify against `geography_distinctive_layer.py`
  output during implementation.
- Backward compatibility: only additive changes (new packages, new functions,
  version bump). No existing public signature changes.
```
