# EquiMed-DSS — Pre-release Metric & Formula Audit

> **STATUS: all findings RESOLVED in v1.5.0** (2026-06-13). Code fixes (DFR
> Wilson CI, JSD consistency, HAFG normalization, IBS DataFrame copy, ddof=1),
> documentation overhaul (METRICS_GUIDE + API_REFERENCE rewritten to the real
> 37-metric API; stale findings removed; MIC/modularity wording corrected), and
> the visualization refactor (return figures, no `plt.show()`) are applied. Tests:
> 124 passed; all 10 examples run. The sections below are the original audit,
> retained as a record; each item is now fixed (see `CHANGELOG.md` [1.5.0]).

---

## Original audit (v1.4.2)

**Scope:** every metric, formula, docstring, and the logic in `equimed_dss/`,
plus tests and docs. **Method:** read each implementation, checked the math
against its docstring/the literature, recomputed selected metrics, and ran the
test suite. **Test suite: 124 passed.**

## Headline verdict

- The **16 Domain-4 and Domain-5 metrics** (SPG, CHR, IVI, GRI, ICE, wHAFG,
  LDDI, REG, CPS, CIDR, DCI, UQG, GRBI, HSSF, ISFV, SRPI) and the **geographic
  module** (BEMI, GCC) — i.e. the metrics used in the manuscript — are
  **correctly specified, correctly implemented, and well documented.** These are
  publication- and release-grade.
- The **older Domain-1/2/3 metrics and the appendix** are mostly correct but
  contain a few real bugs and several inconsistencies.
- The **documentation (`METRICS_GUIDE.md`, package docstrings) is badly out of
  date** and is the single biggest blocker to a clean release: it describes a
  different, older 19-metric set, with wrong formulas, class names, and APIs for
  Domain 1, and omits Domains 4–5 entirely.

Nothing here is a reason not to publish, but the **must-fix** items below should
be done before `pip install equimed-dss` is pointed at real users.

---

## 0. Critical clarification: BEMI is total-variation distance (not 1 − Pearson)

The library is correct and unambiguous:

```
BEMI = 0.5 * sum_r |evidence_share_r - burden_share_r|     (range [0, 1])
```

This is the **total-variation distance**. Computed on the verified corpus
geography it returns **0.6666 ≈ 0.67**, which is the value originally reported in
the manuscript. (An earlier manuscript edit had mistakenly redefined BEMI as
"1 − Pearson(b,e) = 1.51"; that has now been reverted to the correct TVD
definition and value 0.67 in `main_new.tex` and the review package.) The library
needs no change here.

---

## 1. Module-by-module correctness

### Geographic — CORRECT
- `BurdenEvidenceMismatch` (BEMI): TVD, normalizes inputs, handles zeros, range
  proven. ✔
- `GeographicConcentration` (GCC): sample-corrected Gini `G* = R/(R-1)·G_raw`
  and normalized Shannon entropy — both standard and correct, with the right
  finite-category correction. ✔
- `reference_data.py`: `WHO_REGION_IHD_BURDEN` matches the pipeline (AFRO+SEARO =
  36.1%). **Wording nuance:** it normalizes age-standardized DALY **rates** per
  100k (equal regional population weight), so it is a rate-based regional share,
  not a population-weighted "share of global DALYs." Internally consistent, but
  the docstring/manuscript should say "regional IHD DALY-rate share."

### Domain 4 — CORRECT (manuscript metrics)
- CHR, IVI, GRI, SPG all match their docstrings and the literature. ✔
- **SPG returns both `spg_euclidean` and `spg_cosine`.** The manuscript should
  state which variant the reported value is (Euclidean centroid distance ≈ 0.31
  vs cosine ≈ 0.66 are *both* valid outputs of the same call).

### Domain 5 — CORRECT (all 12)
- ICE (group ECE + dICE), CPS/CFU, SRPI, GRBI (KL; independently reproduced at
  1.34 nats), wHAFG, ISFV (proper Shapley with efficiency), HSSF
  (within/between decomposition), LDDI (Guiraud RTTR), REG (Shannon),
  CIDR, DCI, UQG — every formula matches its docstring. ✔ This is the
  best-written part of the library.

### Statistics — CORRECT logic, stale docstrings
- `HierarchicalLinearModeling`: MixedLM with REML for variance/ICC and a proper
  ML refit for AIC/BIC. ✔ **Two issues:** (a) `r_squared_marginal` is actually a
  *proportional reduction in variance*, not the Nakagawa–Schielzeth marginal R²
  the name implies; (b) the docstring carries a **stale finding** ("55.8% …
  Manuscript Section 2.4") from a different study. Also note this class is
  **Gaussian** — it is not the binary-outcome logistic MAIHDA (π²/3) used for
  the CXR VPC in the paper; document that distinction.
- `MediationAnalysis`: product-of-coefficients with bootstrap percentile CIs —
  correct. ✔ **Stale docstring finding** ("72.1% … β=1.274").

### Reporting — CORRECT, good design
- `reporting/tables.py` + `export.py`: clean presentation layer (markdown/LaTeX/
  HTML). ✔

---

## 2. MUST-FIX before release (correctness bugs)

1. **`domain1/dfr.py` (DecisionFlipRate) — broken confidence interval.**
   `ci_lower = np.percentile(flips, 2.5)`, `ci_upper = np.percentile(flips,97.5)`
   on a 0/1 vector returns ≈ [0, 1] almost always; it is **not** a CI for the
   flip rate. Replace with a Wilson interval (you already have the pattern in
   `domain3/ats.py`) or a bootstrap over the mean.

2. **Appendix JSD — two implementations return different quantities.**
   `advanced_metrics.JensenShannonDivergence.calculate_jsd` returns the JS
   **distance** (√divergence ≈ 0.258 on a test case); `info_theory.
   AdvancedInfoTheoryMetrics.calculate_jsd` returns the **divergence** (≈ 0.066).
   Same name, different numbers. Pick one definition, make both agree, and fix
   the range label (with scipy's default base-e the distance maxes at ≈ 0.83,
   not 1.0; pass `base=2` for a true [0,1]).

3. **Appendix MIC — name vs. implementation mismatch.**
   `METRICS_GUIDE` calls it "Maximal Information Coefficient (detects non-linear
   relationships)" (Reshef 2011), but the code computes plain **mutual
   information** (`mutual_info_score`). These are different statistics. Either
   rename to "Mutual Information" everywhere (simplest, matches the code) or
   implement true MIC. Raw MI is unbounded and category-count dependent, so the
   `<0.1 / <0.3` thresholds are not well-calibrated — prefer the
   `normalized_mic` you already compute.

4. **`domain2/hafg.py` (HAFG) — unnormalized + arbitrary threshold + wrong API.**
   The guide defines `HAFG = |H1−H2| / max(H1,H2)` ∈ [0,1]; the code returns the
   raw `|H1−H2|` ∈ [0,∞), which is not comparable across datasets, and the
   verdict uses a hardcoded `>10` ("Arbitrary threshold for example" in the
   code). The documented call `calculate_hafg(..., fn_harm_weight=10, fp_harm_weight=1)`
   doesn't exist (weights are constructor args; default `cost_fp=3`, not 1).
   Normalize, set a principled threshold, and align the API + docs.

## 3. SHOULD-FIX (robustness / correctness-adjacent)

5. **`domain2/ibs.py` mutates the caller's DataFrame** (`df["race_gender"]=...`).
   Operate on a copy. Its outlier rule (mean pairwise distance) also differs from
   the documented "distance to grand centroid"; reconcile.
6. **Sample SD vs population SD:** Bland-Altman (`icc.py`) and the control chart
   (`tfd.py`) use `np.std` (ddof=0). Bland-Altman limits and SPC limits
   conventionally use the sample SD (ddof=1); this slightly narrows the limits.
7. **Threshold inconsistencies between code and guide** (e.g. ERI verdict uses
   `>1.0` in code vs `≥0.1` in the guide; TFD docstring promises Western-Electric
   2-of-3-sigma rules that aren't implemented). Make code and docs agree.
8. **Appendix duplication:** `advanced_metrics.py` reimplements MIC/JSD/WD/
   modularity that also live in `info_theory.py`/`network.py`/`reliability.py`.
   Keep one canonical implementation, deprecate/remove the other (this is the
   root cause of issue #2). `NetworkModularity` docstring says "Louvain" but uses
   greedy modularity — fix the wording.

## 4. DOCUMENTATION — must overhaul before release (highest user impact)

The code is ahead of the docs by an entire metric generation.

- **Metric count is inconsistent:** `README` says **37** (correct: D1–5 = 26 +
  geographic 2 + appendix 9), but `__init__.py` docstring and
  `docs/METRICS_GUIDE.md` say **19**. Standardize on 37 and on "five domains +
  geographic + appendix."
- **`docs/METRICS_GUIDE.md` describes the wrong Domain 1 entirely:** it documents
  "Dynamic Fairness Ratio" and "Expected Calibration Score" with class names
  `DynamicFairnessRatio` / `ExpectedCalibrationScore` and method `calculate_icc`,
  none of which exist. The real classes are `DecisionFlipRate`,
  `EmbeddingConsistencyScore`, `InterRaterReliability` (method
  `calculate_icc_2_1`). Several documented call signatures (HAFG, ATS, GCI, ICC)
  do not match the code. **Rewrite the guide against the actual API**, and add
  the 16 Domain-4/5 metrics, which are missing from it.
- **`__init__.py` docstring** still says "19 novel metrics … domains 1–3" and
  omits domains 4–5 that it imports. Update.
- **Remove stale embedded findings** ("55.8%", "72.1%", "β=1.274", "Manuscript
  Section/Equation X") from `hierarchical.py`, `mediation.py`, and the appendix
  docstrings — they belong to an earlier project and will confuse users.
- Re-verify `docs/API_REFERENCE.md` and `docs/VIGNETTE.md` against the code (not
  read line-by-line here, but likely share the same drift).

## 5. Presentation, structure & plotting

- **`utils/visualization.py` calls `plt.show()` in every plot function and does
  not return the Figure.** This is a library anti-pattern: it blocks in scripts,
  fails headless, can't be embedded/composed, and is the source of the
  `FigureCanvasAgg is non-interactive` warnings in the test run. **Return
  `(fig, ax)`, drop `plt.show()` (or gate it behind `show=False`), and let the
  caller decide to `savefig`/`show`.** Keep the `dpi=300, bbox_inches="tight"`
  you already use.
- Standardize every metric's return dict: you already have a nice
  `{value, ..., interpretation:{range, ideal, verdict}}` pattern in Domains 1–3
  and a flat `{metric, interpretation:str}` in Domains 4–5. Pick one schema so
  downstream tables/plots are uniform.
- The `reporting/` table layer is good; consider a single `summary_table(results)`
  that renders any metric's dict to markdown/LaTeX for manuscripts.
- Add the **GCC and the geographic metrics to a worked figure** (burden vs
  evidence bubble/bar) in the vignette — it is your most visual result.

## 6. Pre-upload checklist (TestPyPI → PyPI → GitHub)

1. Fix bugs #1–#4 (DFR CI, JSD, MIC, HAFG) and add regression tests for each
   (the current suite passes but doesn't cover these).
2. Overhaul `METRICS_GUIDE.md` + `__init__` docstring + metric count to match the
   37-metric code; remove stale findings.
3. Make `visualization` return figures instead of `plt.show()`.
4. Bump version, update `CHANGELOG.md`, confirm `pyproject.toml` metadata and
   classifiers.
5. `python -m build`; `twine check dist/*`; upload to **TestPyPI** first and
   `pip install` from it in a clean venv; run the examples and `pytest`.
6. Then PyPI + tag the GitHub release. Pin the version cited in the manuscript
   (v1.4.2) and archive a Zenodo DOI for the Data-sharing statement.

**Bottom line:** the science-facing metrics are correct and ready. Fix the four
code bugs, realign the documentation with the code, and make the plots return
figures, and the package is release-ready.
