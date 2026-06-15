# EquiMed-DSS Metrics Guide

EquiMed-DSS implements **37 metrics**: five core domains (26), a geographic
module (2), and an advanced appendix (9). Every entry below lists the exact
class, method, formula as implemented, output range, and interpretation. All
classes are instantiated with no arguments unless noted.

| Domain | Module | Metrics |
|---|---|---|
| 1. Reliability & robustness | `domain1` | DecisionFlipRate, EmbeddingConsistencyScore, InterRaterReliability (ICC) |
| 2. Fairness, equity & ethics | `domain2` | HER, HAFG, ERI, IBS |
| 3. Governance & transparency | `domain3` | TFD, ATS, GCI |
| 4. Representation & robustness | `domain4` | SPG, CHR, IVI, GRI |
| 5. Technical-supplement fairness | `domain5` | ICE, wHAFG, LDDI, REG, CPS, CIDR, DCI, UQG, GRBI, HSSF, ISFV, SRPI |
| Geographic | `geographic` | BEMI, GCC |
| Appendix | `appendix` | BCI, power, Bland-Altman, MIC, JSD, WD, NM, TS, RCS |

---

## Domain 1 — Reliability & robustness

### Decision Flip Rate (DFR) — `domain1.DecisionFlipRate.calculate_dfr`
Diagnostic instability under input perturbation (e.g. a demographic flip).
**Formula:** `DFR = (1/n) Σ 1[decision_i ≠ counterfactual_i]`, range [0, 1].
Reported with a **Wilson 95% interval** (`ci_lower`, `ci_upper`), `n_flipped`,
`n_samples`. Lower is better: <0.05 excellent, <0.15 moderate, else high instability.

### Embedding Consistency Score (ECS) — `domain1.EmbeddingConsistencyScore.calculate_ecs`
Semantic shift of embeddings under perturbation. **Formula:** per item
`1 − cos(E_orig, E_pert)`; returns `mean_ecs`, `std_ecs`, `median_ecs`,
range [0, 2] (typically [0, 1]). Lower = more consistent.

### Inter-Rater Reliability (ICC) — `domain1.InterRaterReliability.calculate_icc_2_1`
ICC(2,1), two-way random effects, single measure, absolute agreement.
**Formula:** `ICC = (MSR − MSE) / (MSR + (k−1)·MSE + (k/n)(MSC − MSE))`, range
[0, 1]. `bland_altman_analysis` reports mean difference and 95% limits of
agreement (sample SD). >0.75 excellent, >0.6 good, >0.4 fair.

---

## Domain 2 — Fairness, equity & ethics

### Hierarchical Equity Ratio (HER) — `domain2.HierarchicalEquityRatio`
`calculate_her(group_scores, reference_group)` → `HER_g = score_g / score_ref`
(4/5ths rule: equitable in [0.8, 1.25]). `calculate_bias_gini(scores)` returns
the standard Gini of group scores (dispersion).

### Harm-Adjusted Fairness Gap (HAFG) — `domain2.HarmAdjustedFairnessGap`
Constructor `HarmAdjustedFairnessGap(cost_fn=10.0, cost_fp=3.0)`; method
`calculate_hafg(group1_errors, group2_errors)` with `{"fn":…, "fp":…}` counts.
**Formula:** `H_g = fn_g·cost_fn + fp_g·cost_fp`;
`HAFG = |H1 − H2| / max(H1, H2)`, range **[0, 1]** (normalized; the raw gap is
also returned as `absolute_harm_gap`). <0.1 minimal, <0.2 moderate, else significant.

### Ethical Risk Index (ERI) — `domain2.EthicalRiskIndex.calculate_eri`
`ERI = Σ severity_i / n_total_outputs`; also returns `svr` (violations per 1000).
Lower is better.

### Intersectional Bias Score (IBS) — `domain2.IntersectionalBiasScore`
`calculate_subgroup_similarity(vectors)`: pairwise Euclidean distances,
similarity `1/(1+d)`, flags the subgroup with the largest mean distance.
`interaction_analysis(df)`: variance-decomposition (eta²-style) main and
race×gender interaction effects (does not mutate the input DataFrame).

---

## Domain 3 — Governance & transparency

### Temporal Fairness Drift (TFD) — `domain3.TemporalFairnessDrift.calculate_drift`
3-sigma statistical process control: `UCL/LCL = mean ± 3·SD` (sample SD);
flags out-of-control points. Drift detected if any point exceeds the limits.

### Audit Traceability Score (ATS) — `domain3.AuditTraceabilityScore.calculate_ats`
`calculate_ats(n_traceable, n_total)` → proportion traceable with a **Wilson 95%
interval**; meets standard at ≥0.95.

### Governance Compliance Index (GCI) — `domain3.GovernanceComplianceIndex.calculate_gci`
`calculate_gci(policy_compliance: Dict[str,bool])` → `met / total`, range [0, 1];
returns `compliance_gaps`. 1.0 = full compliance.

---

## Domain 4 — Representation & robustness

### Semantic Parity Gap (SPG) — `domain4.SemanticParityGap.calculate_spg`
Latent bias as the distance between embedding centroids of identical cases that
differ only by a protected attribute. Returns **both** `spg_euclidean`
(`‖c_p − c_m‖₂`) and `spg_cosine` (`1 − cos(c_p, c_m)`). Larger = more identity
sensitivity. (State which variant you report.)

### Clinical Hallucination Rate (CHR) — `domain4.ClinicalHallucinationRate.calculate_chr`
`CHR = (1/|C|) Σ 1[support(c) < τ]` over per-claim NLI/entailment support scores
(default τ=0.5); severity-weighted variant via `weights`. Range [0, 1]; higher is worse.

### Instructional Vulnerability Index (IVI) — `domain4.InstructionalVulnerabilityIndex.calculate_ivi`
`IVI = P(f(q_biased) ≠ f(q_neutral))` over paired neutral/biased outputs;
`ivi_effect` is the directional mean change for numeric outputs. Range [0, 1].

### Geographic Representation Index (GRI) — `domain4.GeographicRepresentationIndex`
`calculate_gri(locations, western_locations)` → `(|L| − |W|)/|L|` (set-based
non-Western variety, [0, 1]). `calculate_geographic_bias(gri_values, error_rates)`
correlates GRI with non-Western error rate.

---

## Domain 5 — Technical-supplement fairness

### Intersectional Calibration Error (ICE) — `domain5.IntersectionalCalibrationError.calculate_ice`
Per intersectional group, `ECE_i = Σ_b (|S_ib|/|S_i|)·|acc − conf|`;
`ICE = Σ_i w_i·ECE_i` (population weighted); `dICE = max_i ECE_i − min_i ECE_i`.

### Weighted Clinical Harm-Adjusted Fairness Gap (wHAFG) — `domain5.WeightedClinicalHarmAdjustedFairnessGap.calculate_whafg`
`H(g) = (1/n_g) Σ ω(Y_i)·L(Ŷ_i, Y_i)`; `wHAFG = max_g H(g) − min_g H(g)`.
Per-sample, severity-weighted generalization of domain-2 HAFG.

### Lexical Diversity Disparity Index (LDDI) — `domain5.LexicalDiversityDisparityIndex.calculate_lddi`
Root Type-Token Ratio `RTTR(g) = |V_g|/√(tokens_g)`; `LDDI = max_g − min_g`,
plus `lddi_norm`.

### Recommendation Entropy Gap (REG) — `domain5.RecommendationEntropyGap.calculate_reg`
`H(T|g) = −Σ_t P(t|g) log₂ P(t|g)`; `REG = max_g − min_g` (bits).

### Counterfactual Parity Score (CPS) — `domain5.CounterfactualParityScore.calculate_cps`
`CPS = mean response similarity under a demographic swap`; counterfactual
unfairness `CFU = 1 − min_pair CPS` (or `1 − CPS` for a single pair). Range [0, 1].

### Clinical Information Density Ratio (CIDR) — `domain5.ClinicalInformationDensityRatio.calculate_cidr`
`CID(g) = mean (concepts/tokens)·100`; `CIDR(g) = CID(g)/max_g CID`; `cidr_min`
is the most information-sparse group (1.0 = parity).

### Diagnostic Completeness Index (DCI) — `domain5.DiagnosticCompletenessIndex.calculate_dci`
`DCI(r) = |D(r) ∩ D*| / |D*|` against a reference differential set D*;
group means and `dDCI = max − min`; optional severity weights.

### Uncertainty Quantification Gap (UQG) — `domain5.UncertaintyQuantificationGap.calculate_uqg`
`UD(r) = hedging terms / sentences`; `UQG = max_g − min_g` (hedging-density disparity).

### Geographic Representation Bias Index (GRBI) — `domain5.GeographicRepresentationBiasIndex.calculate_grbi`
`GRBI = D_KL(P_corpus ‖ P_burden) = Σ_r p_c(r) log(p_c(r)/p_b(r))` (nats);
optional HIC over-representation ratio. Directed KL complement to BEMI.

### Healthcare System Stratified Fairness (HSSF) — `domain5.HealthcareSystemStratifiedFairness.calculate_hssf`
`HSSF = Σ_s P(s)·max_{g,g'}|E[Y|g,s] − E[Y|g',s]|` (within-system gap);
`delta_between = Var_s(E[Y|s])` (between-system).

### Intersectional Shapley Fairness Value (ISFV) — `domain5.IntersectionalShapleyFairnessValue.calculate_isfv`
Cooperative-game Shapley attribution of the disparity
`v(S) = max−min of E[Y | A_S]` to each protected attribute, plus pairwise
interactions `v({i,j}) − v({i}) − v({j})`. Shapley values sum to the total disparity.

### Semantic Robustness Parity Index (SRPI) — `domain5.SemanticRobustnessParityIndex.calculate_srpi`
`SRPI = min_g R(g) / max_g R(g)` over per-group paraphrase robustness (1 = equal).

---

## Geographic

### Burden-Evidence Mismatch Index (BEMI) — `geographic.BurdenEvidenceMismatch.calculate_bemi`
**Total-variation distance** between regional evidence and disease-burden shares:
`BEMI = ½ Σ_r |evidence_r − burden_r|`, range **[0, 1]** (0 = evidence tracks
burden, 1 = disjoint). Use `WHO_REGION_IHD_BURDEN` (normalized IHD DALY-rate
shares, Roth et al. 2020). <0.10 low, <0.25 moderate, ≥0.25 high mismatch.

### Geographic Concentration of Coverage (GCC) — `geographic.GeographicConcentration.calculate_gcc`
Sample-corrected Gini `G* = R/(R−1)·G_raw` (0 even, 1 single-region) and
normalized Shannon entropy `H_norm = −Σ p_r ln p_r / ln R` (1 even, 0 single);
`concentration = 1 − H_norm`.

---

## Appendix — advanced metrics (`appendix.advanced_metrics`, also re-exported)

- **BootstrapConfidenceIntervals** `calculate_bci` — percentile bootstrap CI for any statistic.
- **StatisticalPowerAnalysis** `calculate_sample_size` / `calculate_power`.
- **BiasConcentrationIndex** `calculate_bci` — concentration of bias across subgroups.
- **MutualInformationContent (MIC)** `calculate_mic` — **mutual information** between
  demographics and outcomes (NOT the Reshef Maximal Information Coefficient);
  prefer `normalized_mic` for cross-setting comparison.
- **JensenShannonDivergence (JSD)** `calculate_jsd` — JS **divergence**, base 2,
  range [0, 1]; `jsd_distance` is its square root. (Consistent with
  `appendix.info_theory.AdvancedInfoTheoryMetrics.calculate_jsd`.)
- **WassersteinDistance (WD)** `calculate_wd` — earth-mover distance (scipy).
- **NetworkModularity (NM)** `calculate_modularity` — Newman modularity Q over
  greedy (Clauset-Newman-Moore) communities.
- **TransparencyScore (TS)**, **RobustnessCertificationScore (RCS)**.

`appendix.info_theory`, `appendix.network`, and `appendix.reliability` provide
thin aggregator classes over the same statistics; the canonical implementations
live in `advanced_metrics.py`.

---

## Choosing metrics

- **Diagnostic systems:** ICE/ECS (calibration/consistency), HER (equity), TFD (drift), CHR (faithfulness).
- **Triage/decision support:** IVI (prompt robustness), SPG/CPS (identity sensitivity), HSSF (system confounding).
- **Evidence/corpus audits:** BEMI, GCC, GRBI, GRI (geographic equity).
- **Governance/regulatory:** GCI, ATS, ICC (reliability), ERI.

## References
1. Rajkomar A, et al. Ensuring Fairness in ML to Advance Health Equity. 2018.
2. Obermeyer Z, et al. Dissecting racial bias in an algorithm. Science 2019.
3. Roth GA, et al. Global Burden of Cardiovascular Diseases (GBD). 2020.
4. Shrout PE, Fleiss JL. Intraclass correlations. Psychol Bull 1979.
