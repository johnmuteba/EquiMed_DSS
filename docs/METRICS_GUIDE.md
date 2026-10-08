# EquiMed-DSS Metrics Guide

EquiMed-DSS implements **37 metrics**: five core domains (26), a geographic
module (2), and an advanced appendix (9). Every entry below lists the exact
class, method, formula as implemented, output range, and interpretation. All
classes are instantiated with no arguments unless noted.

> Equations render on GitHub via MathJax. From v1.7.0 a metric result also prints
> its value with a 95% confidence interval (e.g. `DFR = 0.250 :: 95% CI
> [0.046; 0.699]`); see `Metric_Math_Derivations.md` for the interval formulas.

> **Verdict labels are heuristics.** The cut-offs behind labels such as
> "excellent", "compliant" or "deployment ready" are conventions chosen for
> readability, not validated clinical or regulatory thresholds (ICC uses
> Cicchetti's 1994 bands). Report the value and its interval, and judge it
> against a threshold justified for your setting.

> **Missing values raise errors.** From 1.10.0, DFR, IVI and CHR raise
> `ValueError` on missing inputs (None or NaN) instead of counting them as a
> change (DFR, IVI) or as a supported claim (CHR). Drop or impute them first.

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

## Domain 1: Reliability & robustness

### Decision Flip Rate (DFR): `domain1.DecisionFlipRate.calculate_dfr`
Diagnostic instability under input perturbation (e.g. a demographic flip).
```math
DFR = \frac{1}{n}\sum_{i=1}^{n}\mathbb{1}\!\left[d_i \ne d_i^{\mathrm{cf}}\right], \qquad DFR \in [0,1].
```
Reported with a **Wilson 95% interval** (`ci_lower`, `ci_upper`), `n_flipped`,
`n_samples`. Lower is better: <0.05 excellent, <0.15 moderate, else high instability.

### Embedding Consistency Score (ECS): `domain1.EmbeddingConsistencyScore.calculate_ecs`
Semantic shift of embeddings under perturbation, per item:
```math
\mathrm{ECS}_i = 1 - \cos\!\left(E_i^{\mathrm{orig}}, E_i^{\mathrm{pert}}\right).
```
Returns `mean_ecs`, `std_ecs`, `median_ecs` (+ bootstrap 95% CI), range
$[0, 2]$ (typically $[0,1]$). Lower = more consistent.

### Inter-Rater Reliability (ICC): `domain1.InterRaterReliability.calculate_icc_2_1`
ICC(2,1), two-way random effects, single measure, absolute agreement:
```math
ICC = \frac{MS_R - MS_E}{MS_R + (k-1)\,MS_E + \frac{k}{n}\,(MS_C - MS_E)}, \qquad ICC \le 1.
```
ICC can be negative (less agreement than chance). `bland_altman_analysis` reports
mean difference and 95% limits of agreement (sample SD). Verdicts follow
Cicchetti (1994): $\ge 0.75$ excellent, $\ge 0.60$ good, $\ge 0.40$ fair (both
`calculate_icc_2_1` and `interpret_score` use these bands from 1.10.0).

---

## Domain 2: Fairness, equity & ethics

### Hierarchical Equity Ratio (HER): `domain2.HierarchicalEquityRatio`
`calculate_her(group_scores, reference_group)`:
```math
HER_g = \frac{s_g}{s_{\mathrm{ref}}} \quad (\text{4/5ths rule: equitable in } [0.8, 1.25]).
```
`calculate_bias_gini(scores)` returns the standard Gini of group scores (dispersion).

### Harm-Adjusted Fairness Gap (HAFG): `domain2.HarmAdjustedFairnessGap`
Constructor `HarmAdjustedFairnessGap(cost_fn=10.0, cost_fp=3.0)`; method
`calculate_hafg(group1_errors, group2_errors)` with `{"fn":…, "fp":…}` counts:
```math
H_g = fn_g\,c_{fn} + fp_g\,c_{fp}, \qquad
HAFG = \frac{\lvert H_1 - H_2 \rvert}{\max(H_1, H_2)} \in [0,1].
```
The raw gap is also returned as `absolute_harm_gap`. <0.1 minimal, <0.2 moderate, else significant.
HAFG compares **total** harm, so with groups of different sizes it reflects group size
as well as error rates: pass counts per 1,000 patients, or use wHAFG (harm per
patient). With `group1_cases` / `group2_cases`, the 95% CI resamples cases within
each group (stratified, from 1.10.0), and a warning flags case lists that disagree
with the counts or groups whose sizes differ by more than 10%.

### Ethical Risk Index (ERI): `domain2.EthicalRiskIndex.calculate_eri`
```math
ERI = \frac{1}{N}\sum_{i=1}^{N}\mathrm{severity}_i .
```
Also returns `svr` (violations per 1000). Lower is better.

### Intersectional Bias Score (IBS): `domain2.IntersectionalBiasScore`
`calculate_subgroup_similarity(vectors)`: pairwise Euclidean distances,
similarity $1/(1+d)$, flags the subgroup with the largest mean distance.
`interaction_analysis(df)`: variance-decomposition ($\eta^2$-style) main and
race$\times$gender interaction effects (does not mutate the input DataFrame).

---

## Domain 3: Governance & transparency

### Temporal Fairness Drift (TFD): `domain3.TemporalFairnessDrift.calculate_drift`
3-sigma statistical process control; drift if any point exceeds:
```math
\mathrm{UCL},\ \mathrm{LCL} = \mu \pm 3\sigma .
```
By default $\sigma$ is the sample SD of the whole series, which a drift itself
inflates. `sigma_method="moving_range"` uses $\overline{MR}/1.128$, the standard
individuals-chart estimate (Montgomery), and detects shifts more reliably.

### Audit Traceability Score (ATS): `domain3.AuditTraceabilityScore.calculate_ats`
`calculate_ats(n_traceable, n_total)`:
```math
ATS = \frac{n_{\mathrm{traceable}}}{n_{\mathrm{total}}}
```
with a **Wilson 95% interval** (up to 1.9.5 the interval computed was
Agresti-Coull while labelled Wilson); meets standard at $\ge 0.95$.

### Governance Compliance Index (GCI): `domain3.GovernanceComplianceIndex.calculate_gci`
`calculate_gci(policy_compliance: Dict[str,bool])`:
```math
GCI = \frac{\#\,\mathrm{met}}{\#\,\mathrm{total}} \in [0,1].
```
Returns `compliance_gaps`. 1.0 = full compliance.

---

## Domain 4: Representation & robustness

### Semantic Parity Gap (SPG): `domain4.SemanticParityGap.calculate_spg`
Latent bias as the distance between embedding centroids $c_p, c_m$ of identical
cases that differ only by a protected attribute. Returns **both**:
```math
\mathrm{SPG}_{\mathrm{Euc}} = \lVert c_p - c_m \rVert_2, \qquad
\mathrm{SPG}_{\cos} = 1 - \cos(c_p, c_m).
```
Larger = more identity sensitivity. (State which variant you report.) A centroid
distance is positive even without any true difference, so its bootstrap CI never
contains 0; `p_value_permutation` (1000 label permutations) tests the null of no
difference between the groups.

### Clinical Hallucination Rate (CHR): `domain4.ClinicalHallucinationRate.calculate_chr`
```math
CHR = \frac{1}{|C|}\sum_{c \in C}\mathbb{1}\!\left[\mathrm{support}(c) < \tau\right]
```
over per-claim NLI/entailment support scores (default $\tau=0.5$);
severity-weighted variant via `weights`. Range $[0,1]$; higher is worse. Reported
with a Wilson 95% CI and a threshold $p$-value.

### Instructional Vulnerability Index (IVI): `domain4.InstructionalVulnerabilityIndex.calculate_ivi`
```math
IVI = P\!\left(f(q_{\mathrm{biased}}) \ne f(q_{\mathrm{neutral}})\right)
```
over paired neutral/biased outputs; `ivi_effect` is the directional mean change
for numeric outputs. Range $[0,1]$ (Wilson 95% CI + threshold $p$-value).

### Geographic Representation Index (GRI): `domain4.GeographicRepresentationIndex`
`calculate_gri(locations, western_locations)`, with $L$ the set of locations and
$W$ the Western subset:
```math
GRI = \frac{|L| - |W|}{|L|} \in [0,1] \quad (\text{set-based non-Western variety}).
```
`calculate_geographic_bias(gri_values, error_rates)` correlates GRI with non-Western error rate.
GRI is set-based, so its bootstrap interval (over the mention list) shows how
stable the ratio is to which locations are mentioned; it is not a sampling
interval for the full set of locations.

---

## Domain 5: Technical-supplement fairness

### Intersectional Calibration Error (ICE): `domain5.IntersectionalCalibrationError.calculate_ice`
```math
ECE_i = \sum_b \frac{|S_{ib}|}{|S_i|}\,\lvert \mathrm{acc} - \mathrm{conf} \rvert, \qquad
ICE = \sum_i w_i\,ECE_i, \qquad
dICE = \max_i ECE_i - \min_i ECE_i .
```
For a binary risk model pass the predicted probability as `confidences` and the
observed event as `correct`. Binned ECE is biased upward in small groups, so read
dICE alongside `n_by_group`.

### Weighted Clinical Harm-Adjusted Fairness Gap (wHAFG): `domain5.WeightedClinicalHarmAdjustedFairnessGap.calculate_whafg`
```math
H(g) = \frac{1}{n_g}\sum_i \omega(Y_i)\,L(\hat Y_i, Y_i), \qquad
wHAFG = \max_g H(g) - \min_g H(g).
```
Per-sample, severity-weighted generalization of domain-2 HAFG.

### Lexical Diversity Disparity Index (LDDI): `domain5.LexicalDiversityDisparityIndex.calculate_lddi`
```math
RTTR(g) = \frac{|V_g|}{\sqrt{\mathrm{tokens}_g}}, \qquad LDDI = \max_g RTTR(g) - \min_g RTTR(g),
```
plus `lddi_norm`.

### Recommendation Entropy Gap (REG): `domain5.RecommendationEntropyGap.calculate_reg`
```math
H(T\mid g) = -\sum_t P(t\mid g)\log_2 P(t\mid g), \qquad REG = \max_g - \min_g \ \ (\text{bits}).
```

### Counterfactual Parity Score (CPS): `domain5.CounterfactualParityScore.calculate_cps`
$CPS$ = mean response similarity under a demographic swap; counterfactual unfairness
```math
CFU = 1 - \min_{\mathrm{pair}} CPS \quad (\text{or } 1 - CPS \text{ for a single pair}), \qquad \in [0,1].
```

### Clinical Information Density Ratio (CIDR): `domain5.ClinicalInformationDensityRatio.calculate_cidr`
```math
CID(g) = \overline{\left(\tfrac{\mathrm{concepts}}{\mathrm{tokens}}\right)}\cdot 100, \qquad
CIDR(g) = \frac{CID(g)}{\max_g CID}.
```
`cidr_min` is the most information-sparse group (1.0 = parity).

### Diagnostic Completeness Index (DCI): `domain5.DiagnosticCompletenessIndex.calculate_dci`
```math
DCI(r) = \frac{\lvert D(r) \cap D^\star \rvert}{\lvert D^\star \rvert}
```
against a reference differential set $D^\star$; group means and
$dDCI = \max - \min$; optional severity weights.

### Uncertainty Quantification Gap (UQG): `domain5.UncertaintyQuantificationGap.calculate_uqg`
```math
UD(r) = \frac{\mathrm{hedging\ terms}}{\mathrm{sentences}}, \qquad UQG = \max_g UD - \min_g UD .
```
Overlapping terms count once ("cannot rule out" is not also "rule out"), and
decimals such as "0.04" do not end a sentence (both fixed in 1.10.0).

### Geographic Representation Bias Index (GRBI): `domain5.GeographicRepresentationBiasIndex.calculate_grbi`
```math
GRBI = D_{\mathrm{KL}}(P_{\mathrm{corpus}} \Vert P_{\mathrm{burden}}) = \sum_r p_c(r)\log\frac{p_c(r)}{p_b(r)} \ \ (\text{nats}).
```
Optional HIC over-representation ratio. Directed KL complement to BEMI.

### Healthcare System Stratified Fairness (HSSF): `domain5.HealthcareSystemStratifiedFairness.calculate_hssf`
```math
HSSF = \sum_s P(s)\max_{g,g'}\bigl\lvert E[Y\mid g,s] - E[Y\mid g',s]\bigr\rvert \ \ (\text{within-system}),
```
and `delta_between` $= \mathrm{Var}_s\!\left(E[Y\mid s]\right)$ (between-system).

### Intersectional Shapley Fairness Value (ISFV): `domain5.IntersectionalShapleyFairnessValue.calculate_isfv`
Cooperative-game Shapley attribution of the disparity $v(S) = \max - \min$ of
$E[Y \mid A_S]$ to each protected attribute, plus pairwise interactions
$v(\{i,j\}) - v(\{i\}) - v(\{j\})$. Shapley values sum to the total disparity.
Because $v$ is a range over cells, it grows with the number of cells and is
driven by small cells; set `min_cell` (e.g. 30) and compare with permuted
outcomes before interpreting the attribution.

### Semantic Robustness Parity Index (SRPI): `domain5.SemanticRobustnessParityIndex.calculate_srpi`
```math
SRPI = \frac{\min_g R(g)}{\max_g R(g)}
```
over per-group paraphrase robustness ($1$ = equal).

---

## Geographic

### Burden-Evidence Mismatch Index (BEMI): `geographic.BurdenEvidenceMismatch.calculate_bemi`
**Total-variation distance** between regional evidence and disease-burden shares:
```math
BEMI = \tfrac{1}{2}\sum_r \lvert e_r - b_r \rvert \in [0,1]
```
(0 = evidence tracks burden, 1 = disjoint). `WHO_REGION_IHD_BURDEN` holds WHO
Global Health Estimates 2023 IHD DALY count shares by WHO region;
`WHO_REGION_IHD_BURDEN_RATE` the crude-rate shares. Every region with evidence
must appear in `burden_shares`: a region code missing there raises `ValueError`
(`WHO_REGION_CODES` maps GHO codes such as AFR to AFRO). Leave evidence of unknown
origin out and report its share separately. <0.10 low, <0.25 moderate, ≥0.25 high
(heuristic).

### Geographic Concentration of Coverage (GCC): `geographic.GeographicConcentration.calculate_gcc`
Sample-corrected Gini and normalized Shannon entropy:
```math
G^\star = \frac{R}{R-1}\,G_{\mathrm{raw}}, \qquad
H_{\mathrm{norm}} = \frac{-\sum_r p_r \ln p_r}{\ln R}, \qquad
\mathrm{concentration} = 1 - H_{\mathrm{norm}}.
```
$G^\star$: 0 even, 1 single-region; $H_{\mathrm{norm}}$: 1 even, 0 single.

---

## Appendix: advanced metrics (`appendix.advanced_metrics`, also re-exported)

- **BootstrapConfidenceIntervals** `calculate_bci`: percentile bootstrap CI for any statistic.
- **StatisticalPowerAnalysis** `calculate_sample_size` / `calculate_power`
  (two-sample t-test, statsmodels).
- **BiasConcentrationIndex** `calculate_bci`: $1 - \sum p^2/(\sum p)^2$, whose
  maximum with $n$ groups is $1 - 1/n$; `bci_normalized` divides by that maximum,
  and the verdict uses the normalized value (from 1.10.0).
- **MutualInformationContent (MIC)** `calculate_mic`: **mutual information** between
  demographics and outcomes (NOT the Reshef Maximal Information Coefficient);
  prefer `normalized_mic` for cross-setting comparison.
- **JensenShannonDivergence (JSD)** `calculate_jsd`: JS **divergence**, base 2,
  range $[0,1]$; `jsd_distance` is its square root. (Consistent with
  `appendix.info_theory.AdvancedInfoTheoryMetrics.calculate_jsd`.)
- **WassersteinDistance (WD)** `calculate_wd`: earth-mover distance (scipy)
  between two **samples** of values; to compare two histograms pass the bin
  probabilities and their locations as `support`.
- **NetworkModularity (NM)** `calculate_modularity`: Newman modularity $Q$ over
  greedy (Clauset-Newman-Moore) communities, found and scored with the same
  absolute edge weights; the diagonal of a correlation matrix is not an edge.
- **TransparencyScore (TS)**, **RobustnessCertificationScore (RCS)**.

`appendix.info_theory`, `appendix.network`, and `appendix.reliability` provide
thin aggregator classes over the same statistics; the canonical implementations
live in `advanced_metrics.py`. In `appendix.reliability`, `calculate_power_analysis`
uses the two-sample formula $n = 2(z_{1-\alpha/2} + z_{1-\beta})^2/d^2$ per group
and `calculate_bias_concentration` is the Kakwani-Wagstaff-van Doorslaer
concentration index, weighted by `population_share` for grouped data (both
corrected in 1.10.0).

---

## Choosing metrics

- **Diagnostic systems:** ICE/ECS (calibration/consistency), HER (equity), TFD (drift), CHR (faithfulness).
- **Triage/decision support:** IVI (prompt robustness), SPG/CPS (identity sensitivity), HSSF (system confounding).
- **Evidence/corpus audits:** BEMI, GCC, GRBI, GRI (geographic equity).
- **Governance/regulatory:** GCI, ATS, ICC (reliability), ERI.

## References
1. Rajkomar A, et al. Ensuring Fairness in ML to Advance Health Equity. 2018.
2. Obermeyer Z, et al. Dissecting racial bias in an algorithm. Science 2019.
3. World Health Organization. Global Health Estimates 2023: disease burden by cause,
   age, sex, by country and by region, 2000-2023. Geneva: WHO; 2026.
4. Shrout PE, Fleiss JL. Intraclass correlations: uses in assessing rater reliability.
   Psychol Bull 1979;86:420-428.
5. Cicchetti DV. Guidelines, criteria, and rules of thumb for evaluating normed and
   standardized assessment instruments in psychology. Psychol Assess 1994;6:284-290.
6. Wilson EB. Probable inference, the law of succession, and statistical inference.
   J Am Stat Assoc 1927;22:209-212.
7. Kakwani N, Wagstaff A, van Doorslaer E. Socioeconomic inequalities in health:
   measurement, computation, and statistical inference. J Econometrics 1997;77:87-103.
8. Zhao X, Lynch JG, Chen Q. Reconsidering Baron and Kenny: myths and truths about
   mediation analysis. J Consum Res 2010;37:197-206.
9. Montgomery DC. Introduction to Statistical Quality Control. Hoboken: Wiley.
