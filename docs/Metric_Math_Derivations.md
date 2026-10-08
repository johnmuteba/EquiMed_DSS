> **Rendering note.** This Markdown file is the single authoritative version of the derivations; GitHub renders every equation (via MathJax). The LaTeX and PDF copies distributed up to 1.9.5 were removed in 1.10.0 because they had not been kept in step with the code. The final section, *Uncertainty Quantification*, states which interval or test each metric returns.

# Purpose and Verification Scope

This technical document derives the mathematical definitions used by the
EquiMed-DSS library, version 1.10.0. The formulae below are aligned with
the local implementation in the package source code, especially the
modules `domain1`, `domain2`, `domain3`, `domain4`, `domain5`,
`geographic`, `appendix`, and `statistics`. Where an earlier derivation
document used a more general or theoretical expression, the
implementation-specific expression is reported here as the authoritative
formula.

The library documentation lists 37 named metrics. This document gives 39
derivations by adding two implemented statistical estimands used with
the metric suite: hierarchical variance partitioning and
mediation/proportion mediated. These two estimands are implemented in
`equimed_dss.statistics` and are included because they are used to
interpret fairness and governance results in the EquiMed-DSS framework.

Each metric is numbered in the overview table and in its subsection
title. Some implemented classes return companion quantities, for example
HER with Bias-Gini or CPS with CFU. These companion quantities use the
parent metric number with a letter suffix. Every displayed equation is
written in a numbered LaTeX `equation` or `align` environment so it can
be referenced directly after compilation.

# Notation

Let $i=1,\ldots,n$ index observations, $g \in \mathcal{G}$ index
demographic or intersectional groups, $s \in \mathcal{S}$ index
healthcare-system strata, and $r \in \mathcal{R}$ index geographic
regions. Let $\hat{Y}_i$ denote a model prediction, $Y_i$ the
corresponding reference outcome, $C_i$ a confidence score, and
$Z_i=\mathbb{I}(\hat{Y}_i=Y_i)$ a correctness indicator. For text
outputs, let $R_i$ denote a response, $D(R_i)$ the set of diagnoses
mentioned in the response, and $D^\star$ a reference differential
diagnosis set. Unless otherwise stated, all means are empirical means
over the supplied input arrays.

| No. | Metric | Abbreviation | Canonical implementation |
|:---|:---|:---|:---|
| 1 | Inter-rater reliability | ICC(2,1) | `domain1.icc` |
| 2 | Embedding consistency score | ECS | `domain1.ecs` |
| 3 | Decision flip rate | DFR | `domain1.dfr` |
| 4 | Hierarchical equity ratio and Bias-Gini | HER | `domain2.her` |
| 5 | Harm-adjusted fairness gap | HAFG | `domain2.hafg` |
| 6 | Ethical risk index and safety violation rate | ERI | `domain2.eri` |
| 7 | Intersectional bias score | IBS | `domain2.ibs` |
| 8 | Temporal fairness drift | TFD | `domain3.tfd` |
| 9 | Audit traceability score | ATS | `domain3.ats` |
| 10 | Governance compliance index | GCI | `domain3.gci` |
| 11 | Semantic parity gap | SPG | `domain4.spg` |
| 12 | Clinical hallucination rate | CHR | `domain4.chr` |
| 13 | Instructional vulnerability index | IVI | `domain4.ivi` |
| 14 | Geographic representation index and geographic bias | GRI | `domain4.gri` |
| 15 | Intersectional calibration error | ICE | `domain5.calibration` |
| 16 | Weighted clinical harm-adjusted fairness gap | wHAFG | `domain5.harm` |
| 17 | Counterfactual parity score and counterfactual unfairness | CPS | `domain5.counterfactual` |
| 18 | Semantic robustness parity index | SRPI | `domain5.counterfactual` |
| 19 | Lexical diversity disparity index | LDDI | `domain5.text` |
| 20 | Recommendation entropy gap | REG | `domain5.text` |
| 21 | Clinical information density ratio | CIDR | `domain5.text` |
| 22 | Diagnostic completeness index | DCI | `domain5.text` |
| 23 | Uncertainty quantification gap | UQG | `domain5.text` |
| 24 | Geographic representation bias index | GRBI | `domain5.geographic_bias` |
| 25 | Healthcare system stratified fairness | HSSF | `domain5.system` |
| 26 | Intersectional Shapley fairness value | ISFV | `domain5.shapley` |
| 27 | Burden-evidence mismatch index | BEMI | `geographic.burden_evidence` |
| 28 | Geographic concentration of coverage | GCC | `geographic.concentration` |
| 29 | Bootstrap confidence interval | BCI | `appendix.advanced_metrics` |
| 30 | Statistical power analysis | SPA | `appendix.advanced_metrics` |
| 31 | Bias concentration index | BCI-bias | `appendix.advanced_metrics` |
| 32 | Mutual information content | MIC | `appendix.advanced_metrics` |
| 33 | Jensen-Shannon divergence | JSD | `appendix.advanced_metrics` |
| 34 | Wasserstein distance | WD | `appendix.advanced_metrics` |
| 35 | Network modularity | NM | `appendix.advanced_metrics` |
| 36 | Transparency score | TS | `appendix.advanced_metrics` |
| 37 | Observed perturbation agreement (formerly robustness certification score) | RCS | `appendix.advanced_metrics` |
| 38 | Hierarchical variance partitioning | HLM/VPC | `statistics.hierarchical` |
| 39 | Mediation (product of coefficients) and proportion mediated | PM | `statistics.mediation` |

# Domain 1: Reliability and Robustness

## Metric 1: Inter-rater Reliability, ICC(2,1)

Let $X_{ij}$ be the score assigned to item $i$ by judge $j$, with
$n$ items and $k$ judges. Define the grand mean $\bar{X}_{..}$,
item means $\bar{X}_{i.}$, and judge means $\bar{X}_{.j}$. This
follows the Shrout-Fleiss ICC family and the Bland-Altman agreement
convention. The implementation computes
```math
\begin{aligned}
SS_{\mathrm{items}} &= k \sum_{i=1}^{n}(\bar{X}_{i.}-\bar{X}_{..})^2,\\
SS_{\mathrm{judges}} &= n \sum_{j=1}^{k}(\bar{X}_{.j}-\bar{X}_{..})^2,\\
SS_{\mathrm{error}} &= \sum_{i=1}^{n}\sum_{j=1}^{k}(X_{ij}-\bar{X}_{..})^2
 - SS_{\mathrm{items}} - SS_{\mathrm{judges}}.
\end{aligned}
```
The mean squares are
```math
\begin{aligned}
MS_R &= \frac{SS_{\mathrm{items}}}{n-1},&
MS_C &= \frac{SS_{\mathrm{judges}}}{k-1},&
MS_E &= \frac{SS_{\mathrm{error}}}{(n-1)(k-1)}.
\end{aligned}
```
EquiMed-DSS implements the two-way random-effects, single-measure,
absolute agreement intraclass correlation coefficient as
```math
ICC(2,1)=
\frac{MS_R-MS_E}
{MS_R+(k-1)MS_E+\frac{k}{n}(MS_C-MS_E)}.
```
ICC(2,1) is at most 1 and can be negative (less agreement than chance). At
least 2 items and 2 judges are required. Verdicts use Cicchetti's (1994) bands:
at least 0.75 excellent, 0.60 good, 0.40 fair.
The same class also computes Bland-Altman pairwise agreement. For two
judges $a$ and $b$,
```math
\begin{aligned}
d_i &= X_{ia}-X_{ib},\\
\bar{d} &= \frac{1}{n}\sum_{i=1}^{n}d_i,\\
s_d &= \sqrt{\frac{1}{n-1}\sum_{i=1}^{n}(d_i-\bar{d})^2},\\
LOA_{\mathrm{lower}}, LOA_{\mathrm{upper}} &= \bar{d}\pm 1.96s_d.
\end{aligned}
```


**95% CI (this section).** $ICC(2,1)$ is reported with a percentile bootstrap over items (rows of the rating matrix, every judge kept): $B=1000$ resamples, recompute $ICC(2,1)$, and take the empirical percentiles
```math
CI_{95}=\left[\hat\theta^{\ast}_{(0.025)},\ \hat\theta^{\ast}_{(0.975)}\right].
```

## Metric 2: Embedding Consistency Score

For paired original and perturbed embedding vectors $e_i$ and
$\tilde e_i$, the library calculates cosine distance, not cosine
similarity:
```math
ECS_i = 1-\frac{e_i^\top \tilde e_i}{\left\lVert e_i \right\rVert_2\left\lVert \tilde e_i \right\rVert_2}.
```
If either vector has zero norm, the implementation sets the cosine
similarity to zero, so $ECS_i=1$. The reported summary statistics are
```math
\begin{aligned}
\overline{ECS} &= \frac{1}{n}\sum_{i=1}^{n}ECS_i,\\
SD(ECS) &= \sqrt{\frac{1}{n}\sum_{i=1}^{n}(ECS_i-\overline{ECS})^2},\\
\widetilde{ECS} &= \mathrm{median}(ECS_1,\ldots,ECS_n).
\end{aligned}
```
Lower values indicate greater embedding stability.


**95% CI (this section).** $ECS$ is reported with a percentile bootstrap over embedding pairs: resample with replacement $B=1000$ times, recompute $ECS$, and take the empirical percentiles
```math
CI_{95}=\left[\hat\theta^{\ast}_{(0.025)},\ \hat\theta^{\ast}_{(0.975)}\right].
```

## Metric 3: Decision Flip Rate

Let $A_i$ be the model decision on the original input and $B_i$ the
decision on the paired counterfactual or perturbed input. EquiMed-DSS
defines
```math
DFR = \frac{1}{n}\sum_{i=1}^{n}\mathbb{I}(A_i\ne B_i).
```
If $x=\sum_i\mathbb{I}(A_i\ne B_i)$ and $\hat p=x/n$, the Wilson
interval returned by the library is
```math
\begin{aligned}
d &= 1+\frac{z^2}{n},\\
c &= \frac{\hat p+\frac{z^2}{2n}}{d},\\
h &= \frac{z\sqrt{\frac{\hat p(1-\hat p)}{n}+\frac{z^2}{4n^2}}}{d},\\
CI_{95} &= [\max(0,c-h),\min(1,c+h)],
\end{aligned}
```
with $z=1.96$, following Wilson’s score interval for binomial
proportions. A missing decision (None or NaN) raises an error rather than
counting as a flip (from 1.10.0).


**95% CI (this section).** As a binomial proportion, $DFR$ is reported with a Wilson 95% score interval over decisions (Metric 9 form), with a one-sided score test against a tolerated flip rate.

# Domain 2: Fairness, Equity, and Ethics

## Metric 4: Hierarchical Equity Ratio

Let $q_g$ be a group-specific performance score and $q_0$ the
reference-group score. The implementation calculates
```math
HER_g = \frac{q_g}{q_0},
```
The scores must be finite and non-negative and the reference score positive:
a zero reference makes every ratio undefined and raises an error (up to 1.9.5
every group then received $HER_g=0$). Each group is labelled within or outside
the band $[0.8,1.25]$ of the four-fifths convention; the label describes the
ratio, not whether a difference is fair or clinically important.


**95% CI (this section).** The gap $\max_g HER_g-\min_g HER_g$ is reported with a percentile bootstrap that resamples observations within each group (group sizes fixed, so the reference group is present in every replicate): $B=1000$ resamples, recompute the gap, and take the empirical percentiles
```math
CI_{95}=\left[\hat\theta^{\ast}_{(0.025)},\ \hat\theta^{\ast}_{(0.975)}\right].
```

(needs observation-level group scores; otherwise reported unavailable).

## Metric 4a: Bias-Gini Dispersion

For group scores $q_1,\ldots,q_K$ with mean $\bar q$, the
implemented dispersion index is the standard Gini coefficient:
```math
G_{\mathrm{bias}}=
\frac{\sum_{i=1}^{K}\sum_{j=1}^{K}|q_i-q_j|}
{2K^2\bar q}.
```
If the score list is empty or $\bar q=0$, the function returns zero. Scores
must be finite and non-negative.


**95% CI (this section).** Bias-Gini carries an interval only when per-observation scores are supplied (`group_observations`): a percentile bootstrap that resamples observations within each group and recomputes the index from the group means, $B=1000$. Without them no interval is reported, because the group scores are fixed values, not a sample of groups (versions up to 1.9.5 resampled them anyway).
```math
CI_{95}=\left[\hat\theta^{\ast}_{(0.025)},\ \hat\theta^{\ast}_{(0.975)}\right].
```

## Metric 5: Harm-adjusted Fairness Gap

For two groups, let $FN_g$ and $FP_g$ be false-negative and
false-positive counts. With default costs $c_{FN}=10$ and
$c_{FP}=3$, group harm is
```math
H_g = c_{FN}FN_g+c_{FP}FP_g.
```
EquiMed-DSS reports the absolute gap
```math
\Delta_H = |H_1-H_2|
```
and the normalized harm-adjusted fairness gap
```math
HAFG=\frac{|H_1-H_2|}{\max(H_1,H_2)}.
```
If both group harms are zero, the denominator is zero and the
implemented value is $0$. Because $H_g$ is a total, HAFG reflects group size as
well as error rates when the groups differ in size; use counts per 1,000
patients, or the per-patient wHAFG (Metric 16). When the group sizes $n_g$ are
known (from `group1_n`/`group2_n` or the case lists), the result also reports
the harm per patient $H_g/n_g$ and the corresponding gap `hafg_per_patient`,
with a warning when the sizes differ by more than 10%. Case lists that disagree
with the error counts raise an error (from 1.10.0; earlier versions warned and
then reported an interval for a different estimate).


**95% CI (this section).** $HAFG$ is reported with a percentile bootstrap that resamples cases within each group (two-sample bootstrap, group sizes fixed): $B=1000$ resamples, recompute $HAFG$, and take the empirical percentiles
```math
CI_{95}=\left[\hat\theta^{\ast}_{(0.025)},\ \hat\theta^{\ast}_{(0.975)}\right].
```

(needs observation-level cases; otherwise reported unavailable).

## Metric 6: Ethical Risk Index

Let $v=1,\ldots,V$ index detected ethical or safety violations and let
$s_v$ be their severity scores. For $N$ total model outputs,
```math
ERI = \frac{\sum_{v=1}^{V}s_v}{N}.
```
Severities must be finite and non-negative, $N$ must be positive, and there
can be at most one violation per output (from 1.10.0 these raise errors; a
zero $N$ used to return zero even when violations were given).


**95% CI (this section).** $ERI$ is reported with a percentile bootstrap over the per-output severity vector (severity for violations, 0 otherwise): resample with replacement $B=1000$ times, recompute $ERI$, and take the empirical percentiles
```math
CI_{95}=\left[\hat\theta^{\ast}_{(0.025)},\ \hat\theta^{\ast}_{(0.975)}\right].
```

## Metric 6a: Safety Violation Rate

In the same implementation as ERI, the safety violation rate is
```math
SVR = 1000\frac{V}{N},
```
that is, violations per 1000 model outputs.


**95% CI (this section).** As a binomial proportion, $SVR$ is reported with a Wilson 95% score interval over outputs (Metric 9 form).

## Metric 7: Intersectional Bias Score

Let $u_g\in\mathbb{R}^d$ be a vector of metrics for subgroup $g$.
The implementation calculates Euclidean distances
```math
d(g,h)=\left\lVert u_g-u_h \right\rVert_2
```
and converts them to similarities by inverse distance:
```math
S(g,h)=\frac{1}{1+d(g,h)}.
```
The outlier subgroup is the subgroup with the largest mean distance over
the full distance row, including the zero self-distance used by the
implementation:
```math
g^\star=\arg\max_g \frac{1}{K}\sum_{h\in\mathcal{G}} d(g,h).
```
The same class also computes simplified eta-squared-style interaction
effects. For a categorical attribute $A$, with grand mean $\bar Y$
and group means $\bar Y_a$, the main-effect proxy is
```math
\eta_A^2 =
\frac{\sum_a n_a(\bar Y_a-\bar Y)^2}
{\sum_i(Y_i-\bar Y)^2}.
```
For the race-by-gender interaction, the implementation subtracts the
race and gender main-effect proxies from the combined race-gender proxy.
With unbalanced groups the main effects overlap, so this is a descriptive
proxy (it can be negative), not a model-based interaction test; the `formula`
argument is not parsed, and passing one raises a warning.


**95% CI (this section).** The mean similarity is reported with a percentile bootstrap over the metric dimensions (columns of the subgroup vectors): $B=1000$ resamples. It shows how much the similarity depends on which metrics were chosen; it is not a sampling interval for patients. Similarity depends on the scale of each metric, so standardise metrics on different scales first.
```math
CI_{95}=\left[\hat\theta^{\ast}_{(0.025)},\ \hat\theta^{\ast}_{(0.975)}\right].
```

# Domain 3: Governance and Transparency

## Metric 8: Temporal Fairness Drift

For a time series of fairness metric values $m_1,\ldots,m_T$,
EquiMed-DSS computes
```math
\begin{aligned}
\bar m &= \frac{1}{T}\sum_{t=1}^{T}m_t,\\
s_m &= \sqrt{\frac{1}{T-1}\sum_{t=1}^{T}(m_t-\bar m)^2}.
\end{aligned}
```
The three-sigma control limits are
```math
\begin{aligned}
UCL &= \bar m+3s_m,\\
LCL &= \bar m-3s_m.
\end{aligned}
```
A drift point is flagged when $m_t>UCL$ or $m_t<LCL$, following the
Shewhart-style control-chart logic used in statistical process control.
With `sigma_method="moving_range"` (from 1.10.0) the sigma estimate is the
individuals-chart estimate
```math
\hat\sigma=\frac{\overline{MR}}{d_2},\qquad
\overline{MR}=\frac{1}{T-1}\sum_{t=2}^{T}|m_t-m_{t-1}|,\qquad d_2=1.128,
```
which a sustained shift does not inflate, unlike $s_m$. With `baseline_n`
$=T_0$ (from 1.10.0) the centre and limits are estimated from $m_1,\ldots,m_{T_0}$
only and applied prospectively to $m_{T_0+1},\ldots,m_T$, which is how a control
chart should monitor drift. Without it the limits come from the whole series
being monitored, so a shift can move the centre or widen the limits and go
undetected.


**95% CI (this section).** The centre (the mean of the points behind the limits) is reported with a moving-block bootstrap (Künsch 1989): blocks of $L=\mathrm{round}(T^{1/3})$ consecutive points are resampled, which keeps short-range serial dependence that an ordinary bootstrap ignores; $B=1000$ resamples and the empirical percentiles. (Up to 1.9.5 time points were resampled independently.)
```math
CI_{95}=\left[\hat\theta^{\ast}_{(0.025)},\ \hat\theta^{\ast}_{(0.975)}\right].
```

## Metric 9: Audit Traceability Score

Let $x$ be the number of traceable decisions among $n$ audited
decisions. The score is
```math
ATS=\frac{x}{n}.
```
The implementation returns the Wilson score interval (Wilson 1927), with
$\hat p = x/n$ and $z = z_{0.975}$:
```math
\begin{aligned}
c &= \frac{\hat p + z^2/(2n)}{1 + z^2/n},\\
h &= \frac{z}{1 + z^2/n}\sqrt{\frac{\hat p(1-\hat p)}{n} + \frac{z^2}{4n^2}},\\
CI_{95} &= [\,c - h,\; c + h\,].
\end{aligned}
```
Versions up to 1.9.5 labelled the interval "Wilson score" but computed the
Agresti-Coull interval, $\tilde p \pm z\sqrt{\tilde p(1-\tilde p)/(n+z^2)}$ with
$\tilde p = (x+z^2/2)/(n+z^2)$, which has the same centre and is slightly wider.
The result reports whether $ATS\ge 0.95$ (a target, not a regulatory
judgement). With no audited decisions ($n=0$) ATS is undefined and an error is
raised (up to 1.9.5 it was reported as 0).

## Metric 10: Governance Compliance Index

Let $M$ be the number of mandated policies and $E$ the number
evaluated as enforced. The implementation defines
```math
GCI=\frac{E}{M}.
```
GCI is the share of the listed checks that are met; it does not establish
regulatory compliance. With no checks it is undefined and an error is raised
(up to 1.9.5 the value was zero).


**95% CI (this section).** As a proportion, $GCI$ is reported with a Wilson 95% score interval over the listed checks (Metric 9 form). The interval has a sampling meaning only if the checks are a sample from a larger defined set; for a complete, fixed list, report GCI itself.

# Domain 4: Representation and Robustness

## Metric 11: Semantic Parity Gap

Let $P$ be the embeddings for the $n_p$ privileged-group prompts and $M$ the
embeddings for the $n_m$ matched marginalized-group prompts. Their centroids are
```math
\bar p = \frac{1}{n_p}\sum_{i=1}^{n_p} p_i,
\qquad
\bar m = \frac{1}{n_m}\sum_{j=1}^{n_m} m_j.
```
The Euclidean SPG reported by the implementation is
```math
SPG_{\mathrm{Euc}}=\left\lVert \bar p-\bar m \right\rVert_2.
```
The cosine variant is
```math
SPG_{\mathrm{cos}}=1-\frac{\bar p^\top \bar m}{\left\lVert \bar p \right\rVert_2\left\lVert \bar m \right\rVert_2},
```
with value zero if the denominator is zero.

A centroid distance is positive even when both groups come from the same
distribution, so its bootstrap interval never contains zero. From 1.10.0 the
result also carries a permutation $p$-value: with $D_b$ the distance after the
$b$-th random reassignment of the pooled rows to groups of sizes $n_p$ and $n_m$,
```math
p=\frac{1+\#\{b: D_b\ge SPG_{\mathrm{Euc}}\}}{1+B},\qquad B=1000.
```
When row $i$ of both arrays is the same clinical case with only the protected
attribute changed, use `paired=True`: the interval then resamples cases (each
pair kept together) and the permutation test flips the sign of randomly chosen
within-pair differences $p_i-m_i$, so the pairing is respected. A shift in the
model's representation does not by itself show clinically harmful bias; relate
it to differences in the outputs (for example DFR or CPS on the same pairs).


**95% CI (this section).** $SPG$ is reported with a percentile bootstrap over embedding rows within each group (two-sample bootstrap) or, with `paired=True`, over cases: $B=1000$ resamples, recompute $SPG$, and take the empirical percentiles
```math
CI_{95}=\left[\hat\theta^{\ast}_{(0.025)},\ \hat\theta^{\ast}_{(0.975)}\right].
```

## Metric 12: Clinical Hallucination Rate

Let $c=1,\ldots,C$ index extracted clinical claims and
$S(c,K)\in[0,1]$ be a precomputed support score against retrieved
context $K$. With entailment threshold $\tau$, the implemented
hallucination rate is
```math
CHR=\frac{1}{C}\sum_{c=1}^{C}\mathbb{I}\{S(c,K)<\tau\}.
```
With optional severity weights $w_c$,
```math
CHR_w=
\frac{\sum_{c=1}^{C}w_c\mathbb{I}\{S(c,K)<\tau\}}
{\sum_{c=1}^{C}w_c}.
```
If no weights are supplied, the implementation sets $CHR_w=CHR$. A missing
(NaN) or out-of-range support score raises an error (from 1.10.0; a NaN used
to count as a supported claim).


**95% CI (this section).** As a binomial proportion, $CHR$ is reported with a Wilson 95% score interval over claims (Metric 9 form), with a one-sided score test against a tolerated rate.

## Metric 13: Instructional Vulnerability Index

Let $A_i$ be the neutral-output decision and $B_i$ the paired
biased-instruction decision for the same case. The flip component is
```math
IVI=\frac{1}{n}\sum_{i=1}^{n}\mathbb{I}(A_i\ne B_i).
```
If the outputs can be coerced to numeric values, the implementation also
returns the directional effect
```math
IVI_{\mathrm{effect}}=\frac{1}{n}\sum_{i=1}^{n}B_i-\frac{1}{n}\sum_{i=1}^{n}A_i.
```
The paired-counterfactual framing is conceptually related to
counterfactual fairness, although IVI targets prompt framing rather than
protected-attribute interventions.


**95% CI (this section).** As a binomial proportion, $IVI$ is reported with a Wilson 95% score interval over case pairs (Metric 9 form), with a one-sided score test against a tolerated rate.

## Metric 14: Geographic Representation Index

Let $L$ be the set of unique locations represented in a corpus and let
$W\subseteq L$ be those counted as Western or high-income. The
implemented index is set-based:
```math
GRI=\frac{|L|-|W|}{|L|}.
```
Duplicates do not change the score: GRI measures the VARIETY of locations, so
one study from each of many non-Western locations can outweigh thousands from a
single Western location. The result also reports the VOLUME view, the share of
mentions (duplicates included) that are non-Western
(`non_western_mention_share`); report both.


**95% CI (this section).** $GRI$ is reported with a percentile bootstrap over location mentions: $B=1000$ resamples, recompute $GRI$, and take the empirical percentiles. A resample can only lose locations, never add unseen ones, so this interval describes how stable the ratio is to which locations happen to be mentioned, not uncertainty about the full set of locations.
```math
CI_{95}=\left[\hat\theta^{\ast}_{(0.025)},\ \hat\theta^{\ast}_{(0.975)}\right].
```

## Metric 14a: Geographic Bias Correlation

For paired values $(x_i,y_i)$, where $x_i$ is a per-query GRI value
and $y_i$ is the corresponding non-Western error rate, the library
returns either Pearson or Spearman correlation. For Pearson,
```math
GB =
\frac{\sum_i(x_i-\bar x)(y_i-\bar y)}
{\sqrt{\sum_i(x_i-\bar x)^2}\sqrt{\sum_i(y_i-\bar y)^2}}.
```


**95% CI (this section).** No interval is computed: the method returns the correlation with its SciPy $p$-value (Pearson or Spearman).

# Domain 5: Technical-supplement Fairness

## Metric 15: Intersectional Calibration Error

For group $g$ and bin $b$, let $S_{gb}$ be samples in
intersectional group $g$ whose predicted probability $C_i$ falls in bin $b$, and
let $Z_i\in\{0,1\}$ be the outcome that the probability predicts: for a risk
model the observed event, for a classifier's confidence whether its label was
correct (any other value raises an error from 1.10.0). The
implementation uses equal-width bins on $[0,1]$. This extends expected
calibration error as used in neural-network calibration studies. Let
```math
\begin{aligned}
acc(S_{gb}) &= \frac{1}{|S_{gb}|}\sum_{i\in S_{gb}}Z_i,\\
conf(S_{gb}) &= \frac{1}{|S_{gb}|}\sum_{i\in S_{gb}}C_i.
\end{aligned}
```
The group-specific calibration error is
```math
ECE_g = \sum_b \frac{|S_{gb}|}{|S_g|}
\left|acc(S_{gb})-conf(S_{gb})\right|.
```
The intersectional calibration error is the population-weighted average
```math
ICE = \sum_g \frac{|S_g|}{\sum_h |S_h|}ECE_g.
```
The maximum calibration gap returned as `delta_ice` is
```math
\Delta ICE = \max_g ECE_g-\min_g ECE_g.
```
Binned ECE is biased upward in small samples (noise alone separates the mean
prediction and the event rate within a bin), so read $\Delta ICE$ alongside the
group sizes the result reports (`n_by_group`).


**95% CI (this section).** $ICE$ is reported with a percentile bootstrap that resamples (probability, outcome) pairs within each group, so the group sizes and the ICE weights stay fixed: $B=1000$ resamples, recompute $ICE$, and take the empirical percentiles
```math
CI_{95}=\left[\hat\theta^{\ast}_{(0.025)},\ \hat\theta^{\ast}_{(0.975)}\right].
```

## Metric 16: Weighted Clinical Harm-adjusted Fairness Gap

Let $w_i$ be a clinical severity weight and $\ell_i=L(\hat Y_i,Y_i)$
be a per-sample loss. For group $g$,
```math
H(g)=\frac{1}{n_g}\sum_{i:G_i=g}w_i\ell_i.
```
The implemented maximum gap is
```math
wHAFG_{\max}=\max_g H(g)-\min_g H(g).
```


**95% CI (this section).** $wHAFG$ is reported with a percentile bootstrap that resamples samples within each group (group sizes fixed): $B=1000$ resamples, recompute $wHAFG$, and take the empirical percentiles
```math
CI_{95}=\left[\hat\theta^{\ast}_{(0.025)},\ \hat\theta^{\ast}_{(0.975)}\right].
```

## Metric 17: Counterfactual Parity Score

Let $s_i\in[0,1]$ be a precomputed semantic similarity between the
original response and the response under a demographic swap (values outside
$[0,1]$ raise an error; rescale a cosine similarity $c$ as $(1+c)/2$). For a
single pair type,
```math
CPS=\frac{1}{n}\sum_{i=1}^{n}s_i.
```
The demographic-swap construction is grounded in the counterfactual
fairness intuition that decisions should be stable under
protected-attribute changes when clinically relevant facts are unchanged
. For multiple swap-pair labels $p$, the library calculates
```math
CPS_p=\frac{1}{n_p}\sum_{i\in p}s_i,\qquad
CPS=\frac{1}{\sum_p n_p}\sum_p\sum_{i\in p}s_i.
```


**95% CI (this section).** $CPS$ is reported with a percentile bootstrap that resamples cases within each swap pair: $B=1000$ resamples, recompute $CPS$, and take the empirical percentiles
```math
CI_{95}=\left[\hat\theta^{\ast}_{(0.025)},\ \hat\theta^{\ast}_{(0.975)}\right].
```

## Metric 17a: Counterfactual Unfairness

The implementation returns a complement to CPS. For a single pair,
```math
CFU=1-CPS.
```
For multiple pairs,
```math
CFU=1-\min_p CPS_p.
```


**95% CI (this section).** $CFU$ is reported with the same stratified bootstrap as CPS (`cfu_ci_lower`, `cfu_ci_upper`); with similarities on $[0,1]$, $CFU\in[0,1]$. (Versions up to 1.9.5 documented a Wilson interval here that the code did not compute.)

## Metric 18: Semantic Robustness Parity Index

Let $R_i$ be a per-query robustness score, usually the mean similarity
among responses to semantically equivalent paraphrases. For group $g$,
```math
R(g)=\frac{1}{n_g}\sum_{i:G_i=g}R_i.
```
The implemented parity ratio is
```math
SRPI=\frac{\min_g R(g)}{\max_g R(g)}.
```
Scores must lie in $[0,1]$. If every group has zero robustness, SRPI is
undefined (NaN; up to 1.9.5 it was 0, which read as maximal disparity). SRPI
describes parity only (equal but low robustness also gives 1), so the result
also reports the smallest and largest group robustness.


**95% CI (this section).** $SRPI$ is reported with a percentile bootstrap that resamples queries within each group: $B=1000$ resamples, recompute $SRPI$, and take the empirical percentiles
```math
CI_{95}=\left[\hat\theta^{\ast}_{(0.025)},\ \hat\theta^{\ast}_{(0.975)}\right].
```

## Metric 19: Lexical Diversity Disparity Index

Let $T_g$ be the multiset of tokens pooled across all responses in
group $g$ and $V_g$ its vocabulary. The implemented root type-token
ratio is
```math
RTTR(g)=\frac{|V_g|}{\sqrt{|T_g|}}.
```
The disparity index is
```math
LDDI=\max_g RTTR(g)-\min_g RTTR(g).
```
With $RTTR_{\mathrm{all}}$ computed after pooling all groups,
```math
LDDI_{\mathrm{norm}}=\frac{LDDI}{RTTR_{\mathrm{all}}}.
```


**95% CI (this section).** $LDDI$ is reported with a percentile bootstrap that resamples responses within each group (group sizes fixed): $B=1000$ resamples, recompute $LDDI$, and take the empirical percentiles
```math
CI_{95}=\left[\hat\theta^{\ast}_{(0.025)},\ \hat\theta^{\ast}_{(0.975)}\right].
```

## Metric 20: Recommendation Entropy Gap

Let $P_g(t)$ be the empirical distribution of recommendation labels
$t$ in group $g$. The group recommendation entropy is based on
Shannon entropy:
```math
H(T\mid g)=-\sum_t P_g(t)\log_2 P_g(t).
```
The implemented gap is
```math
REG=\max_g H(T\mid g)-\min_g H(T\mid g).
```
The implementation also returns
```math
REG_{KL}=\max_g \sum_t P_g(t)\log_2\frac{P_g(t)}{P(t)},
```
where $P(t)$ is the marginal recommendation distribution.


**95% CI (this section).** $REG$ is reported with a percentile bootstrap that resamples recommendations within each group (group sizes fixed): $B=1000$ resamples, recompute $REG$, and take the empirical percentiles
```math
CI_{95}=\left[\hat\theta^{\ast}_{(0.025)},\ \hat\theta^{\ast}_{(0.975)}\right].
```

## Metric 21: Clinical Information Density Ratio

For response $i$, let $a_i$ be the number of extracted clinical
concepts and $b_i$ the number of tokens. The response-level clinical
information density is
```math
CID_i=100\frac{a_i}{b_i}.
```
For group $g$,
```math
CID(g)=\frac{1}{n_g}\sum_{i:G_i=g}CID_i.
```
The group ratio is
```math
CIDR(g)=\frac{CID(g)}{\max_h CID(h)},
```
and the reported parity summary is
```math
CIDR_{\min}=\min_g CIDR(g).
```
If the maximum group density is zero, all implemented ratios are set to
zero.


**95% CI (this section).** $CIDR$ is reported with a percentile bootstrap that resamples responses within each group (group sizes fixed): $B=1000$ resamples, recompute $CIDR$, and take the empirical percentiles
```math
CI_{95}=\left[\hat\theta^{\ast}_{(0.025)},\ \hat\theta^{\ast}_{(0.975)}\right].
```

## Metric 22: Diagnostic Completeness Index

Let $D^\star_i$ be the reference differential diagnosis set for the case
behind response $i$: one shared list, or, when the cases differ, a
case-specific list per response (`references_by_group`). For response $i$,
```math
DCI_i=\frac{|D(R_i)\cap D^\star_i|}{|D^\star_i|}.
```
A group with no responses raises an error (it used to score 0).
The group mean is
```math
DCI(g)=\frac{1}{n_g}\sum_{i:G_i=g}DCI_i,
```
and the implemented disparity is
```math
\Delta DCI=\max_g DCI(g)-\min_g DCI(g).
```
When diagnosis weights $w_d$ are supplied, the weighted response score
is
```math
wDCI_i=
\frac{\sum_{d\in D(R_i)\cap D^\star}w_d}{\sum_{d\in D^\star}w_d}.
```


**95% CI (this section).** $DCI$ is reported with a percentile bootstrap that resamples responses within each group (group sizes fixed): $B=1000$ resamples, recompute $DCI$, and take the empirical percentiles
```math
CI_{95}=\left[\hat\theta^{\ast}_{(0.025)},\ \hat\theta^{\ast}_{(0.975)}\right].
```

## Metric 23: Uncertainty Quantification Gap

Let $h_i$ be the number of hedging terms in response $R_i$ and
$q_i$ the number of sentences. The uncertainty density is
```math
UD_i=\frac{h_i}{q_i}.
```
Hedging terms are matched as whole words or phrases, longest first and without
overlap, so "cannot rule out" counts once and not also as "rule out"; a sentence
ends at '.', '!' or '?' followed by white space or the end of the text, so a
decimal such as "0.04" does not end a sentence (both from 1.10.0).
If no sentence is detected, $UD_i=0$. For group $g$,
```math
UD(g)=\frac{1}{n_g}\sum_{i:G_i=g}UD_i.
```
The implemented uncertainty quantification gap is
```math
UQG=\max_g UD(g)-\min_g UD(g).
```
UQG counts hedging words; it measures how often hedging language is used, not
whether stated uncertainty is calibrated, and the default lexicon has not been
validated against annotated clinical language.


**95% CI (this section).** $UQG$ is reported with a percentile bootstrap that resamples responses within each group (group sizes fixed): $B=1000$ resamples, recompute $UQG$, and take the empirical percentiles
```math
CI_{95}=\left[\hat\theta^{\ast}_{(0.025)},\ \hat\theta^{\ast}_{(0.975)}\right].
```

## Metric 24: Geographic Representation Bias Index

Let $p_c(r)$ be the normalized corpus evidence share in region $r$
and $p_b(r)$ the normalized burden share. EquiMed-DSS implements
directed KL divergence in nats:
```math
GRBI=D_{\mathrm{KL}}(P_c\Vert P_b)=\sum_{r:p_c(r)>0}p_c(r)\log\frac{p_c(r)}{p_b(r)}.
```
Counts and shares must be finite and non-negative. If $p_c(r)>0$ and
$p_b(r)=0$, the implementation raises an error because KL is undefined. With optional high-income regions
$\mathcal{H}$, the high-income overrepresentation ratio is
```math
HIC_{\mathrm{ratio}}=
\frac{\sum_{r\in\mathcal{H}}p_c(r)}
{\sum_{r\in\mathcal{H}}p_b(r)}.
```


**95% CI (this section).** $GRBI$ is reported with a percentile bootstrap over evidence records: resample with replacement $B=1000$ times, recompute $GRBI$, and take the empirical percentiles
```math
CI_{95}=\left[\hat\theta^{\ast}_{(0.025)},\ \hat\theta^{\ast}_{(0.975)}\right].
```

(needs record-level input; otherwise reported unavailable).

## Metric 25: Healthcare System Stratified Fairness

For system stratum $s$, let
```math
\Delta_s=\max_g \mathbb{E}[Y\mid G=g,S=s]-\min_g \mathbb{E}[Y\mid G=g,S=s].
```
Only systems with at least two groups (each with at least `min_group_n`
observations) have a gap; the others are listed in
`systems_without_comparison` and excluded (up to 1.9.5 they counted as
$\Delta_s=0$, which made sparse systems look fair). With $w_s$ the number of
observations compared in system $s$ and $\mathcal{S}$ the eligible systems,
```math
HSSF=\sum_{s\in\mathcal{S}}\frac{w_s}{\sum_{u\in\mathcal{S}}w_u}\Delta_s .
```
The implementation returns $\Delta_{\mathrm{within}}=HSSF$. Separately, it
describes how mean outcomes differ BETWEEN systems, in outcome units: the range
$\max_s\mathbb{E}[Y\mid S=s]-\min_s\mathbb{E}[Y\mid S=s]$ and the
population-weighted standard deviation $\sqrt{\Delta_{\mathrm{between}}}$, where
```math
\Delta_{\mathrm{between}}=
\sum_s P(S=s)\left(\mathbb{E}[Y\mid S=s]-\sum_{u}P(S=u)\mathbb{E}[Y\mid S=u]\right)^2
```
is kept for compatibility. The within-system gap and the between-system spread
are separate descriptors, not additive components of a total disparity; the
variance (squared units) must not be compared with HSSF directly.


**95% CI (this section).** $HSSF$ is reported with a percentile bootstrap that resamples observations within each system-by-group cell: $B=1000$ resamples, recompute $HSSF$, and take the empirical percentiles
```math
CI_{95}=\left[\hat\theta^{\ast}_{(0.025)},\ \hat\theta^{\ast}_{(0.975)}\right].
```

## Metric 26: Intersectional Shapley Fairness Value

Let $\mathcal{A}=\{A_1,\ldots,A_m\}$ be protected attributes. The
attribution formula follows Shapley’s cooperative-game value. For a
subset $S\subseteq\mathcal{A}$, the implemented characteristic
function is
```math
v(S)=
\max_{a\in\mathrm{dom}(S)}\mathbb{E}[Y\mid A_S=a]
-
\min_{a\in\mathrm{dom}(S)}\mathbb{E}[Y\mid A_S=a],
```
with $v(\varnothing)=0$. Cells below the configured minimum cell size
are ignored. The Shapley attribution for attribute $A_j$ is
```math
\phi_j=
\sum_{S\subseteq\mathcal{A}\setminus\{A_j\}}
\frac{|S|!(m-|S|-1)!}{m!}
\left[v(S\cup\{A_j\})-v(S)\right].
```
The total disparity is $v(\mathcal{A})$. Pairwise interactions are
```math
I(A_j,A_k)=v(\{A_j,A_k\})-v(\{A_j\})-v(\{A_k\}).
```
Because $v$ is a range of cell means, it grows with the number of cells and is
driven by small cells even when outcomes do not differ, so an attribute with
more categories tends to receive a larger share; set `min_cell` (for example
30). The interaction compares ranges and has no direction, so a positive value
is not by itself an intersectional penalty. From 1.10.0 the result includes a
permutation check of $v(\mathcal{A})$ against shuffled outcomes ($P=200$;
`p_value_permutation` and `total_disparity_null_mean`, the range expected from
noise alone).


**95% CI (this section).** The total disparity $v(\mathcal{A})$ and each attribution $\phi_j$ (`shapley_ci`) are reported with a percentile bootstrap that resamples within the full cross-classified cells: $B=500$ resamples (the documentation said 1000 up to 1.9.5; the code has always used 500), recompute, and take the empirical percentiles
```math
CI_{95}=\left[\hat\theta^{\ast}_{(0.025)},\ \hat\theta^{\ast}_{(0.975)}\right].
```

# Geographic Module

## Metric 27: Burden-evidence Mismatch Index

Let $e(r)$ be the normalized evidence share and $b(r)$ the
normalized burden share over the union of supplied regions. EquiMed-DSS
defines BEMI as total variation distance:
```math
BEMI=\frac{1}{2}\sum_{r\in\mathcal{R}}|e(r)-b(r)|.
```
The per-region mismatch and ratio returned by the implementation are
```math
\begin{aligned}
M(r)&=e(r)-b(r),\\
\rho(r)&=\frac{e(r)}{b(r)}.
\end{aligned}
```
The ratio $\rho(r)$ is finite only when $b(r)>0$; the implementation
records a missing value when the burden share is zero. The most
underserved region is the region with the minimum $M(r)$. Every region with
evidence must appear in the burden mapping: a region code missing there raises
an error (mismatched codes such as AFR against AFRO would otherwise be scored as
evidence outside every burden region), and inputs must be finite and
non-negative.


**95% CI (this section).** $BEMI$ is reported with a percentile bootstrap over geolocated evidence records: resample with replacement $B=1000$ times, recompute $BEMI$, and take the empirical percentiles
```math
CI_{95}=\left[\hat\theta^{\ast}_{(0.025)},\ \hat\theta^{\ast}_{(0.975)}\right].
```

(needs record-level input; otherwise reported unavailable).

## Metric 28: Geographic Concentration of Coverage

Let $x_r\ge 0$ be regional evidence counts for $R$ regions and
$p_r=x_r/\sum_u x_u$. The raw categorical Gini is
```math
G_{\mathrm{raw}}=
\frac{\sum_{r=1}^{R}\sum_{u=1}^{R}|x_r-x_u|}
{2R\sum_{r=1}^{R}x_r}.
```
Because the maximum raw Gini for $R$ categories is $(R-1)/R$, the
implementation reports the normalized value (a rescaling so that the maximum is
1, not a correction for sampling bias; earlier documentation called it
"sample-corrected")
```math
G^\star=\frac{R}{R-1}G_{\mathrm{raw}}.
```
The normalized Shannon entropy follows Shannon’s entropy definition:
```math
H_{\mathrm{norm}}=
-\frac{\sum_{r:p_r>0}p_r\log p_r}{\log R},
```
and the concentration score is
```math
C_{\mathrm{geo}}=1-H_{\mathrm{norm}}.
```


**95% CI (this section).** $GCC$ is reported with a percentile bootstrap over geolocated evidence records: resample with replacement $B=1000$ times, recompute $GCC$, and take the empirical percentiles
```math
CI_{95}=\left[\hat\theta^{\ast}_{(0.025)},\ \hat\theta^{\ast}_{(0.975)}\right].
```

(needs record-level input; otherwise reported unavailable).

# Appendix Metrics

## Metric 29: Bootstrap Confidence Interval

Given data $x_1,\ldots,x_n$ and a statistic $T(\cdot)$, EquiMed-DSS
draws $B$ bootstrap resamples $x^{\ast b}$ of size $n$ with
replacement and computes
```math
\theta_b^\ast=T(x^{\ast b}),\qquad b=1,\ldots,B.
```
For significance level $\alpha$, the implemented percentile interval
is
```math
CI_{1-\alpha}=
\left[
Q_{\alpha/2}(\theta_1^\ast,\ldots,\theta_B^\ast),
Q_{1-\alpha/2}(\theta_1^\ast,\ldots,\theta_B^\ast)
\right].
```
The observed statistic is $T(x_1,\ldots,x_n)$. The percentile
construction follows the nonparametric bootstrap framework of Efron and
Tibshirani.


**95% CI (this section).** this metric IS the percentile bootstrap interval.

## Metric 30: Statistical Power Analysis

The canonical implementation delegates two-sample $t$-test power and
sample-size calculations to
`statsmodels.stats.power.tt_ind_solve_power`. The underlying planning
target is Cohen’s standardized effect
```math
d=\frac{\mu_1-\mu_2}{\sigma}.
```
The solver returns the per-group sample size $n$ satisfying
```math
\mathrm{Power}=
P\left(\mathrm{reject}\ H_0:\mu_1=\mu_2\mid d,\alpha,n\right)
```
for the requested alternative. The returned per-group sample size is
$\lceil n\rceil$, and the returned total is $2\lceil n\rceil$ (two equal
groups; up to 1.9.5 it was $\lceil 2n\rceil$, which could be one short). The standardized effect-size
scale follows Cohen’s two-sample convention.


**95% CI (this section).** an analytic design quantity; no sampling CI of its own.

## Metric 31: Bias Concentration Index

For nonnegative group bias proportions $p_1,\ldots,p_K$, the
implementation defines
```math
BCI_{\mathrm{bias}}=
1-\frac{\sum_{g=1}^{K}p_g^2}{\left(\sum_{g=1}^{K}p_g\right)^2}.
```
If the vector is empty or sums to zero, the value is zero. For a
normalized nonnegative vector, the finite-group upper bound is
$1-1/K$, attained by an even distribution. Larger values indicate more
distributed bias; smaller values indicate concentration in fewer groups. The
result also reports
```math
BCI^{\mathrm{norm}}_{\mathrm{bias}}=\frac{BCI_{\mathrm{bias}}}{1-1/K}\in[0,1],
```
and from 1.10.0 the verdict thresholds (0.3, 0.7) apply to this normalized value,
so that an even distribution over few groups is not labelled concentrated. The
index describes how bias is distributed, not how large it is: an even spread of
a large bias scores as evenly distributed. Negative proportions raise an error.


**95% CI (this section).** No interval is reported: the inputs are one fixed value per group, not a sample (versions up to 1.9.5 resampled them as if they were).

## Metric 32: Mutual Information Content

Let $D$ be a demographic categorical variable and $O$ a DISCRETE outcome
(labels, decisions or binned scores); non-integer numeric outcomes raise an
error, because when every value is unique the mutual information approaches
$H(D)$ simply because each value identifies a row. Categories may be strings.
The raw mutual information implemented via `mutual_info_score` uses
Shannon mutual information:
```math
MIC=I(D;O)=\sum_{d,o}p(d,o)\log\frac{p(d,o)}{p(d)p(o)}.
```
The implementation also reports a normalized value using demographic
entropy:
```math
MIC_{\mathrm{norm}}=\frac{I(D;O)}{H(D)},\qquad
H(D)=-\sum_d p(d)\log p(d).
```
If $H(D)=0$, the normalized value is zero. Plug-in mutual information is
biased upward in small samples, so the result reports a permutation null: the
mean of $I(D;O^{\pi})$ over $P=200$ shuffles of the outcomes (`mic_null_mean`)
and the add-one $p$-value (`p_value_permutation`). An association can reflect
clinical need or case mix; adjust for clinically relevant factors before
interpreting it as bias.


**95% CI (this section).** $MIC$ is reported with a percentile bootstrap over paired (demographic, outcome) observations: resample with replacement $B=1000$ times, recompute $MIC$, and take the empirical percentiles
```math
CI_{95}=\left[\hat\theta^{\ast}_{(0.025)},\ \hat\theta^{\ast}_{(0.975)}\right].
```

## Metric 33: Jensen-Shannon Divergence

For two nonnegative vectors normalized to probability distributions
$p$ and $q$, let
```math
m=\frac{p+q}{2}.
```
The implementation returns the base-2 Jensen-Shannon divergence:
```math
JSD(p,q)=
\frac{1}{2}D_{\mathrm{KL}}^{(2)}(p\Vert m)+\frac{1}{2}D_{\mathrm{KL}}^{(2)}(q\Vert m),
```
where $D_{\mathrm{KL}}^{(2)}$ uses $\log_2$. The SciPy function returns
the distance $\sqrt{JSD}$, so the implementation squares that value.
Both `jsd` and `jsd_distance` are reported.


**95% CI (this section).** No interval is computed: the inputs are two aggregated distributions (or counts over the same categories), so there is no observation-level sample to resample. Raw samples must be binned on common bins first; inputs must be non-empty, of equal length, finite and non-negative.

## Metric 34: Wasserstein Distance

For one-dimensional empirical samples $P_n=\{x_1,\ldots,x_n\}$ and
$Q_m=\{y_1,\ldots,y_m\}$, the implemented metric delegates to SciPy’s
first Wasserstein distance without explicit sample weights:
```math
WD(P_n,Q_m)=\inf_{\gamma\in\Gamma(P_n,Q_m)}
\int_{\mathbb{R}\times\mathbb{R}}|x-y|\,d\gamma(x,y),
```
where $P_n$ and $Q_m$ are empirical distributions and
$\Gamma(P_n,Q_m)$ is the set of couplings with these marginals.
Equivalently, in one dimension,
```math
WD(P_n,Q_m)=\int_{-\infty}^{\infty}|F_{P_n}(t)-F_{Q_m}(t)|\,dt.
```
The inputs are samples, not probability vectors. To compare two histograms
$(w^P_k)$ and $(w^Q_k)$ on shared bin locations $x_k$, pass `support` $=(x_k)$
(from 1.10.0); the distance is then computed between the weighted empirical
distributions $\sum_k w^P_k\delta_{x_k}$ and $\sum_k w^Q_k\delta_{x_k}$, with no
bootstrap interval. The distance is in the units of the inputs, so there is no
universal threshold.


**95% CI (this section).** $WD$ is reported with a percentile bootstrap over each sample independently (two-sample bootstrap): resample with replacement $B=1000$ times, recompute $WD$, and take the empirical percentiles
```math
CI_{95}=\left[\hat\theta^{\ast}_{(0.025)},\ \hat\theta^{\ast}_{(0.975)}\right].
```

## Metric 35: Network Modularity

Given an adjacency or correlation matrix $A$, the implementation
constructs an undirected graph using $|A|$ with the diagonal set to zero (a
correlation of 1 between a metric and itself is not an edge) and detects
communities with greedy modularity on the same edge weights used to score them
(both from 1.10.0). For total edge weight $2m$ (summing all entries
$A_{ij}$), degree $k_i=\sum_j A_{ij}$, and community assignment $c_i$, Newman
modularity is
```math
Q=\frac{1}{2m}\sum_{i,j}
\left(A_{ij}-\frac{k_i k_j}{2m}\right)\mathbb{I}(c_i=c_j).
```


**95% CI (this section).** When the observations behind a correlation matrix are supplied (`observations`, shape observations by metrics), $NM$ is reported with a percentile bootstrap over OBSERVATIONS: $B=200$ resamples, recompute the absolute correlation matrix and its modularity, and take the empirical percentiles. Without them no interval is reported: resampling metric nodes (as up to 1.9.5) does not describe sampling uncertainty.
```math
CI_{95}=\left[\hat\theta^{\ast}_{(0.025)},\ \hat\theta^{\ast}_{(0.975)}\right].
```

## Metric 36: Transparency Score

For each explained decision $i$, the canonical appendix implementation
expects three scores in $[0,1]$: explanation quality $e_i$, feature
importance $f_i$, and interpretability $u_i$. This score is an
implementation-level aggregate for post-hoc explanation adequacy,
aligned with the clinical need to expose reasons for model outputs
rather than predictions alone. The per-decision transparency
contribution is
```math
t_i=\frac{e_i+f_i+u_i}{3}.
```
The transparency score is the empirical mean
```math
TS=\frac{1}{n}\sum_{i=1}^{n}t_i.
```
All three ratings are required for every decision and must lie in $[0,1]$;
an empty list, a missing rating or an out-of-range rating raises an error (from
1.10.0; a missing rating used to count as 0). TS reflects transparency only as
well as the supplied ratings do and does not establish readiness for clinical
use.


**95% CI (this section).** $TS$ is reported with a percentile bootstrap over per-decision transparency scores: resample with replacement $B=1000$ times, recompute $TS$, and take the empirical percentiles
```math
CI_{95}=\left[\hat\theta^{\ast}_{(0.025)},\ \hat\theta^{\ast}_{(0.975)}\right].
```

## Metric 37: Observed Perturbation Agreement (formerly Robustness Certification Score)

Let $a_{ib}$ be the agreement indicator between the original
prediction for case $i$ and the prediction under perturbation batch
$b$:
```math
a_{ib}=\mathbb{I}(\hat Y_i=\hat Y_{ib}^{\mathrm{pert}}).
```
For each perturbation batch,
```math
r_b=\frac{1}{n}\sum_{i=1}^{n}a_{ib}.
```
The implemented observed perturbation agreement (class
`ObservedPerturbationAgreement`; `RobustnessCertificationScore` is a deprecated
alias) is
```math
RCS=\frac{1}{B}\sum_{b=1}^{B}r_b.
```
It describes the stability observed on the perturbations tried; it is not a
certified robustness bound.
The implementation also reports
```math
\begin{aligned}
SD(RCS) &= \sqrt{\frac{1}{B}\sum_{b=1}^{B}(r_b-RCS)^2},\\
r_{\min} &= \min_b r_b,\qquad r_{\max}=\max_b r_b.
\end{aligned}
```
The input argument $\epsilon$ is recorded in the output but is not
used in the calculation itself. An empty list of perturbations raises an error.


**95% CI (this section).** $RCS$ is reported with a percentile bootstrap over per-perturbation agreement scores: $B=1000$ resamples, recompute $RCS$, and take the empirical percentiles
```math
CI_{95}=\left[\hat\theta^{\ast}_{(0.025)},\ \hat\theta^{\ast}_{(0.975)}\right].
```

# Implemented Statistical Estimands Included to Reach 39 Derivations

## Metric 38: Hierarchical Variance Partitioning and MAIHDA-style VPC

For a Gaussian mixed model with outcome $Y_{ij}$ for individual $i$
in group $j$, EquiMed-DSS fits a random-intercept model using the
standard multilevel variance-partitioning framework:
```math
Y_{ij}=\beta_0+X_{ij}^{\top}\beta+u_j+\varepsilon_{ij},
\qquad
u_j\sim N(0,\sigma_u^2),\quad
\varepsilon_{ij}\sim N(0,\sigma_e^2).
```
The implemented intraclass correlation from the null model is
```math
ICC_{\mathrm{HLM}}=
\frac{\sigma_u^2}{\sigma_u^2+\sigma_e^2}.
```
The reported pseudo $R^2$ is the proportional reduction in total
variance from the null model to the full model:
```math
R^2_{\mathrm{pseudo}}=
1-\frac{\sigma_{u,\mathrm{full}}^2+\sigma_{e,\mathrm{full}}^2}
{\sigma_{u,\mathrm{null}}^2+\sigma_{e,\mathrm{null}}^2}.
```
For a binary outcome analysed on the logistic latent scale, the package
documentation notes the MAIHDA-style variance partition coefficient
```math
VPC_{\mathrm{logit}}=
\frac{\sigma_u^2}{\sigma_u^2+\pi^2/3}.
```
If the mixed-model fit fails, the implementation falls back to an
ANOVA-style decomposition with $J$ groups:
```math
\begin{aligned}
SS_B &= \sum_j n_j(\bar Y_j-\bar Y)^2,\\
SS_W &= \sum_i (Y_i-\bar Y_{g(i)})^2,\\
MS_B &= \frac{SS_B}{J-1},\qquad
MS_W = \frac{SS_W}{n-J}.
\end{aligned}
```
With $n_0=\big(n-\sum_j n_j^2/n\big)/(J-1)$, the effective group size for
unbalanced groups, the fallback reports the variance components
$\hat\sigma_u^2=\max\{0,(MS_B-MS_W)/n_0\}$ and $\hat\sigma_e^2=MS_W$ and the
ICC bounded to $[0,1]$:
```math
ICC_{\mathrm{fallback}}=
\max\left\{0,\min\left[1,
\frac{MS_B-MS_W}{MS_B+(n_0-1)MS_W}
\right]\right\}.
```
(Up to 1.9.5 the fallback used the mean group size and reported the mean
squares themselves as variance components.) The fallback is announced with a
warning and the result's `method` is "anova"; group-level predictors are
included as fixed effects in the mixed model.

## Metric 39: Mediation (Product of Coefficients) and Proportion Mediated

Let $X$ be the treatment or exposure, $M$ the mediator, $Y$ the
outcome, and $C$ optional covariates. The implemented
product-of-coefficients mediation uses linear regression models:
```math
\begin{aligned}
M &= \alpha_0+\alpha_1X+C^\top\alpha_C+\varepsilon_M,\\
Y &= \beta_0+\beta_1X+\beta_2M+C^\top\beta_C+\varepsilon_Y.
\end{aligned}
```
A separate total-effect model is
```math
Y=\tau_0+\tau_1X+C^\top\tau_C+\varepsilon_T.
```
The indirect, direct, and total effects are
```math
\begin{aligned}
IE &= \alpha_1\beta_2,\\
DE &= \beta_1,\\
TE &= \tau_1.
\end{aligned}
```
The implemented proportion mediated is
```math
PM=\frac{IE}{TE},
```
and $PM$ is undefined (NaN) if $|TE|\le 10^{-10}$ (up to 1.9.5 it was set to 0).
The indirect-effect confidence interval is a percentile bootstrap over the
product $\alpha_1^\ast\beta_2^\ast$, and the direct effect has its own
bootstrap interval over $\beta_1^\ast$. The mediation type follows Zhao,
Lynch and Chen (2010): indirect effect significant and direct effect not,
complete mediation; both significant, partial mediation, complementary or
competitive by the signs; indirect effect not significant, no mediation. These
are associations from linear models: reading them as causal effects requires no
unmeasured confounding of the treatment-mediator, treatment-outcome and
mediator-outcome relations (Imai, Keele and Tingley 2010). A two-level
categorical treatment is coded 0/1 (first level in sorted order as reference)
and categorical covariates are dummy-coded. The Sobel test implemented in the same
class is
```math
\begin{aligned}
SE_{\mathrm{Sobel}} &=
\sqrt{\alpha_1^2SE(\beta_2)^2+\beta_2^2SE(\alpha_1)^2},\\
z_{\mathrm{Sobel}} &= \frac{\alpha_1\beta_2}{SE_{\mathrm{Sobel}}},\\
p &= 2\left[1-\Phi(|z_{\mathrm{Sobel}}|)\right].
\end{aligned}
```

# References

Shrout PE, Fleiss JL. Intraclass correlations: uses in assessing rater
reliability. *Psychological Bulletin*. 1979;86(2):420-428.
doi:10.1037/0033-2909.86.2.420.

Bland JM, Altman DG. Statistical methods for assessing agreement between
two methods of clinical measurement. *Lancet*. 1986;327(8476):307-310.
doi:10.1016/S0140-6736(86)90837-8.

Wilson EB. Probable inference, the law of succession, and statistical
inference. *Journal of the American Statistical Association*.
1927;22(158):209-212. doi:10.1080/01621459.1927.10502953.

Equal Employment Opportunity Commission, Civil Service Commission,
Department of Labor, Department of Justice. Uniform Guidelines on
Employee Selection Procedures. *Federal Register*.
1978;43(166):38290-38315.

Gini C. *Variabilita e Mutabilita*. Bologna: Tipografia di Paolo
Cuppini; 1912.

Shewhart WA. *Economic Control of Quality of Manufactured Product*. New
York: D. Van Nostrand Company; 1931.

Guo C, Pleiss G, Sun Y, Weinberger KQ. On calibration of modern neural
networks. In: *Proceedings of the 34th International Conference on
Machine Learning*. PMLR. 2017;70:1321-1330.

Kusner MJ, Loftus J, Russell C, Silva R. Counterfactual fairness. In:
*Advances in Neural Information Processing Systems*. 2017;30.

Shannon CE. A mathematical theory of communication. *The Bell System
Technical Journal*. 1948;27(3):379-423, 27(4):623-656.

Kullback S, Leibler RA. On information and sufficiency. *Annals of
Mathematical Statistics*. 1951;22(1):79-86. doi:10.1214/aoms/1177729694.

Lin J. Divergence measures based on the Shannon entropy. *IEEE
Transactions on Information Theory*. 1991;37(1):145-151.
doi:10.1109/18.61115.

Gibbs AL, Su FE. On choosing and bounding probability metrics.
*International Statistical Review*. 2002;70(3):419-435.
doi:10.1111/j.1751-5823.2002.tb00178.x.

Efron B. Bootstrap methods: another look at the jackknife. *Annals of
Statistics*. 1979;7(1):1-26. doi:10.1214/aos/1176344552.

Efron B, Tibshirani RJ. *An Introduction to the Bootstrap*. New York:
Chapman and Hall/CRC; 1993.

Cohen J. *Statistical Power Analysis for the Behavioral Sciences*. 2nd
ed. Hillsdale: Lawrence Erlbaum Associates; 1988.

Villani C. *Optimal Transport: Old and New*. Berlin: Springer; 2009.

Newman MEJ. Modularity and community structure in networks. *Proceedings
of the National Academy of Sciences of the United States of America*.
2006;103(23):8577-8582. doi:10.1073/pnas.0601602103.

Clauset A, Newman MEJ, Moore C. Finding community structure in very
large networks. *Physical Review E*. 2004;70:066111.
doi:10.1103/PhysRevE.70.066111.

Shapley LS. A value for n-person games. In: Kuhn HW, Tucker AW, eds.
*Contributions to the Theory of Games II*. Princeton: Princeton
University Press; 1953:307-317.

Ribeiro MT, Singh S, Guestrin C. “Why should I trust you?”: explaining
the predictions of any classifier. In: *Proceedings of the 22nd ACM
SIGKDD International Conference on Knowledge Discovery and Data Mining*.
2016:1135-1144. doi:10.1145/2939672.2939778.

Raudenbush SW, Bryk AS. *Hierarchical Linear Models: Applications and
Data Analysis Methods*. 2nd ed. Thousand Oaks: Sage Publications; 2002.

Sobel ME. Asymptotic confidence intervals for indirect effects in
structural equation models. *Sociological Methodology*. 1982;13:290-312.
doi:10.2307/270723.

Imai K, Keele L, Tingley D. A general approach to causal mediation
analysis. *Psychological Methods*. 2010;15(4):309-334.
doi:10.1037/a0020761.

Zhao X, Lynch JG Jr, Chen Q. Reconsidering Baron and Kenny: myths and truths
about mediation analysis. *Journal of Consumer Research*. 2010;37(2):197-206.
doi:10.1086/651257.

Cicchetti DV. Guidelines, criteria, and rules of thumb for evaluating normed
and standardized assessment instruments in psychology. *Psychological
Assessment*. 1994;6(4):284-290. doi:10.1037/1040-3590.6.4.284.

Künsch HR. The jackknife and the bootstrap for general stationary
observations. *Annals of Statistics*. 1989;17(3):1217-1241.
doi:10.1214/aos/1176347265.

Montgomery DC. *Introduction to Statistical Quality Control*. Hoboken:
Wiley.


# Implementation Notes

- BEMI is total variation distance, not one minus a correlation.

- GRBI is directed KL divergence in nats, not a symmetric distance.

- JSD is reported as divergence in base 2. The corresponding distance is
  also exposed as the square root.

- ECS is implemented as cosine distance, so lower values mean greater
  consistency.

- HAFG in `domain2` is a two-group count-based metric normalized by the
  larger harm. wHAFG in `domain5` is a per-sample, severity-weighted
  multi-group generalization.

- Bootstrap confidence intervals in the canonical appendix
  implementation use percentile intervals only.

- The prior PDF derivations for SPG, CHR, IVI, GRI, ICE, wHAFG, LDDI,
  and related metrics are broadly consistent with the implementation,
  but this file uses the exact implemented details where the older
  derivations were more general.

# Uncertainty Quantification (v1.10.0)

Most metrics return a 95% confidence interval with the estimate. The table at
the end states, for every metric, which interval or test the code returns, or
why none is returned (for example, when the inputs are aggregate values rather
than a sample). Throughout, $z_{1-\alpha/2}=\Phi^{-1}(1-\alpha/2)$, $\Phi$ is the
standard-normal CDF, and $B$, $P$ are the numbers of bootstrap and permutation
resamples.

## A. Proportions: Wilson interval and score test

A metric that is a proportion $\hat p = k/n$ (DFR, CHR, IVI, the Safety
Violation Rate, ATS, GCI) is reported with the **Wilson score interval**

```math
\mathrm{CI}_{1-\alpha}
= \frac{1}{1+\frac{z^{2}}{n}}
\left[\;\hat p + \frac{z^{2}}{2n}
\;\pm\; z\sqrt{\frac{\hat p(1-\hat p)}{n}+\frac{z^{2}}{4n^{2}}}\;\right],
\qquad z=z_{1-\alpha/2},
```

and, for DFR, CHR and IVI, against an acceptability threshold $p_0$ (default
$0.05$; it must lie strictly between 0 and 1), the one-sided **score test**
(`p_value_above_threshold`)

```math
Z=\frac{\hat p-p_0}{\sqrt{p_0(1-p_0)/n}},
\qquad p = 1-\Phi(Z)\quad(\text{alternative } \hat p>p_0).
```

## B. Bootstrap intervals

For a statistic $\hat\theta=T(x_1,\dots,x_n)$ the **percentile bootstrap**
draws $B$ resamples $x^{\ast(b)}$ with replacement and forms

```math
\hat\theta^{\ast(b)}=T\!\big(x^{\ast(b)}\big),
\qquad
\mathrm{CI}_{1-\alpha}
=\Big[\;\hat\theta^{\ast}_{(\alpha/2)},\;\hat\theta^{\ast}_{(1-\alpha/2)}\;\Big],
```

the $\alpha/2$ and $1-\alpha/2$ empirical quantiles of the replicates. Three
resampling designs are used:

- **Ordinary**: rows are resampled (single-sample statistics such as ECS, ERI,
  TS, and the record-based geographic indices).
- **Stratified**: rows are resampled WITHIN each group (or system-by-group cell,
  swap pair, or cross-classified cell), so every replicate keeps every group at
  its observed size and the interval describes the same groups as the estimate.
  This is used for every between-group statistic (HER, Bias-Gini, HAFG, ICE,
  wHAFG, CPS and CFU, SRPI, LDDI, REG, CIDR, DCI, UQG, HSSF, ISFV). Up to 1.9.5
  most of these pooled the groups, so a small group could be absent from a
  replicate (a group of one is absent from about 37% of them) and the interval
  then compared a different set of groups. A warning flags groups with fewer
  than five observations, and replicates with an undefined statistic are
  dropped and counted.
- **Moving block**: for the TFD time series, blocks of consecutive points are
  resampled (Künsch 1989).

When observations are **clustered** (several evaluations of the same patient or
visit), resample whole clusters instead of rows,

```math
\{g_1^{\ast},\dots,g_G^{\ast}\}\stackrel{\text{iid}}{\sim}\mathrm{Unif}\{1,\dots,G\},
\qquad
\hat\theta^{\ast(b)}=T\!\Big(\textstyle\bigcup_{j} x_{g_j^{\ast}}\Big),
```

which widens the interval to reflect within-cluster correlation. The metrics do
not do this themselves; wrap a metric with `inference.bootstrap_metric(metric_fn,
data, value_key=..., clusters=...)` or call `inference.bootstrap_ci` with
`clusters`.

## C. Permutation tests

Three metrics return a permutation $p$-value: SPG (group labels, or within-pair
swaps with `paired=True`), ISFV and MIC (shuffled outcomes). For any other
between-group gap $\Delta=\hat\theta_A-\hat\theta_B$,
`inference.permutation_test` shuffles the group labels $P$ times and reports the
add-one $p$-value

```math
p=\frac{1+\#\{\,b:\ |\Delta^{\pi_b}|\ge|\Delta_{\mathrm{obs}}|\,\}}{1+P},
```

which is never exactly zero. The metrics do not run this test themselves.

## Per-metric interval and test

| # | Metric | Estimand | Resampling unit | Returned interval / test |
|--:|---|---|---|---|
| 1 | Inter-rater Reliability ICC(2,1) | variance ratio | items | B, ordinary |
| 2 | Embedding Consistency Score | mean cosine distance | embedding pairs | B, ordinary |
| 3 | Decision Flip Rate | proportion | decisions | **A** (Wilson, $z=1.96$) + score test |
| 4 | Hierarchical Equity Ratio | max-min ratio gap | observations within groups | B, stratified (needs `group_observations`) |
| 4a | Bias-Gini Dispersion | Gini of group means | observations within groups | B, stratified (needs `group_observations`); otherwise none |
| 5 | Harm-adjusted Fairness Gap | normalized harm gap | cases within groups | B, stratified (needs case lists) |
| 6 | Ethical Risk Index | mean severity per output | outputs | B, ordinary |
| 6a | Safety Violation Rate | proportion | outputs | **A** |
| 7 | Intersectional Bias Score | mean subgroup similarity | metric dimensions | B, ordinary (sensitivity to the metrics chosen) |
| 8 | Temporal Fairness Drift | centre of the chart | consecutive points | B, moving block |
| 9 | Audit Traceability Score | proportion | audited decisions | **A** |
| 10 | Governance Compliance Index | proportion | listed checks | **A** (sampling meaning only if the checks are sampled) |
| 11 | Semantic Parity Gap | centroid distance | rows within groups, or cases when paired | B + permutation $p$ |
| 12 | Clinical Hallucination Rate | proportion | claims | **A** + score test |
| 13 | Instructional Vulnerability Index | proportion | case pairs | **A** + score test |
| 14 | Geographic Representation Index | set-based ratio | location mentions | B, ordinary (stability of the ratio) |
| 14a | Geographic Bias Correlation | correlation | query pairs | none ($p$-value only) |
| 15 | Intersectional Calibration Error | weighted ECE | samples within groups | B, stratified |
| 16 | Weighted Clinical HAFG | max-min weighted harm | samples within groups | B, stratified |
| 17 | Counterfactual Parity Score | mean similarity | cases within swap pairs | B, stratified |
| 17a | Counterfactual Unfairness | 1 minus the lowest pair mean | cases within swap pairs | B, stratified |
| 18 | Semantic Robustness Parity Index | min/max robustness | queries within groups | B, stratified |
| 19 | Lexical Diversity Disparity Index | range of RTTR | responses within groups | B, stratified |
| 20 | Recommendation Entropy Gap | range of entropy | recommendations within groups | B, stratified |
| 21 | Clinical Information Density Ratio | minimum density ratio | responses within groups | B, stratified |
| 22 | Diagnostic Completeness Index | range of coverage | responses within groups | B, stratified |
| 23 | Uncertainty Quantification Gap | range of hedging density | responses within groups | B, stratified |
| 24 | Geographic Representation Bias Index | KL divergence | evidence records | B, ordinary (needs records) |
| 25 | Healthcare System Stratified Fairness | weighted within-system gap | observations within system-group cells | B, stratified |
| 26 | Intersectional Shapley Fairness Value | range-based disparity and Shapley shares | observations within cross-classified cells | B, stratified ($B=500$) + permutation $p$ |
| 27 | Burden-Evidence Mismatch Index | total-variation distance | evidence records | B, ordinary (needs records) |
| 28 | Geographic Concentration of Coverage | normalized Gini | evidence records | B, ordinary (needs records) |
| 31 | Bias Concentration Index | one minus Herfindahl | none | none (one fixed value per group) |
| 32 | Mutual Information Content | mutual information | (demographic, outcome) pairs | B, ordinary + permutation null |
| 33 | Jensen-Shannon Divergence | divergence | none | none (aggregated distributions) |
| 34 | Wasserstein Distance | distance | each sample separately | B, two-sample; none for histograms |
| 35 | Network Modularity | modularity | observations | B (needs `observations`); otherwise none |
| 36 | Transparency Score | mean of ratings | decisions | B, ordinary |
| 37 | Observed Perturbation Agreement | mean agreement | perturbations | B, ordinary |

Metrics 29 (Bootstrap Confidence Interval) and 30 (Statistical Power Analysis)
are themselves inference utilities and define, rather than consume, the
machinery above.

### Worked instances (illustrative numbers)

These use invented counts, chosen only to show the arithmetic; they are not
results of any study.

- **CHR** with 12 unsupported of 80 claims: $12/80 = 0.150$, Wilson 95% CI
  $[0.088,\,0.244]$, one-sided $p<0.001$ vs $p_0=0.05$ (estimator **A**).
- **IVI** with 9 of 60 case pairs flipped: $9/60 = 0.150$, Wilson 95% CI
  $[0.081,\,0.261]$, one-sided $p<0.001$ vs $p_0=0.05$ (estimator **A**).
- **BEMI** for 90 studies (AFRO 5, AMRO 40, EMRO 2, EURO 30, SEARO 3, WPRO 10)
  against `WHO_REGION_IHD_BURDEN`: $\tfrac12\sum_r|e_r-b_r| = 0.485$, with a
  record-level bootstrap 95% CI of $[0.418,\,0.573]$ (estimator **B**, 1000
  resamples, seed 0); the most under-served region is SEARO.
