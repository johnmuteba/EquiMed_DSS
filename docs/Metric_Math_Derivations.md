> **Rendering note.** This Markdown version renders every equation directly on GitHub (via MathJax). The LaTeX source (`Metric_Math_Derivations.tex`) and a compiled PDF (`Metric_Math_Derivations.pdf`) are in this folder. The final section, *Uncertainty Quantification for Every Metric*, gives the confidence-interval and p-value formulas added in v1.7.0.

# Purpose and Verification Scope

This technical document derives the mathematical definitions used by the
EquiMed-DSS library, version 1.7.0. The formulae below are aligned with
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
| 37 | Robustness certification score | RCS | `appendix.advanced_metrics` |
| 38 | Hierarchical variance partitioning | HLM/VPC | `statistics.hierarchical` |
| 39 | Causal mediation and proportion mediated | PM | `statistics.mediation` |

# Domain 1: Reliability and Robustness

## Metric 1: Inter-rater Reliability, ICC(2,1)

Let $X_{ij}$ be the score assigned to item $i$ by judge $j$, with
$n$ items and $k$ judges. Define the grand mean $\bar{X}_{..}$,
item means $\bar{X}_{i.}$, and judge means $\bar{X}_{.j}$. This
follows the Shrout-Fleiss ICC family and the Bland-Altman agreement
convention . The implementation computes
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
CI_{95\%} &= [\max(0,c-h),\min(1,c+h)],
\end{aligned}
```
with $z=1.96$, following Wilson’s score interval for binomial
proportions .

# Domain 2: Fairness, Equity, and Ethics

## Metric 4: Hierarchical Equity Ratio

Let $q_g$ be a group-specific performance score and $q_0$ the
reference-group score. The implementation calculates
```math
HER_g = \frac{q_g}{q_0},
```
with $HER_g=0$ if $q_0=0$. Values in $[0.8,1.25]$ are labelled
equitable by the implemented four-fifths-rule convention .

## Metric 4a: Bias-Gini Dispersion

For group scores $q_1,\ldots,q_K$ with mean $\bar q$, the
implemented dispersion index is the standard Gini coefficient :
```math
G_{\mathrm{bias}}=
\frac{\sum_{i=1}^{K}\sum_{j=1}^{K}|q_i-q_j|}
{2K^2\bar q}.
```
If the score list is empty or $\bar q=0$, the function returns zero.

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
implemented value is $0$.

## Metric 6: Ethical Risk Index

Let $v=1,\ldots,V$ index detected ethical or safety violations and let
$s_v$ be their severity scores. For $N$ total model outputs,
```math
ERI = \frac{\sum_{v=1}^{V}s_v}{N}.
```
If $N=0$, the function returns zero.

## Metric 6a: Safety Violation Rate

In the same implementation as ERI, the safety violation rate is
```math
SVR = 1000\frac{V}{N},
```
that is, violations per 1000 model outputs.

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
Shewhart-style control-chart logic used in statistical process control .

## Metric 9: Audit Traceability Score

Let $x$ be the number of traceable decisions among $n$ audited
decisions. The score is
```math
ATS=\frac{x}{n}.
```
The implementation returns a Wilson-style shrinkage interval using
```math
\begin{aligned}
z &= 1.96,\\
\tilde p &= \frac{x+z^2/2}{n+z^2},\\
SE_{\tilde p} &= \sqrt{\frac{\tilde p(1-\tilde p)}{n+z^2}},\\
CI_{95\%} &= [\max(0,\tilde p-zSE_{\tilde p}),\min(1,\tilde p+zSE_{\tilde p})].
\end{aligned}
```
The interval uses Wilson score shrinkage for a binomial proportion . The
score is labelled compliant at $ATS\ge 0.95$.

## Metric 10: Governance Compliance Index

Let $M$ be the number of mandated policies and $E$ the number
evaluated as enforced. The implementation defines
```math
GCI=\frac{E}{M}.
```
When no policies are provided, the returned value is zero.

# Domain 4: Representation and Robustness

## Metric 11: Semantic Parity Gap

Let $P=\{p_i\}_{i=1}^{n_p}$ be embeddings for privileged-group prompts
and $M=\{m_j\}_{j=1}^{n_m}$ embeddings for matched marginalized-group
prompts. Their centroids are
```math
\begin{aligned}
\bar p &= \frac{1}{n_p}\sum_{i=1}^{n_p}p_i,&
\bar m &= \frac{1}{n_m}\sum_{j=1}^{n_m}m_j.
\end{aligned}
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
If no weights are supplied, the implementation sets $CHR_w=CHR$.

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
protected-attribute interventions .

## Metric 14: Geographic Representation Index

Let $L$ be the set of unique locations represented in a corpus and let
$W\subseteq L$ be those counted as Western or high-income. The
implemented index is set-based:
```math
GRI=\frac{|L|-|W|}{|L|}.
```
Duplicates do not change the score.

## Metric 14a: Geographic Bias Correlation

For paired values $(x_i,y_i)$, where $x_i$ is a per-query GRI value
and $y_i$ is the corresponding non-Western error rate, the library
returns either Pearson or Spearman correlation. For Pearson,
```math
GB =
\frac{\sum_i(x_i-\bar x)(y_i-\bar y)}
{\sqrt{\sum_i(x_i-\bar x)^2}\sqrt{\sum_i(y_i-\bar y)^2}}.
```

# Domain 5: Technical-supplement Fairness

## Metric 15: Intersectional Calibration Error

For group $g$ and bin $b$, let $S_{gb}$ be samples in
intersectional group $g$ whose confidence falls in bin $b$. The
implementation uses equal-width bins on $[0,1]$. This extends expected
calibration error as used in neural-network calibration studies . Let
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

## Metric 17: Counterfactual Parity Score

Let $s_i\in[0,1]$ be a precomputed semantic similarity between the
original response and the response under a demographic swap. For a
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

## Metric 17a: Counterfactual Unfairness

The implementation returns a complement to CPS. For a single pair,
```math
CFU=1-CPS.
```
For multiple pairs,
```math
CFU=1-\min_p CPS_p.
```

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
If the maximum group robustness is zero, the function returns zero.

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

## Metric 20: Recommendation Entropy Gap

Let $P_g(t)$ be the empirical distribution of recommendation labels
$t$ in group $g$. The group recommendation entropy is based on
Shannon entropy :
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

## Metric 22: Diagnostic Completeness Index

Let $D^\star$ be the reference differential diagnosis set. For
response $i$,
```math
DCI_i=\frac{|D(R_i)\cap D^\star|}{|D^\star|}.
```
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

## Metric 23: Uncertainty Quantification Gap

Let $h_i$ be the number of hedging terms in response $R_i$ and
$q_i$ the number of sentences. The uncertainty density is
```math
UD_i=\frac{h_i}{q_i}.
```
If no sentence is detected, $UD_i=0$. For group $g$,
```math
UD(g)=\frac{1}{n_g}\sum_{i:G_i=g}UD_i.
```
The implemented uncertainty quantification gap is
```math
UQG=\max_g UD(g)-\min_g UD(g).
```

## Metric 24: Geographic Representation Bias Index

Let $p_c(r)$ be the normalized corpus evidence share in region $r$
and $p_b(r)$ the normalized burden share. EquiMed-DSS implements
directed KL divergence in nats :
```math
GRBI=D_{\mathrm{KL}}(P_c\Vert P_b)=\sum_{r:p_c(r)>0}p_c(r)\log\frac{p_c(r)}{p_b(r)}.
```
If $p_c(r)>0$ and $p_b(r)=0$, the implementation raises an error
because KL is undefined. With optional high-income regions
$\mathcal{H}$, the high-income overrepresentation ratio is
```math
HIC_{\mathrm{ratio}}=
\frac{\sum_{r\in\mathcal{H}}p_c(r)}
{\sum_{r\in\mathcal{H}}p_b(r)}.
```

## Metric 25: Healthcare System Stratified Fairness

For system stratum $s$, let
```math
\Delta_s=\max_g \mathbb{E}[Y\mid G=g,S=s]-\min_g \mathbb{E}[Y\mid G=g,S=s].
```
The implemented within-system fairness gap is
```math
HSSF=\sum_s P(S=s)\Delta_s.
```
The implementation returns $\Delta_{\mathrm{within}}=HSSF$. It also
returns the population-weighted between-system variance
```math
\Delta_{\mathrm{between}}=
\sum_s P(S=s)\left(\mathbb{E}[Y\mid S=s]-\sum_{u}P(S=u)\mathbb{E}[Y\mid S=u]\right)^2.
```

## Metric 26: Intersectional Shapley Fairness Value

Let $\mathcal{A}=\{A_1,\ldots,A_m\}$ be protected attributes. The
attribution formula follows Shapley’s cooperative-game value . For a
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

# Geographic Module

## Metric 27: Burden-evidence Mismatch Index

Let $e(r)$ be the normalized evidence share and $b(r)$ the
normalized burden share over the union of supplied regions. EquiMed-DSS
defines BEMI as total variation distance :
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
underserved region is the region with the minimum $M(r)$.

## Metric 28: Geographic Concentration of Coverage

Let $x_r\ge 0$ be regional evidence counts for $R$ regions and
$p_r=x_r/\sum_u x_u$. The raw categorical Gini is
```math
G_{\mathrm{raw}}=
\frac{\sum_{r=1}^{R}\sum_{u=1}^{R}|x_r-x_u|}
{2R\sum_{r=1}^{R}x_r}.
```
Because the maximum raw Gini for $R$ categories is $(R-1)/R$, the
implementation uses the corrected value
```math
G^\star=\frac{R}{R-1}G_{\mathrm{raw}}.
```
The normalized Shannon entropy follows Shannon’s entropy definition :
```math
H_{\mathrm{norm}}=
-\frac{\sum_{r:p_r>0}p_r\log p_r}{\log R},
```
and the concentration score is
```math
C_{\mathrm{geo}}=1-H_{\mathrm{norm}}.
```

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
Tibshirani .

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
$\lceil n\rceil$, while the returned total sample size follows the
implementation as $\lceil 2n\rceil$. The standardized effect-size
scale follows Cohen’s two-sample convention .

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
distributed bias; smaller values indicate concentration in fewer groups.

## Metric 32: Mutual Information Content

Let $D$ be a demographic categorical variable and $O$ an outcome
categorical variable. In the implementation, $D$ is expected to be
encoded as nonnegative integer categories for the entropy normalization.
The raw mutual information implemented via `mutual_info_score` uses
Shannon mutual information :
```math
MIC=I(D;O)=\sum_{d,o}p(d,o)\log\frac{p(d,o)}{p(d)p(o)}.
```
The implementation also reports a normalized value using demographic
entropy:
```math
MIC_{\mathrm{norm}}=\frac{I(D;O)}{H(D)},\qquad
H(D)=-\sum_d p(d)\log p(d).
```
If $H(D)=0$, the normalized value is zero.

## Metric 33: Jensen-Shannon Divergence

For two nonnegative vectors normalized to probability distributions
$p$ and $q$, let
```math
m=\frac{p+q}{2}.
```
The implementation returns the base-2 Jensen-Shannon divergence :
```math
JSD(p,q)=
\frac{1}{2}D_{\mathrm{KL}}^{(2)}(p\Vert m)+\frac{1}{2}D_{\mathrm{KL}}^{(2)}(q\Vert m),
```
where $D_{\mathrm{KL}}^{(2)}$ uses $\log_2$. The SciPy function returns
the distance $\sqrt{JSD}$, so the implementation squares that value.
Both `jsd` and `jsd_distance` are reported.

## Metric 34: Wasserstein Distance

For one-dimensional empirical samples $P_n=\{x_1,\ldots,x_n\}$ and
$Q_m=\{y_1,\ldots,y_m\}$, the implemented metric delegates to SciPy’s
first Wasserstein distance without explicit sample weights :
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

## Metric 35: Network Modularity

Given an adjacency or correlation matrix $A$, the implementation
constructs an undirected graph using $|A|$ and detects communities
with greedy modularity. For total edge weight $2m=\sum_{ij}A_{ij}$,
degree $k_i=\sum_j A_{ij}$, and community assignment $c_i$, Newman
modularity is
```math
Q=\frac{1}{2m}\sum_{i,j}
\left(A_{ij}-\frac{k_i k_j}{2m}\right)\mathbb{I}(c_i=c_j).
```

## Metric 36: Transparency Score

For each explained decision $i$, the canonical appendix implementation
expects three scores in $[0,1]$: explanation quality $e_i$, feature
importance $f_i$, and interpretability $u_i$. This score is an
implementation-level aggregate for post-hoc explanation adequacy,
aligned with the clinical need to expose reasons for model outputs
rather than predictions alone . The per-decision transparency
contribution is
```math
t_i=\frac{e_i+f_i+u_i}{3}.
```
The transparency score is the empirical mean
```math
TS=\frac{1}{n}\sum_{i=1}^{n}t_i.
```
If no explanations are provided, the implementation returns zero.

## Metric 37: Robustness Certification Score

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
The implemented robustness certification score is
```math
RCS=\frac{1}{B}\sum_{b=1}^{B}r_b.
```
The implementation also reports
```math
\begin{aligned}
SD(RCS) &= \sqrt{\frac{1}{B}\sum_{b=1}^{B}(r_b-RCS)^2},\\
r_{\min} &= \min_b r_b,\qquad r_{\max}=\max_b r_b.
\end{aligned}
```
The input argument $\epsilon$ is recorded in the output but is not
used in the calculation itself.

# Implemented Statistical Estimands Included to Reach 39 Derivations

## Metric 38: Hierarchical Variance Partitioning and MAIHDA-style VPC

For a Gaussian mixed model with outcome $Y_{ij}$ for individual $i$
in group $j$, EquiMed-DSS fits a random-intercept model using the
standard multilevel variance-partitioning framework :
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
The reported pseudo-$R^2$ is the proportional reduction in total
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
With $\bar n$ denoting the mean group size, the fallback ICC is
bounded to $[0,1]$:
```math
ICC_{\mathrm{fallback}}=
\max\left\{0,\min\left[1,
\frac{MS_B-MS_W}{MS_B+(\bar n-1)MS_W}
\right]\right\}.
```

## Metric 39: Causal Mediation and Proportion Mediated

Let $X$ be the treatment or exposure, $M$ the mediator, $Y$ the
outcome, and $C$ optional covariates. The implemented
product-of-coefficients mediation uses linear regression models :
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
with $PM=0$ if $|TE|\le 10^{-10}$. The indirect-effect confidence
interval is a percentile bootstrap over the product
$\alpha_1^\ast\beta_2^\ast$. The Sobel test implemented in the same
class is
```math
\begin{aligned}
SE_{\mathrm{Sobel}} &=
\sqrt{\alpha_1^2SE(\beta_2)^2+\beta_2^2SE(\alpha_1)^2},\\
z_{\mathrm{Sobel}} &= \frac{\alpha_1\beta_2}{SE_{\mathrm{Sobel}}},\\
p &= 2\left[1-\Phi(|z_{\mathrm{Sobel}}|)\right].
\end{aligned}
```

<div class="thebibliography">

99

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

</div>

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

# Uncertainty Quantification for Every Metric (v1.7.0)

From v1.7.0 a metric is never reported as a bare point estimate. Each estimate
$\hat\theta$ is accompanied by a 95% confidence interval and, where a
pre-specified null or acceptability threshold exists, a $p$-value. Three master
estimators cover every metric in this library; the table at the end tags each
metric with the one it uses, and the proportion metrics (CHR, IVI, DFR) carry
their CI and $p$-value directly in the result dict.

Throughout, $z_{1-\alpha/2}=\Phi^{-1}(1-\alpha/2)$ (so $z=1.96$ for a 95%
interval), $\Phi$ is the standard-normal CDF, and $B$, $P$ are the numbers of
bootstrap and permutation resamples.

## A. Binomial-proportion metrics — Wilson interval + score test

A metric that is a proportion $\hat p = k/n$ (CHR, IVI, DFR, Safety Violation
Rate, Audit Traceability, Counterfactual Unfairness, Transparency Score,
Robustness Certification) is reported with the **Wilson score interval**

```math
\mathrm{CI}_{1-\alpha}
= \frac{1}{1+\frac{z^{2}}{n}}
\left[\;\hat p + \frac{z^{2}}{2n}
\;\pm\; z\sqrt{\frac{\hat p(1-\hat p)}{n}+\frac{z^{2}}{4n^{2}}}\;\right],
\qquad z=z_{1-\alpha/2},
```

and, against an acceptability threshold $p_0$ (default $0.05$), the one-sided
**score test** (`p_value_above_threshold`)

```math
Z=\frac{\hat p-p_0}{\sqrt{p_0(1-p_0)/n}},
\qquad p = 1-\Phi(Z)\quad(\text{alternative } \hat p>p_0).
```

For example the Clinical Hallucination Rate returns
$\widehat{\mathrm{CHR}}=k/n$ with the Wilson CI above and
$p=1-\Phi\!\big((\hat p-p_0)\sqrt{n/(p_0(1-p_0))}\big)$.

## B. Sample statistics (means, distances, divergences) — bootstrap

For a metric $\hat\theta=T(x_1,\dots,x_n)$ that is a mean, centroid distance,
entropy, Gini, total-variation distance, or KL/JS divergence (ECS, ICC, SPG,
ICE, CPS, SRPI, LDDI, REG, CIDR, UQG, GRBI, BEMI, GCC, ISFV, MIC, JSD,
Wasserstein, and the remaining Domain-2/3/5 indices), the **percentile
bootstrap** draws $B$ resamples $x^{*(b)}$ with replacement and forms

```math
\hat\theta^{*(b)}=T\!\big(x^{*(b)}\big),
\qquad
\mathrm{CI}_{1-\alpha}
=\Big[\;\hat\theta^{*}_{(\alpha/2)},\;\hat\theta^{*}_{(1-\alpha/2)}\;\Big],
```

the $\alpha/2$ and $1-\alpha/2$ empirical quantiles of
$\{\hat\theta^{*(1)},\dots,\hat\theta^{*(B)}\}$, with standard error
$\widehat{\mathrm{se}}=\operatorname{sd}\big(\hat\theta^{*(1)},\dots,\hat\theta^{*(B)}\big)$.
When observations are **clustered** (multiple evaluations of the same
patient/visit), the **cluster bootstrap** resamples whole clusters
$g\in\{1,\dots,G\}$ rather than rows,

```math
\{g_1^{*},\dots,g_G^{*}\}\stackrel{\text{iid}}{\sim}\mathrm{Unif}\{1,\dots,G\},
\qquad
\hat\theta^{*(b)}=T\!\Big(\textstyle\bigcup_{j} x_{g_j^{*}}\Big),
```

which widens the interval to reflect within-cluster correlation; this is the
interval reported for the manuscript's per-patient proportions. Any metric can
be wrapped with `inference.bootstrap_metric(metric_fn, data, value_key=...,
clusters=...)`.

## C. Between-group gaps — permutation test

For a fairness gap $\Delta=\hat\theta_A-\hat\theta_B$ between groups $A$ and $B$
(HER, HAFG/wHAFG, IBS, HSSF, the between-group Theil component, Temporal
Fairness Drift), the **permutation test** shuffles the group labels $P$ times,
recomputes $\Delta^{\pi_b}$, and reports the add-one $p$-value

```math
p=\frac{1+\#\{\,b:\ |\Delta^{\pi_b}|\ge|\Delta_{\mathrm{obs}}|\,\}}{1+P},
```

which is never exactly zero. A bootstrap CI for $\Delta$ is available from the
same resampling.

## Per-metric interval / test assignment

| # | Metric | Estimand | Sampling unit | Method (CI / test) |
|--:|---|---|---|---|
| 1 | Inter-rater Reliability ICC(2,1) | variance ratio | targets × judges | B (bootstrap CI) |
| 2 | Embedding Consistency Score | mean cosine similarity | embedding pairs | B |
| 3 | Decision Flip Rate | proportion | decisions | **A** (Wilson + score test) |
| 4 | Hierarchical Equity Ratio | per-group ratio | group scores | C / B |
| 4a | Bias-Gini Dispersion | Gini | group scores | B |
| 5 | Harm-adjusted Fairness Gap | weighted gap | group errors | C / B |
| 6 | Ethical Risk Index | weighted mean | cases | B |
| 6a | Safety Violation Rate | proportion | cases | **A** |
| 7 | Intersectional Bias Score | dispersion | strata | B |
| 8 | Temporal Fairness Drift | difference over time | time windows | C / B |
| 9 | Audit Traceability Score | proportion | audit items | **A** |
| 10 | Governance Compliance Index | weighted proportion | checklist items | A / B |
| 11 | Semantic Parity Gap | centroid distance | prompt embeddings | B |
| 12 | Clinical Hallucination Rate | proportion | claims | **A** (Wilson + score test) |
| 13 | Instructional Vulnerability Index | proportion | case pairs | **A** (Wilson + score test) |
| 14 | Geographic Representation Index | set coverage | country types | B |
| 14a | Geographic Bias Correlation | correlation | regions | B (Fisher-$z$ / bootstrap) |
| 15 | Intersectional Calibration Error | mean abs. calibration gap | strata × bins | B |
| 16 | Weighted Clinical HAFG | weighted gap | group errors | C / B |
| 17 | Counterfactual Parity Score | mean cosine similarity | counterfactual pairs | B |
| 17a | Counterfactual Unfairness | proportion / $1-$CPS | pairs | **A** / B |
| 18 | Semantic Robustness Parity Index | parity ratio | paraphrase sets | B |
| 19 | Lexical Diversity Disparity Index | range of RTTR | group responses | B |
| 20 | Recommendation Entropy Gap | entropy difference | group recommendations | B |
| 21 | Clinical Information Density Ratio | ratio | responses | B |
| 22 | Diagnostic Completeness Index | proportion | responses | A / B |
| 23 | Uncertainty Quantification Gap | hedging-density gap | group responses | B |
| 24 | Geographic Representation Bias Index | KL divergence | evidence records | B |
| 25 | Healthcare System Stratified Fairness | within/between gap | system strata | C / B |
| 26 | Intersectional Shapley Fairness Value | Shapley share | strata | B |
| 27 | Burden-Evidence Mismatch Index | total-variation distance | evidence records | B |
| 28 | Geographic Concentration of Coverage | Gini / norm. entropy | regions | B |
| 31 | Bias Concentration Index | concentration | groups | B |
| 32 | Mutual Information Content | mutual information | paired observations | B |
| 33 | Jensen-Shannon Divergence | divergence | distribution pair | B |
| 34 | Wasserstein Distance | distance | distribution pair | B |
| 35 | Network Modularity | modularity | graph edges | B (edge bootstrap) |
| 36 | Transparency Score | proportion | checklist items | **A** |
| 37 | Robustness Certification Score | proportion | perturbations | **A** |

Metrics 29 (Bootstrap Confidence Interval) and 30 (Statistical Power Analysis)
are themselves inference utilities and define, rather than consume, the
machinery above.

### Worked instances (manuscript metrics)

- **CHR** $= 274/285 = 0.961$, Wilson 95% CI $[0.932,\,0.978]$, $p<0.001$ vs
  $p_0=0.05$.
- **IVI** $= 36/132 = 0.273$, Wilson 95% CI $[0.204,\,0.354]$, $p<0.001$ vs
  $p_0=0.05$.
- **Per-patient accuracy** $= 0.263$, cluster (by-visit) bootstrap 95% CI
  $[0.235,\,0.292]$ — entirely below the $0.443$ majority-class baseline.
- **BEMI** $= \tfrac12\sum_r|e_r-b_r| = 0.67$; a record-level bootstrap over the
  geolocated evidence gives its CI via estimator **B**.
