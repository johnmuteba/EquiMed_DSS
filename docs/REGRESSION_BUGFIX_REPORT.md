# EquiMed-DSS v1.5.0 - Regression & Bug-fix Verification Report

_Seeded synthetic data; tables demonstrate the corrected metric behaviour and the reporting layer's regression tables._

### 1. Bug-fix verification (four v1.5.0 fixes)

| Fix                          | Before (bug)                       | After (v1.5.0)               | Recomputed value                      |
|:-----------------------------|:-----------------------------------|:-----------------------------|:--------------------------------------|
| DecisionFlipRate CI          | percentile of 0/1 vector ≈ [0, 1]  | Wilson 95% interval          | flip=0.20, CI=(0.057, 0.510)          |
| JSD consistency              | distance vs divergence (disagreed) | JS divergence, base 2, [0,1] | advanced=0.3651 == info_theory=0.3651 |
| HAFG normalization           | raw |H1−H2| in [0, ∞)              | |H1−H2|/max(H1,H2) in [0,1]  | hafg=0.5625 (abs gap=45)              |
| Sample SD (Bland-Altman/TFD) | population SD (ddof=0)             | sample SD (ddof=1)           | SD_diff=1.7321                        |

### 2. Hierarchical mixed-effects regression coefficients

| term      |   estimate |   std_err |      t |   p_value |   ci_lower |   ci_upper |
|:----------|-----------:|----------:|-------:|----------:|-----------:|-----------:|
| Intercept |      0.045 |     0.127 |  0.354 |     0.724 |     -0.205 |      0.295 |
| age       |      0.381 |     0.025 | 15.033 |     0     |      0.331 |      0.43  |
| acuity    |     -0.195 |     0.025 | -7.856 |     0     |     -0.244 |     -0.147 |

_ICC = 0.312; n = 1000, groups = 25; AIC = 2471.6, BIC = 2496.1._

### 3. Mediation effects (bootstrap 95% CI)

| effect   |   estimate |   ci_lower |   ci_upper |   proportion_mediated | outside_bounds   | classification                    |
|:---------|-----------:|-----------:|-----------:|----------------------:|:-----------------|:----------------------------------|
| direct   |      0.238 |    nan     |    nan     |               nan     | False            |                                   |
| indirect |      0.407 |      0.345 |      0.467 |                 0.631 | False            | Partial mediation (complementary) |
| total    |      0.644 |    nan     |    nan     |               nan     | False            |                                   |

_Proportion mediated = 63.1%; indirect 95% CI = (0.345, 0.467)._
