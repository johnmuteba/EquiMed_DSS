# EquiMed-DSS Vignette

This vignette walks through a practical EquiMed-DSS workflow: generating or loading data, calculating fairness and reliability metrics, running statistical analyses, and producing publication-ready visualizations.

The examples use synthetic data so they can be run without access to private clinical records. Replace the synthetic data with your own model outputs once your columns are standardized.

## Installation

Install the package from PyPI:

```bash
pip install equimed_dss
```

For development from a cloned repository:

```bash
pip install -e ".[dev]"
```

## Typical Data Shape

Most fairness workflows need three categories of information:

- **A model output (`prediction`)**: what your decision-support system produced
  for each case. This can be a continuous probability or confidence (for example
  a sepsis model outputting `0.82`, meaning an 82% predicted risk), or a binary
  decision (`1` = recommend admission, `0` = discharge). For calibration metrics
  use the probability; for flip-rate or harm metrics use the binary decision.
- **A reference outcome (`actual`)**: the ground truth you score the model
  against. This is the observed result, for example `1` if the patient actually
  developed sepsis and `0` if not, or the clinician-confirmed diagnosis, or the
  documented workup. It is the "right answer" the prediction is compared to.
- **Demographic columns**, such as `race`, `gender`, `age_group`, or `ses`,
  used to stratify any metric and expose disparities between groups.

Worked example of the pairing: a chest-pain triage model sees a patient and
outputs `prediction = 0.73` (73% predicted probability of acute coronary
syndrome). The patient is later confirmed to have ACS, so `actual = 1`. Across
many patients, calibration asks whether cases predicted at 73% actually turn out
positive about 73% of the time; a flip-rate asks whether the binary decision
changes when only the patient's race label is swapped.

The built-in sample generator creates a DataFrame with those ingredients:

```python
from equimed_dss.utils import SampleDataGenerator

generator = SampleDataGenerator(random_state=42)
df = generator.generate_fairness_data(
    n_samples=1000,
    include_bias=True,
    bias_magnitude=0.12,
)

print(df.head())
print(df.groupby("race")["prediction"].mean().round(3))
```

The synthetic data is not clinical evidence. It is designed to show how to call the library and how to interpret returned objects.

## Example 1: Reliability Under Perturbation

Domain 1 focuses on reliability and robustness. The library can measure whether decisions change after counterfactual perturbations and whether embeddings remain stable.

```python
import numpy as np

from equimed_dss.domain1 import DecisionFlipRate, EmbeddingConsistencyScore
from equimed_dss.utils import generate_synthetic_embeddings

rng = np.random.RandomState(42)

# Binary model decisions before and after a counterfactual perturbation.
original_decisions = rng.binomial(1, 0.55, size=200)
counterfactual_decisions = original_decisions.copy()
counterfactual_decisions[rng.choice(200, size=18, replace=False)] ^= 1

dfr = DecisionFlipRate()
dfr_result = dfr.calculate_dfr(original_decisions, counterfactual_decisions)

print("Decision flip rate:", round(dfr_result["flip_rate"], 3))
print("Interpretation:", dfr_result["interpretation"]["verdict"])

# Embedding consistency compares original and perturbed representations.
original_embeddings = generate_synthetic_embeddings(n_samples=200, dim=32)
perturbed_embeddings = original_embeddings + rng.normal(0, 0.03, original_embeddings.shape)

ecs = EmbeddingConsistencyScore()
ecs_result = ecs.calculate_ecs(original_embeddings, perturbed_embeddings)

print("Mean ECS:", round(ecs_result["mean_ecs"], 4))
print("Interpretation:", ecs_result["interpretation"]["verdict"])
```

Use this pattern when you want to ask: "Does the model preserve its decision or representation when clinically irrelevant details are changed?"

## Example 2: Inter-Rater Reliability

If you compare AI outputs against clinician reviewers or multiple adjudicators, use the inter-rater reliability tools.

```python
from equimed_dss.domain1 import InterRaterReliability
from equimed_dss.utils import SampleDataGenerator

generator = SampleDataGenerator(random_state=7)
ratings = generator.generate_reliability_data(
    n_subjects=40,
    n_raters=4,
    agreement_level=0.82,
)

irr = InterRaterReliability()
icc_result = irr.calculate_icc_2_1(ratings)
ba_result = irr.bland_altman_analysis(ratings)

print("ICC(2,1):", round(icc_result["score"], 3))
print("ICC interpretation:", icc_result["interpretation"]["verdict"])
print("First Bland-Altman pair:", next(iter(ba_result.items())))
```

An ICC close to 1 suggests strong agreement across raters. Bland-Altman output helps inspect whether pairs of raters have systematic disagreement.

## Example 3: Group Equity With HER and Bias-Gini

Domain 2 evaluates group fairness and clinical harm. A common first pass is to compare performance across demographic groups.

```python
from equimed_dss.domain2 import HierarchicalEquityRatio
from equimed_dss.utils import SampleDataGenerator

generator = SampleDataGenerator(random_state=42)
df = generator.generate_fairness_data(n_samples=1200, include_bias=True)

group_performance = df.groupby("race")["prediction"].mean().to_dict()

her = HierarchicalEquityRatio()
her_scores = her.calculate_her(group_performance, reference_group="White")
bias_gini = her.calculate_bias_gini(list(group_performance.values()))

for group, result in her_scores.items():
    print(group, round(result["score"], 3), result["interpretation"]["verdict"])

print("Bias-Gini:", round(bias_gini, 3))
```

HER compares each group to a reference group. Values near 1 are closer to parity. Bias-Gini summarizes dispersion across all groups; lower is better.

## Example 4: Harm-Adjusted Fairness Gap

Clinical AI often requires harm-aware fairness. False negatives and false positives do not always carry equal clinical consequences.

```python
from equimed_dss.domain2 import HarmAdjustedFairnessGap

hafg = HarmAdjustedFairnessGap(cost_fn=10.0, cost_fp=3.0)

result = hafg.calculate_hafg(
    group1_errors={"fn": 9, "fp": 12},
    group2_errors={"fn": 4, "fp": 15},
)

print("Group 1 harm:", result["harm_group1"])
print("Group 2 harm:", result["harm_group2"])
print("HAFG:", result["hafg"])
print("Interpretation:", result["interpretation"]["verdict"])
```

This is useful for diagnostic or triage settings where a missed case can be more harmful than an unnecessary follow-up.

## Example 5: Intersectional Bias

Single-axis fairness can miss subgroup harms. Intersectional analysis examines combinations such as race by gender.

```python
import numpy as np

from equimed_dss.domain2 import IntersectionalBiasScore

subgroup_vectors = {
    "White_Female": np.array([0.86, 0.88, 0.84]),
    "White_Male": np.array([0.85, 0.87, 0.83]),
    "Black_Female": np.array([0.67, 0.70, 0.64]),
    "Black_Male": np.array([0.76, 0.78, 0.73]),
    "Hispanic_Female": np.array([0.79, 0.81, 0.77]),
}

ibs = IntersectionalBiasScore()
result = ibs.calculate_subgroup_similarity(subgroup_vectors)

print("Mean similarity:", round(result["mean_similarity"], 3))
print("Outlier subgroup:", result["outlier_subgroup"])
```

Use the outlier subgroup list as a triage signal. It should guide deeper review, not replace clinical or statistical judgment.

## Example 6: Governance Monitoring

Domain 3 supports deployment monitoring and governance checks.

```python
from equimed_dss.domain3 import AuditTraceabilityScore, GovernanceComplianceIndex, TemporalFairnessDrift
from equimed_dss.utils import SampleDataGenerator

generator = SampleDataGenerator(random_state=11)
time_series = generator.generate_temporal_data(
    n_timepoints=40,
    drift_point=26,
    drift_magnitude=0.2,
)

tfd = TemporalFairnessDrift()
drift_result = tfd.calculate_drift(time_series.tolist())

ats = AuditTraceabilityScore()
ats_result = ats.calculate_ats(n_traceable=92, n_total=100)

gci = GovernanceComplianceIndex()
gci_result = gci.calculate_gci(
    {
        "model_card_complete": True,
        "bias_audit_complete": True,
        "monitoring_plan_complete": True,
        "human_review_workflow": False,
    }
)

print("Drift detected:", drift_result["drift_detected"])
print("ATS:", ats_result["ats_score"])
print("GCI:", gci_result["gci"])
```

These metrics are useful for deployment dashboards: fairness drift, audit traceability, and policy compliance can be tracked alongside model performance.

## Example 7: Advanced Appendix Metrics

The appendix metrics provide uncertainty, distributional comparison, power analysis, explainability, and robustness checks.

```python
import numpy as np

from equimed_dss.appendix import (
    BiasConcentrationIndex,
    BootstrapConfidenceIntervals,
    JensenShannonDivergence,
    RobustnessCertificationScore,
    StatisticalPowerAnalysis,
    TransparencyScore,
    WassersteinDistance,
)
from equimed_dss.utils import SampleDataGenerator

generator = SampleDataGenerator(random_state=42)

# Bootstrap confidence interval for a mean performance metric.
scores = np.array([0.82, 0.85, 0.87, 0.79, 0.88, 0.84, 0.83])
bci = BootstrapConfidenceIntervals(n_bootstrap=500, random_state=42)
ci_result = bci.calculate_bci(scores)
print("Bootstrap CI:", round(ci_result["ci_lower"], 3), round(ci_result["ci_upper"], 3))

# Sample size planning for group comparisons.
spa = StatisticalPowerAnalysis()
power_result = spa.calculate_sample_size(effect_size=0.35, power=0.8)
print("Required n per group:", power_result["n_per_group"])

# Bias concentration across groups.
concentration = BiasConcentrationIndex()
concentration_result = concentration.calculate_bci([0.55, 0.20, 0.15, 0.10])
print("Bias concentration:", round(concentration_result["bci"], 3))

# Distributional comparison between two prediction distributions.
dist_a, dist_b = generator.generate_distribution_data(n_samples=300, difference=0.4)
jsd = JensenShannonDivergence()
wd = WassersteinDistance()
print("JSD:", round(jsd.calculate_jsd(dist_a, dist_b)["jsd"], 3))
print("WD:", round(wd.calculate_wd(dist_a, dist_b)["wasserstein_distance"], 3))

# Explanation quality and perturbation robustness.
explanations = generator.generate_explanation_data(n_decisions=50, quality_level=0.78)
ts_result = TransparencyScore().calculate_ts(explanations)
print("Transparency score:", round(ts_result["ts"], 3))

original, perturbed = generator.generate_perturbation_data(
    n_samples=120,
    n_perturbations=5,
    robustness=0.88,
)
rcs_result = RobustnessCertificationScore().calculate_rcs(original, perturbed)
print("Robustness certification:", round(rcs_result["rcs"], 3))
```

## Example 8: Data Loading and Demographic Processing

When using local files, standardize clinical text into a `content` column and keep demographic variables in their own columns.

```python
from pathlib import Path

import pandas as pd

from equimed_dss.utils import CorpusLoader, DemographicProcessor

data_dir = Path("example_outputs")
data_dir.mkdir(exist_ok=True)

raw = pd.DataFrame(
    {
        "id": ["doc_1", "doc_2", "doc_3"],
        "clinical_note": [
            "Patient presents with chest pain.",
            "Follow-up for diabetes management.",
            "Evaluation after shortness of breath.",
        ],
        "race": ["White", "Black", "Hispanic"],
        "gender": ["Female", "Male", "Female"],
        "ses": ["Middle", "Low", "High"],
        "age_group": ["Middle Age", "Elderly", "Young Adult"],
    }
)

csv_path = data_dir / "sample_corpus.csv"
raw.to_csv(csv_path, index=False)

loader = CorpusLoader()
corpus = loader.load_from_csv(csv_path, text_column="clinical_note")
validation = loader.validate_format(corpus)

processor = DemographicProcessor()
intersections = processor.create_intersections(["race", "gender"])
counts = processor.get_subgroup_counts(corpus, "race")

print("Valid corpus:", validation["valid"])
print("Standard columns:", list(corpus.columns))
print("Number of race-gender intersections:", len(intersections))
print("Race counts:", counts)
```

## Example 9: Statistical Analyses

EquiMed-DSS includes statistical helpers for hierarchical modeling, mediation, reliability, and network analysis.

```python
import numpy as np
import pandas as pd

from equimed_dss.statistics import (
    HierarchicalLinearModeling,
    MediationAnalysis,
    NetworkStatistics,
    ReliabilityAnalysis,
)

rng = np.random.RandomState(42)
rows = []

for hospital_id in range(6):
    hospital_effect = rng.normal(0, 1)
    for _ in range(30):
        treatment = rng.binomial(1, 0.5)
        mediator = 0.6 * treatment + hospital_effect + rng.normal(0, 0.5)
        outcome = 0.3 * treatment + 0.5 * mediator + rng.normal(0, 0.6)
        rows.append(
            {
                "hospital": hospital_id,
                "treatment": treatment,
                "mediator": mediator,
                "outcome": outcome,
            }
        )

analysis_df = pd.DataFrame(rows)

hlm = HierarchicalLinearModeling()
hlm_result = hlm.fit_model(
    data=analysis_df,
    outcome_var="outcome",
    level1_predictors=["treatment", "mediator"],
    level2_var="hospital",
)
print("Hospital-level ICC:", round(hlm_result["icc"], 3))

mediation = MediationAnalysis(n_bootstrap=100, random_state=42)
mediation_result = mediation.analyze_mediation(
    data=analysis_df,
    treatment_var="treatment",
    mediator_var="mediator",
    outcome_var="outcome",
)
print("Indirect effect:", round(mediation_result["indirect_effect"], 3))

adjacency = np.array(
    [
        [0.0, 0.8, 0.3, 0.1],
        [0.8, 0.0, 0.4, 0.2],
        [0.3, 0.4, 0.0, 0.7],
        [0.1, 0.2, 0.7, 0.0],
    ]
)
network_result = NetworkStatistics().analyze_network(
    adjacency_matrix=adjacency,
    node_labels=["DFR", "HER", "HAFG", "TFD"],
)
print("Network density:", round(network_result["density"], 3))

items = np.column_stack(
    [
        analysis_df["outcome"] + rng.normal(0, 0.2, len(analysis_df))
        for _ in range(5)
    ]
)
alpha_result = ReliabilityAnalysis().cronbachs_alpha(items)
print("Cronbach alpha:", round(alpha_result["alpha"], 3))
```

## Example 10: Basic Visualizations

The visualization utilities save standard matplotlib figures when `save_path` is provided.

```python
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from equimed_dss.domain2 import HierarchicalEquityRatio
from equimed_dss.domain3 import TemporalFairnessDrift
from equimed_dss.utils import SampleDataGenerator
from equimed_dss.utils.visualization import (
    plot_control_chart,
    plot_correlation_matrix,
    plot_her_heatmap,
    plot_metric_distribution,
    plot_network_graph,
)

output_dir = Path("example_outputs")
output_dir.mkdir(exist_ok=True)

generator = SampleDataGenerator(random_state=42)
df = generator.generate_fairness_data(n_samples=800, include_bias=True)

# HER heatmap across hypothetical corpora.
her = HierarchicalEquityRatio()
race_scores = df.groupby("race")["prediction"].mean().to_dict()
her_current = {
    group: value["score"]
    for group, value in her.calculate_her(race_scores, reference_group="White").items()
}

her_scores = {
    "Current model": her_current,
    "Candidate model": {group: min(1.25, score + 0.04) for group, score in her_current.items()},
}
plot_her_heatmap(
    her_scores,
    title="Hierarchical Equity Ratio by Model",
    save_path=output_dir / "her_heatmap.png",
)

# Control chart for temporal fairness drift.
time_series = generator.generate_temporal_data(n_timepoints=35, drift_point=24)
drift = TemporalFairnessDrift().calculate_drift(time_series.tolist())
plot_control_chart(
    time_series.tolist(),
    ucl=drift["ucl"],
    lcl=drift["lcl"],
    title="Temporal Fairness Drift Control Chart",
    save_path=output_dir / "tfd_control_chart.png",
)

# Correlation matrix for selected model outputs.
metric_frame = pd.DataFrame(
    {
        "prediction": df["prediction"],
        "confidence": df["confidence"],
        "actual": df["actual"],
    }
)
plot_correlation_matrix(
    metric_frame.corr(),
    title="Metric Correlation Matrix",
    save_path=output_dir / "metric_correlation.png",
)

# Distribution of prediction scores.
plot_metric_distribution(
    df["prediction"].tolist(),
    title="Prediction Distribution",
    save_path=output_dir / "prediction_distribution.png",
)

# Network plot.
G = nx.Graph()
G.add_edge("HER", "Bias-Gini", weight=0.72)
G.add_edge("HER", "HAFG", weight=0.51)
G.add_edge("TFD", "RCS", weight=0.64)
G.add_edge("TS", "GCI", weight=0.58)
plot_network_graph(
    G,
    title="Fairness Metric Relationship Network",
    save_path=output_dir / "metric_network.png",
)

print(f"Saved plots to {output_dir.resolve()}")
```

## Example 11: Manuscript-Style Visualizations

The package also includes larger multi-panel figure functions inspired by the EquiMed-DSS manuscript. These are useful when preparing reports or supplements.

Each `plot_figure*` function expects a structured dictionary of inputs; the exact keys are listed in the function's docstring. The fastest way to start is `generate_figure_data()`, which returns ready-to-use sample inputs for every figure. Render them all, then replace the values with your own data using the same keys:

```python
from equimed_dss.utils import (
    generate_figure_data,
    plot_figure2_reliability_dashboard,
    plot_figure3_corpus_comparison,
    plot_figure4_temporal_robustness,
    plot_figure5_ethics_governance,
    plot_figure6_metric_networks,
    plot_figure7_intersectional_heatmap,
)

figs = generate_figure_data()
plot_figure2_reliability_dashboard(figs["fig2"], save_path="figures/fig2.png")
plot_figure3_corpus_comparison(figs["fig3"], save_path="figures/fig3.png")
plot_figure4_temporal_robustness(figs["fig4"], save_path="figures/fig4.png")
plot_figure5_ethics_governance(figs["fig5"], save_path="figures/fig5.png")
plot_figure6_metric_networks(figs["fig6"], save_path="figures/fig6.png")
plot_figure7_intersectional_heatmap(figs["fig7"], save_path="figures/fig7.png")
```

The remainder of this example shows the explicit dictionary structure for two of the figures, so you can see exactly what each key holds when you build your own.

```python
from pathlib import Path

import numpy as np

from equimed_dss.utils.visualization import (
    plot_figure2_reliability_dashboard,
    plot_figure7_intersectional_heatmap,
)

output_dir = Path("example_outputs")
output_dir.mkdir(exist_ok=True)

rng = np.random.RandomState(42)

reliability_data = {
    "icc_scores": {
        "Clinician A": 0.84,
        "Clinician B": 0.79,
        "AI System": 0.88,
        "Consensus": 0.91,
    },
    "cronbach_alpha": {"alpha": 0.86, "ci_lower": 0.80, "ci_upper": 0.91},
    "bland_altman": {
        "means": rng.normal(50, 8, 80),
        "diffs": rng.normal(0.2, 2.0, 80),
        "loa_upper": 4.1,
        "loa_lower": -3.7,
    },
    "temporal": {
        "timepoints": list(range(1, 9)),
        "reliability_scores": [0.82, 0.84, 0.83, 0.86, 0.85, 0.87, 0.86, 0.88],
    },
}

plot_figure2_reliability_dashboard(
    reliability_data,
    save_path=output_dir / "figure2_reliability_dashboard.png",
)

intersectional_data = {
    "fairness_matrix": np.array(
        [
            [0.88, 0.86, 0.84],
            [0.72, 0.68, 0.70],
            [0.79, 0.76, 0.78],
            [0.90, 0.87, 0.85],
        ]
    ),
    "row_labels": ["White", "Black", "Hispanic", "Asian"],
    "col_labels": ["Female", "Male", "Non-binary"],
    "pareto_optimal": np.array(
        [
            [True, False, False],
            [False, False, False],
            [False, False, False],
            [True, True, False],
        ]
    ),
    "marginal_row": [0.86, 0.70, 0.79, 0.88],
    "marginal_col": [0.82, 0.78, 0.79],
}

plot_figure7_intersectional_heatmap(
    intersectional_data,
    save_path=output_dir / "figure7_intersectional_heatmap.png",
)

print(f"Saved manuscript-style figures to {output_dir.resolve()}")
```

## Suggested Workflow For A Real Fairness Audit

1. Standardize model output data with `CorpusLoader` or your own DataFrame pipeline.
2. Check data completeness by demographic group before calculating fairness metrics.
3. Start with HER, Bias-Gini, HAFG, and IBS for group and intersectional equity.
4. Add reliability checks such as DFR, ECS, ICC, and Bland-Altman analysis.
5. Add appendix metrics when you need uncertainty intervals, distribution shifts, robustness checks, or sample size planning.
6. Monitor deployment with TFD, ATS, and GCI.
7. Save plots and metric dictionaries into your audit record so each conclusion remains reproducible.

## Geographic Equity Metrics

These quantify whether the evidence a system relies on matches where disease
burden actually falls. They use raw counts per region (normalized internally).

### Burden-Evidence Mismatch Index (BEMI)

Class: `BurdenEvidenceMismatch` in `equimed_dss.geographic`.

Formula, where $e_r$ is the evidence share of region $r$ and $b_r$ its burden share (both summing to 1):

$$\mathrm{BEMI} = \frac{1}{2} \sum_{r} \left| e_r - b_r \right|$$

This is the total-variation distance between the two distributions. Range
[0, 1]: 0 means evidence tracks burden exactly, 1 means they are completely
disjoint. BEMI equals the fraction of evidence that would have to be moved
across regions to match the burden distribution.

Clinical meaning: a system grounded on literature that under-represents
high-burden regions may give advice that does not transfer to those
populations. A high BEMI is a transferability and equity warning.

```python
from equimed_dss.geographic import BurdenEvidenceMismatch, WHO_REGION_IHD_BURDEN

bemi = BurdenEvidenceMismatch()
result = bemi.calculate_bemi(
    evidence_counts={"AFRO": 5, "AMRO": 40, "EURO": 30, "SEARO": 3, "WPRO": 10, "EMRO": 2},
    burden_shares=WHO_REGION_IHD_BURDEN,
)
print(round(result["bemi"], 3))            # total-variation distance in [0, 1]
print(result["most_underserved_region"])   # region most under-represented vs burden
print(result["interpretation"])            # human-readable verdict
```

### Geographic Concentration of Coverage (GCC)

Class: `GeographicConcentration` in `equimed_dss.geographic`.

Formulas, where $x_r$ is the count in region $r$, $p_r$ its share, and $R$ the number of regions:

$$G_{\text{raw}} = \frac{\sum_{i}\sum_{j} \left| x_i - x_j \right|}{2 R \sum_{k} x_k}, \qquad G^{\ast} = \frac{R}{R-1}\, G_{\text{raw}}$$

$$H_{\text{norm}} = -\frac{\sum_{r} p_r \ln p_r}{\ln R}, \qquad \text{concentration} = 1 - H_{\text{norm}}$$

$G^{\ast}$ range [0, 1]: 0 = perfectly even coverage, 1 = all evidence in one region.
The R/(R-1) correction is required because the raw Gini of R categories can only
reach (R-1)/R, so without it the index could never reach 1. H_norm range [0, 1]:
1 = even, 0 = single-region. G* and H_norm run in opposite directions, so
`concentration = 1 - H_norm` gives a single "higher = more concentrated" reading.

Clinical meaning: even when evidence matches burden on average (low BEMI), it can
still be concentrated in a few regions. GCC exposes that fragility.

```python
from equimed_dss.geographic import GeographicConcentration

gcc = GeographicConcentration()
result = gcc.calculate_gcc({"AFRO": 5, "AMRO": 40, "EURO": 30, "SEARO": 3, "WPRO": 10, "EMRO": 2})
print(round(result["gini_corrected"], 3))      # G* in [0, 1]
print(round(result["entropy_normalized"], 3))  # H_norm in [0, 1]
```

### The bundled reference: WHO_REGION_IHD_BURDEN

`WHO_REGION_IHD_BURDEN` holds normalized ischaemic-heart-disease (IHD) burden
shares per WHO region, derived from age-standardized IHD DALYs per 100,000
reported by Roth GA et al., 2020 (GBD Compare for IHD), rounded to the nearest
100:

```
AFRO 2730, AMRO 2070, EMRO 4200, EURO 3550, SEARO 3850, WPRO 1830
total = 18230
```

Dividing each by the total gives its share; for example AFRO = 2730 / 18230 =
0.150 and SEARO = 3850 / 18230 = 0.211.

Where the "about 36%" comes from: AFRO and SEARO together account for

$$\frac{2730 + 3850}{18230} = \frac{6580}{18230} = 0.361,$$

that is about 36% of global IHD burden. That is the figure cited in the
manuscript's geographic-gap finding.
These are published aggregate statistics (not patient-level data), so they are
safe to bundle. Pass your own `burden_shares` to use a different reference or
disease.

## Metric Formulas And Clinical Meaning

Every metric below is listed with its formula, a short clinical interpretation,
and a runnable example that prints its result. The examples share this setup:

```python
import numpy as np
import pandas as pd
from equimed_dss.utils import SampleDataGenerator

rng = np.random.RandomState(42)
gen = SampleDataGenerator(random_state=42)
```

### Domain 1: reliability and robustness

**Decision Flip Rate (DFR)**, `DecisionFlipRate`.
Clinical interpretation: the fraction of cases whose recommendation changes when
only a sensitive attribute (e.g. race) is altered; a high value means decisions
depend on identity rather than clinical need.

$$\mathrm{DFR} = \frac{1}{n} \sum_{i=1}^{n} \mathbb{1}\!\left[ d_i \neq d_i' \right]$$

```python
from equimed_dss.domain1 import DecisionFlipRate
orig = [1, 0, 1, 1, 0, 1, 0, 0]
counterfactual = [1, 0, 1, 0, 0, 1, 1, 0]   # 2 of 8 flip
print(DecisionFlipRate().calculate_dfr(orig, counterfactual)["flip_rate"])   # 0.25
```

**Embedding Consistency Score (ECS)**, `EmbeddingConsistencyScore`.
Clinical interpretation: how much the model's internal representation of a case
drifts under a benign rewording; higher means more brittle to phrasing. It is a
distance (higher = less consistent).

$$\mathrm{ECS} = \frac{1}{n}\sum_{i=1}^{n}\left(1 - \cos(\mathbf{o}_i, \mathbf{p}_i)\right)$$

```python
from equimed_dss.domain1 import EmbeddingConsistencyScore
o = rng.normal(size=(20, 16)); p = o + rng.normal(scale=0.05, size=(20, 16))
print(round(EmbeddingConsistencyScore().calculate_ecs(o, p)["mean_ecs"], 4))   # ~0.0015
```

**Inter-Rater Reliability (ICC)**, `InterRaterReliability`.
Clinical interpretation: agreement among raters/judges scoring the same cases
(e.g. several LLM judges); above 0.75 is excellent.

$$\mathrm{ICC}(2,1) = \frac{MS_R - MS_E}{MS_R + (k-1)MS_E + \frac{k}{n}\left(MS_C - MS_E\right)}$$

```python
from equimed_dss.domain1 import InterRaterReliability
judges = np.array([[4, 4, 5], [3, 3, 4], [5, 5, 5], [2, 3, 2], [4, 5, 4]])
print(round(InterRaterReliability().calculate_icc_2_1(judges)["score"], 3))   # ~0.789
```

### Domain 2: equity, fairness, ethics

**Hierarchical Equity Ratio (HER)**, `HierarchicalEquityRatio`.
Clinical interpretation: each group's performance relative to a reference group;
values near 1 indicate parity (the 0.8 to 1.25 band is the four-fifths rule). The
companion Bias-Gini summarizes dispersion across all groups (lower is better).

$$\mathrm{HER}_g = \frac{\text{metric}_g}{\text{metric}_{\text{ref}}}, \qquad G = \frac{\sum_i\sum_j |s_i - s_j|}{2 n^2\, \bar{s}}$$

```python
from equimed_dss.domain2 import HierarchicalEquityRatio
scores = {"White": 0.85, "Black": 0.78, "Hispanic": 0.80, "Asian": 0.87}
her = HierarchicalEquityRatio()
print({k: round(v["score"], 3) for k, v in her.calculate_her(scores).items()})
print(round(her.calculate_bias_gini(list(scores.values())), 4))   # 0.0242
```

**Harm-Adjusted Fairness Gap (HAFG)**, `HarmAdjustedFairnessGap`.
Clinical interpretation: error-rate gap weighted by clinical cost, so a missed
diagnosis (false negative) is penalized more than an over-call.

$$\text{harm}_g = \mathrm{FN}_g\, c_{\mathrm{FN}} + \mathrm{FP}_g\, c_{\mathrm{FP}}, \qquad \mathrm{HAFG} = \left| \text{harm}_1 - \text{harm}_2 \right|$$

```python
from equimed_dss.domain2 import HarmAdjustedFairnessGap
print(HarmAdjustedFairnessGap().calculate_hafg({"fn": 5, "fp": 10}, {"fn": 2, "fp": 5})["hafg"])   # 45.0
```

**Ethical Risk Index (ERI)**, `EthicalRiskIndex`.
Clinical interpretation: mean ethical severity per output, plus the rate of
severe violations per 1000 outputs.

$$\mathrm{ERI} = \frac{\text{total severity}}{n_{\text{outputs}}}, \qquad \mathrm{SVR} = \frac{n_{\text{violations}}}{n_{\text{outputs}}} \times 1000$$

```python
from equimed_dss.domain2 import EthicalRiskIndex
v = [{"severity": 0.8}, {"severity": 0.3}, {"severity": 0.9}]
print(EthicalRiskIndex().calculate_eri(v, n_total_outputs=100)["eri"])   # 0.02
```

**Intersectional Bias Score (IBS)**, `IntersectionalBiasScore`.
Clinical interpretation: detects bias that appears only at intersections (e.g. a
specific race-and-gender subgroup) by flagging outlier subgroups.

$$\text{sim}_{ij} = \frac{1}{1 + \lVert \mathbf{v}_i - \mathbf{v}_j \rVert_2}$$

```python
from equimed_dss.domain2 import IntersectionalBiasScore
sub = {"White_M": np.array([0.85, 0.9]), "Black_F": np.array([0.7, 0.6]),
       "Asian_M": np.array([0.88, 0.85])}
print(IntersectionalBiasScore().calculate_subgroup_similarity(sub)["outlier_subgroup"])   # Black_F
```

### Domain 3: governance and transparency

**Audit Traceability Score (ATS)**, `AuditTraceabilityScore`.
Clinical interpretation: the share of decisions traceable to a specific source,
with a Wilson 95% interval (the proper small-sample interval for a proportion).

$$\mathrm{ATS} = \frac{n_{\text{traceable}}}{n_{\text{total}}}, \qquad \tilde{p} = \frac{x + z^2/2}{n + z^2}$$

```python
from equimed_dss.domain3 import AuditTraceabilityScore
print(AuditTraceabilityScore().calculate_ats(n_traceable=92, n_total=100)["ats_score"])   # 0.92
```

**Governance Compliance Index (GCI)**, `GovernanceComplianceIndex`.
Clinical interpretation: the fraction of mandated governance policies actually
enforced.

$$\mathrm{GCI} = \frac{n_{\text{enforced}}}{n_{\text{mandated}}}$$

```python
from equimed_dss.domain3 import GovernanceComplianceIndex
policies = {"audit_logging": True, "bias_testing": True, "human_oversight": False}
print(round(GovernanceComplianceIndex().calculate_gci(policies)["gci"], 3))   # 0.667
```

**Temporal Fairness Drift (TFD)**, `TemporalFairnessDrift`.
Clinical interpretation: detects when a deployed model's fairness metric drifts
over time, using statistical-process-control limits.

$$\text{control limits} = \mu \pm k\,\sigma$$

```python
from equimed_dss.domain3 import TemporalFairnessDrift
print(TemporalFairnessDrift().calculate_drift([0.80, 0.82, 0.79, 0.85, 0.91, 0.95])["drift_detected"])
```

### Domain 4: representation and robustness

**Semantic Parity Gap (SPG)**, `SemanticParityGap`.
Clinical interpretation: how strongly the model's latent representation of an
identical clinical case shifts when only a protected attribute changes; a larger
gap means more demographic sensitivity in the model's internal encoding.

$$\mathrm{SPG} = \left\lVert \frac{1}{n}\sum_i E(x_{p,i}) - \frac{1}{m}\sum_j E(x_{m,j}) \right\rVert_2$$

```python
from equimed_dss.domain4 import SemanticParityGap
ep = rng.normal(size=(10, 8)); em = rng.normal(loc=0.3, size=(10, 8))
print(round(SemanticParityGap().calculate_spg(ep, em)["spg_euclidean"], 3))
```

**Clinical Hallucination Rate (CHR)**, `ClinicalHallucinationRate`.
Clinical interpretation: the fraction of clinical claims a response makes that
are not supported by the retrieved evidence (below an entailment threshold);
higher is worse and signals unsupported assertions.

$$\mathrm{CHR} = \frac{1}{|C(y)|}\sum_{c} \mathbb{1}\!\left[ S(c, K) < \tau \right]$$

```python
from equimed_dss.domain4 import ClinicalHallucinationRate
print(ClinicalHallucinationRate().calculate_chr([0.2, 0.4, 0.8, 0.9], tau=0.5)["chr"])   # 0.5
```

**Instructional Vulnerability Index (IVI)**, `InstructionalVulnerabilityIndex`.
Clinical interpretation: how susceptible the model is to bias-priming, i.e. how
often a biased or leading instruction changes the clinical decision relative to a
neutral one; higher means the model can be steered by suggestive prompts.

$$\mathrm{IVI} = P\!\left( f(q_b, K) \neq f(q_0, K) \right)$$

```python
from equimed_dss.domain4 import InstructionalVulnerabilityIndex
neutral = ["acs", "non_cardiac", "acs"]; biased = ["acs", "acs", "acs"]
print(round(InstructionalVulnerabilityIndex().calculate_ivi(neutral, biased)["ivi_flip_rate"], 3))
```

**Geographic Representation Index (GRI)**, `GeographicRepresentationIndex`.
Clinical interpretation: the share of represented locations that are non-Western;
values near 0 indicate a Western-centric knowledge base (by variety of locations).

$$\mathrm{GRI} = \frac{|L| - |W|}{|L|}$$

```python
from equimed_dss.domain4 import GeographicRepresentationIndex
res = GeographicRepresentationIndex().calculate_gri(["US", "GB", "BR", "CN", "TZ"], ["US", "GB", "DE"])
print(res["gri"])   # 0.6
```

### Domain 5: technical-supplement fairness metrics

**Intersectional Calibration Error (ICE)**, `IntersectionalCalibrationError`.
Clinical interpretation: reveals calibration that is good on average but poor for
a specific intersectional subgroup; dICE is the worst-case calibration gap.

$$\mathrm{ECE}_i = \sum_{b=1}^{B} \frac{|S_{i,b}|}{|S_i|}\,\bigl|\mathrm{acc}(S_{i,b}) - \mathrm{conf}(S_{i,b})\bigr|, \quad \Delta\mathrm{ICE} = \max_{i,j} |\mathrm{ECE}_i - \mathrm{ECE}_j|$$

```python
from equimed_dss.domain5 import IntersectionalCalibrationError
g = ["A"] * 4 + ["B"] * 4
print(round(IntersectionalCalibrationError().calculate_ice(
    g, [0.9] * 8, [1, 1, 1, 0, 0, 0, 0, 0], n_bins=5)["delta_ice"], 3))
```

**Weighted Clinical Harm-Adjusted Fairness Gap (wHAFG)**, `WeightedClinicalHarmAdjustedFairnessGap`.
Clinical interpretation: the largest gap in clinical-severity-weighted harm
between groups, prioritizing disparities that cause the most clinical harm.

$$H(g) = \frac{1}{n_g}\sum_i \omega(Y_i)\,L(\hat{Y}_i, Y_i), \qquad \mathrm{wHAFG} = \max_{g,g'} |H(g) - H(g')|$$

```python
from equimed_dss.domain5 import WeightedClinicalHarmAdjustedFairnessGap
print(WeightedClinicalHarmAdjustedFairnessGap().calculate_whafg(
    ["m", "m", "p", "p"], [1, 1, 1, 1], [1, 1, 0, 0])["whafg_max"])   # 1.0
```

**Lexical Diversity Disparity Index (LDDI)**, `LexicalDiversityDisparityIndex`.
Clinical interpretation: whether response vocabulary richness varies by group; a
large value can indicate more templated or stereotyped responses for some groups.

$$\mathrm{RTTR}(g) = \frac{|V(\cup_i R_i^g)|}{\sqrt{\sum_i |R_i^g|}}, \qquad \mathrm{LDDI} = \max_g \mathrm{RTTR}(g) - \min_g \mathrm{RTTR}(g)$$

```python
from equimed_dss.domain5 import LexicalDiversityDisparityIndex
print(round(LexicalDiversityDisparityIndex().calculate_lddi(
    {"A": ["pain pain pain"], "B": ["chest pain radiating to the left arm"]})["lddi"], 3))
```

**Recommendation Entropy Gap (REG)**, `RecommendationEntropyGap`.
Clinical interpretation: whether the spread of treatment/diagnosis
recommendations differs across groups, signalling differential treatment patterns.

$$H(T\mid g) = -\sum_t P(t\mid g)\log_2 P(t\mid g), \qquad \mathrm{REG} = \max_{g,g'} |H(T\mid g) - H(T\mid g')|$$

```python
from equimed_dss.domain5 import RecommendationEntropyGap
print(RecommendationEntropyGap().calculate_reg(
    {"A": ["acs", "acs"], "B": ["acs", "non_cardiac"]})["reg"])   # 1.0
```

**Counterfactual Parity Score (CPS)**, `CounterfactualParityScore`.
Clinical interpretation: how similar the full response stays when only the
patient's protected attribute is swapped; CPS near 1 is good, CFU = 1 - CPS is
the counterfactual unfairness.

$$\mathrm{CPS}(a,a') = \frac{1}{n}\sum_i \mathrm{sim}\!\left( f(x_i), f(x_{i, A\leftarrow a'}) \right), \qquad \mathrm{CFU} = 1 - \min_{a,a'} \mathrm{CPS}(a,a')$$

```python
from equimed_dss.domain5 import CounterfactualParityScore
res = CounterfactualParityScore().calculate_cps([1.0, 0.8, 0.9])
print(res["cps"], res["cfu"])   # 0.9 0.1
```

**Clinical Information Density Ratio (CIDR)**, `ClinicalInformationDensityRatio`.
Clinical interpretation: whether responses are equally concept-rich across
groups; CIDR_min near 1 means parity, lower means some group gets thinner
clinical content. Takes precomputed (concepts, tokens) per response.

$$\mathrm{CID}(r) = \frac{|C(r)|}{|\text{tokens}(r)|}\times 100, \qquad \mathrm{CIDR}_{\min} = \min_g \frac{\mathrm{CID}(g)}{\max_{g'} \mathrm{CID}(g')}$$

```python
from equimed_dss.domain5 import ClinicalInformationDensityRatio
print(ClinicalInformationDensityRatio().calculate_cidr(
    {"A": [(5, 100)], "B": [(10, 100)]})["cidr_min"])   # 0.5
```

**Diagnostic Completeness Index (DCI)**, `DiagnosticCompletenessIndex`.
Clinical interpretation: whether responses cover the guideline-required
differential diagnoses equally across groups; dDCI is the worst-case coverage gap.

$$\mathrm{DCI}(r) = \frac{|D(r) \cap D^{\ast}|}{|D^{\ast}|}, \qquad \Delta\mathrm{DCI} = \max_g \mathrm{DCI}(g) - \min_g \mathrm{DCI}(g)$$

```python
from equimed_dss.domain5 import DiagnosticCompletenessIndex
print(round(DiagnosticCompletenessIndex().calculate_dci(
    ["ACS", "PE", "GERD"], {"A": [["ACS", "PE", "GERD"]], "B": [["ACS"]]})["delta_dci"], 3))   # 0.667
```

**Uncertainty Quantification Gap (UQG)**, `UncertaintyQuantificationGap`.
Clinical interpretation: whether the model hedges (expresses uncertainty) equally
across groups; a large gap can indicate overconfidence for some groups, a
missed-diagnosis risk.

$$\mathrm{UD}(r) = \frac{|\{t \in r : t \in U\}|}{|\text{sentences}(r)|}, \qquad \mathrm{UQG} = \max_g \mathrm{UD}(g) - \min_g \mathrm{UD}(g)$$

```python
from equimed_dss.domain5 import UncertaintyQuantificationGap
print(round(UncertaintyQuantificationGap().calculate_uqg(
    {"A": ["This is ACS."], "B": ["This may be ACS. Consider PE."]})["uqg"], 3))
```

**Geographic Representation Bias Index (GRBI)**, `GeographicRepresentationBiasIndex`.
Clinical interpretation: how far the corpus geography departs from global disease
burden (KL divergence); the HIC ratio shows high-income overrepresentation.
Complements BEMI (which is the symmetric total-variation distance).

$$\mathrm{GRBI} = D_{\mathrm{KL}}\!\left( P_C \,\|\, P_{\text{burden}} \right) = \sum_r P_C(r)\log\frac{P_C(r)}{P_{\text{burden}}(r)}$$

```python
from equimed_dss.domain5 import GeographicRepresentationBiasIndex
print(round(GeographicRepresentationBiasIndex().calculate_grbi(
    {"AMRO": 100, "EURO": 10, "AFRO": 1}, {"AMRO": 0.2, "EURO": 0.4, "AFRO": 0.4})["grbi"], 3))
```

**Healthcare System Stratified Fairness (HSSF)**, `HealthcareSystemStratifiedFairness`.
Clinical interpretation: separates demographic disparity within each healthcare
system from variation between systems, so an apparent group gap is not mistaken
for a system (access) effect.

$$\Delta_s(g,g') = \bigl| \mathbb{E}[Y \mid g, s] - \mathbb{E}[Y \mid g', s] \bigr|, \qquad \mathrm{HSSF} = \sum_s P(s)\,\max_{g,g'} \Delta_s(g,g')$$

```python
from equimed_dss.domain5 import HealthcareSystemStratifiedFairness
print(HealthcareSystemStratifiedFairness().calculate_hssf(
    ["US", "US", "UK", "UK"], ["m", "p", "m", "p"], [0.0, 0.4, 0.1, 0.2])["hssf"])
```

**Intersectional Shapley Fairness Value (ISFV)**, `IntersectionalShapleyFairnessValue`.
Clinical interpretation: fairly attributes an intersectional disparity to each
protected attribute and their interaction; a positive interaction is a
superadditive (intersectional) penalty.

$$\phi_i = \sum_{S \subseteq A\setminus\{i\}} \frac{|S|!\,(m-|S|-1)!}{m!}\bigl( v(S\cup\{i\}) - v(S) \bigr)$$

```python
from equimed_dss.domain5 import IntersectionalShapleyFairnessValue
race = rng.choice(["W", "B"], 200); gender = rng.choice(["F", "M"], 200)
isfv = IntersectionalShapleyFairnessValue(min_cell=5).calculate_isfv(
    {"race": race, "gender": gender}, (race == "B").astype(float))
print({k: round(v, 3) for k, v in isfv["shapley_by_attribute"].items()})
```

**Semantic Robustness Parity Index (SRPI)**, `SemanticRobustnessParityIndex`.
Clinical interpretation: whether the model is equally robust to paraphrasing
across groups; SRPI near 1 means equal robustness, lower means some group's
outputs are more sensitive to wording.

$$\mathrm{SRPI} = \frac{\min_g R(g)}{\max_g R(g)}$$

```python
from equimed_dss.domain5 import SemanticRobustnessParityIndex
print(round(SemanticRobustnessParityIndex().calculate_srpi(
    {"A": [0.9, 0.9], "B": [0.6, 0.6]})["srpi"], 3))   # 0.667
```

### Statistics module

**HierarchicalLinearModeling**: variance decomposition with
$\mathrm{ICC} = \sigma^{2}_{\text{between}} / (\sigma^{2}_{\text{between}} + \sigma^{2}_{\text{within}})$,
fixed-effect coefficients (estimate, SE, t, p, 95% CI), and AIC/BIC from the
maximum-likelihood fit. Use it to quantify how much variation sits at the
institution/group level versus the individual level.

```python
from equimed_dss.statistics import HierarchicalLinearModeling
rows = []
for grp in range(8):
    ge = rng.normal(0, 2)
    for _ in range(25):
        x = rng.normal(0, 1); rows.append({"group": grp, "x": x, "y": ge + 0.5 * x + rng.normal(0, 1)})
df = pd.DataFrame(rows)
res = HierarchicalLinearModeling().fit_model(df, outcome_var="y", level1_predictors=["x"], level2_var="group")
print(round(res["icc"], 3), round(res["aic"], 1))
```

**MediationAnalysis**: indirect effect $a\,b$, total $c$, direct $c'$,
$\text{proportion mediated} = ab/c$, with a Sobel standard error and a bootstrap
CI. Use it to test whether a demographic effect operates through an intermediate
mechanism.

```python
from equimed_dss.statistics import MediationAnalysis
n = 300; X = rng.normal(0, 1, n); M = 0.5 * X + rng.normal(0, 1, n); Y = 0.3 * X + 0.4 * M + rng.normal(0, 1, n)
res = MediationAnalysis().analyze_mediation(
    pd.DataFrame({"X": X, "M": M, "Y": Y}), treatment_var="X", mediator_var="M", outcome_var="Y")
print(round(res["proportion_mediated"], 3))
```

**NetworkStatistics**: degree, betweenness, and closeness centrality plus
clustering from a metric correlation graph; use it to see which metrics cluster
together.

```python
from equimed_dss.statistics import NetworkStatistics
adj = np.array([[0, .6, .2, 0], [.6, 0, 0, .5], [.2, 0, 0, 0], [0, .5, 0, 0]])
res = NetworkStatistics().analyze_network(adj, node_labels=["DFR", "ECS", "HER", "IBS"])
print(round(res["density"], 3))
```

**ReliabilityAnalysis**: Cronbach's alpha and Bland-Altman limits of agreement
for rater/instrument reliability.

```python
from equimed_dss.statistics import ReliabilityAnalysis
ratings = np.array([[4, 4, 5], [3, 3, 4], [5, 5, 5], [2, 3, 2], [4, 5, 4]])
print(round(ReliabilityAnalysis().cronbachs_alpha(ratings)["alpha"], 3))
```

### Appendix: advanced metrics

**Bias Concentration Index (BCI)**, `BiasConcentrationIndex`. Herfindahl-style
concentration of bias across groups: $\sum_r p_r^2 / (\sum_r p_r)^2$.

```python
from equimed_dss.appendix import BiasConcentrationIndex
print(round(BiasConcentrationIndex().calculate_bci([0.1, 0.4, 0.3, 0.2])["bci"], 3))
```

**Bootstrap Confidence Intervals**, `BootstrapConfidenceIntervals`. Percentile
bootstrap CI for any statistic (uses `RandomState` for reproducibility).

```python
from equimed_dss.appendix import BootstrapConfidenceIntervals
boot = BootstrapConfidenceIntervals(n_bootstrap=500, random_state=42).calculate_bci(rng.normal(0.7, 0.1, 100))
print(round(boot["ci_lower"], 3), round(boot["ci_upper"], 3))
```

**Jensen-Shannon Divergence (JSD)**, `JensenShannonDivergence`. Symmetric
distributional distance, $\mathrm{JSD} = \mathrm{jensenshannon}(p,q)^2 \in [0, \ln 2]$.

```python
from equimed_dss.appendix import JensenShannonDivergence
p = np.array([0.9, 0.85, 0.78, 0.92]); q = np.array([0.75, 0.70, 0.68, 0.72])
print(round(JensenShannonDivergence().calculate_jsd(p, q)["jsd"], 4))
```

**Wasserstein Distance (WD)**, `WassersteinDistance`. Earth-mover distance
$W_1(u,v)$ between two score distributions.

```python
from equimed_dss.appendix import WassersteinDistance
print(round(WassersteinDistance().calculate_wd(p, q)["wasserstein_distance"], 4))
```

**Mutual Information Content (MIC)**, `MutualInformationContent`. Shared
information $I(X;Y)$ between a decision and a demographic variable.

```python
from equimed_dss.appendix import MutualInformationContent
print(round(MutualInformationContent().calculate_mic(rng.randint(0, 2, 200), rng.randint(0, 2, 200))["mic"], 4))
```

**Network Modularity (NM)**, `NetworkModularity`. Community structure of a metric
graph.

```python
from equimed_dss.appendix import NetworkModularity
adj = np.array([[0, .8, .1, 0], [.8, 0, 0, .7], [.1, 0, 0, .6], [0, .7, .6, 0]])
print(round(NetworkModularity().calculate_modularity(adj)["modularity"], 3))
```

**Robustness Certification Score (RCS)**, `RobustnessCertificationScore`. Certified
stability of predictions under bounded perturbations.

```python
from equimed_dss.appendix import RobustnessCertificationScore
orig = rng.normal(0.8, 0.05, 50); pert = [rng.normal(0.8, 0.05, 50) for _ in range(5)]
print(round(RobustnessCertificationScore().calculate_rcs(orig, pert, epsilon=0.1)["rcs"], 3))
```

**Transparency Score (TS)**, `TransparencyScore`. Quality of model explanations.

```python
from equimed_dss.appendix import TransparencyScore
exps = [{"explanation_quality": 0.9, "feature_importance": 0.8, "interpretability": 0.85},
        {"explanation_quality": 0.7, "feature_importance": 0.6, "interpretability": 0.65}]
print(round(TransparencyScore().calculate_ts(exps)["ts"], 3))   # 0.75
```

**Statistical Power Analysis (SPA)**, `StatisticalPowerAnalysis`. Sample-size and
power planning for disparity detection.

```python
from equimed_dss.appendix import StatisticalPowerAnalysis
print(StatisticalPowerAnalysis().calculate_sample_size(effect_size=0.2, alpha=0.05, power=0.8)["n_per_group"])   # 394
```

## Notes For Clinical Use

EquiMed-DSS is an assessment library. It does not determine whether a clinical model is safe, legal, or ready for deployment by itself. Use the metrics as structured evidence inside a broader review process that includes clinical validation, data governance, regulatory review, and stakeholder oversight.
