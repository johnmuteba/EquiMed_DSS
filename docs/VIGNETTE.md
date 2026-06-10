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

$$G_{\text{raw}} = \frac{\sum_{i}\sum_{j} \left| x_i - x_j \right|}{2 R \sum_{k} x_k}, \qquad G^{*} = \frac{R}{R-1}\, G_{\text{raw}}$$

$$H_{\text{norm}} = -\frac{\sum_{r} p_r \ln p_r}{\ln R}, \qquad \text{concentration} = 1 - H_{\text{norm}}$$

$G^{*}$ range [0, 1]: 0 = perfectly even coverage, 1 = all evidence in one region.
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

Each metric is listed with its formula and its clinical interpretation. Two
short notes flag presentation conventions worth knowing.

### Domain 1: reliability and robustness

**Decision Flip Rate (DFR)**, `DecisionFlipRate`.

$$\mathrm{DFR} = \frac{1}{n} \sum_{i=1}^{n} \mathbb{1}\!\left[\, d_i \neq d_i' \,\right]$$

where $d_i$ and $d_i'$ are the decisions before and after a counterfactual
perturbation. Range [0, 1], lower is better. Clinical meaning: the fraction of
cases whose recommendation changes when only a sensitive attribute (for example
the race label) is altered; a high value means decisions depend on identity.
Note: the reported `ci_lower`/`ci_upper` are percentiles of the 0/1 flip vector,
so they are coarse; treat `flip_rate` as the primary quantity.

**Embedding Consistency Score (ECS)**, `EmbeddingConsistencyScore`.

$$\mathrm{ECS} = \frac{1}{n} \sum_{i=1}^{n} \left( 1 - \cos(\mathbf{o}_i, \mathbf{p}_i) \right), \qquad \cos(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\lVert \mathbf{a} \rVert\, \lVert \mathbf{b} \rVert}$$

for original embeddings $\mathbf{o}_i$ and perturbed embeddings $\mathbf{p}_i$.
Range [0, 2], lower is better. Clinical meaning: how much the model's internal
representation of a case drifts under a benign rewording; higher means more
brittle to phrasing. Note: despite "Score", this returns a distance (higher =
less consistent).

**Inter-Rater Reliability (ICC)**, `InterRaterReliability`.
Two-way random-effects ICC(2,1) (Shrout and Fleiss) for $k$ raters and $n$ items:

$$\mathrm{ICC}(2,1) = \frac{MS_R - MS_E}{MS_R + (k-1) MS_E + \frac{k}{n}\left(MS_C - MS_E\right)}$$

Range about [0, 1]. Clinical meaning: agreement among raters or judges scoring
the same cases (for example multiple LLM judges); above 0.75 is excellent. Also
reports Bland-Altman limits of agreement.

### Domain 2: equity, fairness, ethics

**Hierarchical Equity Ratio (HER)**, `HierarchicalEquityRatio`.

$$\mathrm{HER}_g = \frac{\text{metric}_g}{\text{metric}_{\text{ref}}}$$

The 0.8 to 1.25 band is the four-fifths rule. Values near 1 indicate parity with
the reference group. Companion Bias-Gini over group scores $s_g$ (lower is
better), with mean $\bar{s}$:

$$G = \frac{\sum_{i}\sum_{j} \left| s_i - s_j \right|}{2 n^2\, \bar{s}}$$

**Harm-Adjusted Fairness Gap (HAFG)**, `HarmAdjustedFairnessGap`.

$$\text{harm}_g = \mathrm{FN}_g\, c_{\mathrm{FN}} + \mathrm{FP}_g\, c_{\mathrm{FP}}, \qquad \mathrm{HAFG} = \left| \text{harm}_1 - \text{harm}_2 \right|$$

Error counts are weighted by clinical cost, so a missed diagnosis (false
negative) can be penalized more heavily than an over-call.

**Ethical Risk Index (ERI)**, `EthicalRiskIndex`.

$$\mathrm{ERI} = \frac{\text{total severity}}{n_{\text{outputs}}}, \qquad \mathrm{SVR} = \frac{n_{\text{violations}}}{n_{\text{outputs}}} \times 1000$$

Clinical meaning: mean ethical severity per output, and the rate of severe
violations per 1000 outputs.

**Intersectional Bias Score (IBS)**, `IntersectionalBiasScore`.
Pairwise subgroup similarity from metric vectors $\mathbf{v}_i, \mathbf{v}_j$:

$$\text{sim}_{ij} = \frac{1}{1 + \lVert \mathbf{v}_i - \mathbf{v}_j \rVert_2}$$

plus an interaction analysis (variance attributable to race $\times$ gender
beyond main effects). Clinical meaning: detects bias that appears only at
intersections (for example a specific race-and-gender subgroup), not in any
single axis.

### Domain 3: governance and transparency

**Audit Traceability Score (ATS)**, `AuditTraceabilityScore`.
Proportion traceable, with a Wilson 95% score interval ($z = 1.96$):

$$\mathrm{ATS} = \frac{n_{\text{traceable}}}{n_{\text{total}}}, \qquad \tilde{p} = \frac{x + z^2/2}{n + z^2}, \qquad \mathrm{SE} = \sqrt{\frac{\tilde{p}\,(1 - \tilde{p})}{n + z^2}}$$

Clinical meaning: the share of decisions traceable to a specific source; the
Wilson interval is the proper small-sample interval for a proportion.

**Governance Compliance Index (GCI)**, `GovernanceComplianceIndex`.

$$\mathrm{GCI} = \frac{n_{\text{enforced}}}{n_{\text{mandated}}}$$

Range [0, 1]: the fraction of mandated governance policies actually enforced.

**Temporal Fairness Drift (TFD)**, `TemporalFairnessDrift`.
Statistical process control on a fairness time series with control limits

$$\mu \pm k\,\sigma$$

drift is flagged when a point falls outside the limits. Clinical meaning:
detects when a deployed model's fairness metric drifts over time.

### Appendix: advanced metrics

**Bootstrap Confidence Intervals**, `BootstrapConfidenceIntervals`: percentile
bootstrap of any statistic over resamples (uses `RandomState` for reproducibility).

**Bias Concentration Index (BCI)**, `BiasConcentrationIndex`, a Herfindahl-style
concentration of group bias proportions $p_r$:

$$\mathrm{BCI} = \frac{\sum_{r} p_r^{2}}{\left(\sum_{r} p_r\right)^{2}}$$

**Jensen-Shannon Divergence (JSD)**, `JensenShannonDivergence`. scipy's
`jensenshannon` returns the JS distance, so squaring recovers the divergence:

$$\mathrm{JSD}(p, q) = \mathrm{jensenshannon}(p, q)^{2}, \qquad \mathrm{JSD} \in [0, \ln 2]$$

**Mutual Information Content (MIC)**, `MutualInformationContent`: the mutual
information $I(X; Y)$ between a decision $X$ and a demographic variable $Y$
(`mutual_info_score`).

**Wasserstein Distance**, `WassersteinDistance`: the earth-mover distance
$W_1(u, v)$ between two score distributions (`scipy.stats.wasserstein_distance`).

**Network Modularity / Robustness Certification / Transparency / Statistical
Power**: graph modularity, certified robustness radius, transparency scoring, and
power and sample-size planning, respectively.

### Statistics module

**HierarchicalLinearModeling**: variance decomposition

$$\mathrm{ICC} = \frac{\sigma^{2}_{\text{between}}}{\sigma^{2}_{\text{between}} + \sigma^{2}_{\text{within}}}$$

plus fixed-effect coefficients (estimate, SE, $t$, $p$, 95% CI). AIC and BIC are
computed from the maximum-likelihood fit (REML does not define them).

**MediationAnalysis**: indirect effect $a \cdot b$, total effect $c$, direct
effect $c'$, with

$$\text{proportion mediated} = \frac{a\,b}{c}, \qquad \mathrm{SE}_{ab} = \sqrt{b^{2}\,\mathrm{SE}_a^{2} + a^{2}\,\mathrm{SE}_b^{2}}$$

(the Sobel standard error). `proportion_mediated` is reported unclamped and
flagged when outside [0, 1]; a bootstrap CI is also provided.

**NetworkStatistics**: degree, betweenness, and closeness centrality plus
clustering, computed from the metric correlation graph.

### Domain 4: representation and robustness

**Semantic Parity Gap (SPG)**, `SemanticParityGap`. Centroid distance between the
embeddings of clinical prompts differing only by a protected attribute:

$$\mathrm{SPG} = \left\lVert \frac{1}{n}\sum_i E(x_{p,i}) - \frac{1}{m}\sum_j E(x_{m,j}) \right\rVert_2, \qquad \mathrm{SPG}_{\cos} = 1 - \frac{\bar{v}_p \cdot \bar{v}_m}{\lVert \bar{v}_p\rVert\,\lVert \bar{v}_m\rVert}$$

**Clinical Hallucination Rate (CHR)**, `ClinicalHallucinationRate`. Fraction of
clinical claims not supported by the retrieved context (support score below tau):

$$\mathrm{CHR} = \frac{1}{|C(y)|}\sum_{c \in C(y)} \mathbb{1}\!\left[ S(c, K) < \tau \right], \qquad \mathrm{CHR}_{w} = \frac{\sum_c w_c\,\mathbb{1}[S(c,K)<\tau]}{\sum_c w_c}$$

**Instructional Vulnerability Index (IVI)**, `InstructionalVulnerabilityIndex`.
Decision-flip rate between a neutral and a biased instruction:

$$\mathrm{IVI} = P\!\left( f(q_b, K) \neq f(q_0, K) \right), \qquad \mathrm{IVI}_{\text{effect}} = \mathbb{E}[Y \mid q_b] - \mathbb{E}[Y \mid q_0]$$

**Geographic Representation Index (GRI)**, `GeographicRepresentationIndex`. Non-Western
share of represented locations, with the geographic-bias correlation:

$$\mathrm{GRI} = \frac{|L| - |W|}{|L|}, \qquad \mathrm{GB} = \rho\!\left( \mathrm{GRI}(K),\ \text{error rate}_m \right)$$

### Domain 5: technical-supplement fairness metrics

**Intersectional Calibration Error (ICE)**, `IntersectionalCalibrationError`:

$$\mathrm{ECE}_i = \sum_{b=1}^{B} \frac{|S_{i,b}|}{|S_i|}\,\bigl|\mathrm{acc}(S_{i,b}) - \mathrm{conf}(S_{i,b})\bigr|, \quad \mathrm{ICE} = \sum_i w_i\,\mathrm{ECE}_i, \quad \Delta\mathrm{ICE} = \max_{i,j} |\mathrm{ECE}_i - \mathrm{ECE}_j|$$

**Weighted Clinical Harm-Adjusted Fairness Gap (wHAFG)**, `WeightedClinicalHarmAdjustedFairnessGap`:

$$H(g) = \frac{1}{n_g}\sum_i \omega(Y_i)\,L(\hat{Y}_i, Y_i), \qquad \mathrm{wHAFG} = \max_{g,g'} |H(g) - H(g')|$$

**Lexical Diversity Disparity Index (LDDI)**, `LexicalDiversityDisparityIndex`:

$$\mathrm{RTTR}(g) = \frac{|V(\cup_i R_i^g)|}{\sqrt{\sum_i |R_i^g|}}, \qquad \mathrm{LDDI} = \max_g \mathrm{RTTR}(g) - \min_g \mathrm{RTTR}(g)$$

**Recommendation Entropy Gap (REG)**, `RecommendationEntropyGap`:

$$H(T\mid g) = -\sum_t P(t\mid g)\log_2 P(t\mid g), \qquad \mathrm{REG} = \max_{g,g'} |H(T\mid g) - H(T\mid g')|$$

**Counterfactual Parity Score (CPS)**, `CounterfactualParityScore`:

$$\mathrm{CPS}(a,a') = \frac{1}{n}\sum_i \mathrm{sim}\!\left( f(x_i), f(x_{i, A\leftarrow a'}) \right), \qquad \mathrm{CFU} = 1 - \min_{a,a'} \mathrm{CPS}(a,a')$$

**Clinical Information Density Ratio (CIDR)**, `ClinicalInformationDensityRatio`:

$$\mathrm{CID}(r) = \frac{|C(r)|}{|\text{tokens}(r)|}\times 100, \qquad \mathrm{CIDR}_{\min} = \min_g \frac{\mathrm{CID}(g)}{\max_{g'} \mathrm{CID}(g')}$$

**Diagnostic Completeness Index (DCI)**, `DiagnosticCompletenessIndex`:

$$\mathrm{DCI}(r) = \frac{|D(r) \cap D^{*}|}{|D^{*}|}, \qquad \Delta\mathrm{DCI} = \max_g \mathrm{DCI}(g) - \min_g \mathrm{DCI}(g)$$

**Uncertainty Quantification Gap (UQG)**, `UncertaintyQuantificationGap`:

$$\mathrm{UD}(r) = \frac{|\{t \in r : t \in U\}|}{|\text{sentences}(r)|}, \qquad \mathrm{UQG} = \max_g \mathrm{UD}(g) - \min_g \mathrm{UD}(g)$$

**Geographic Representation Bias Index (GRBI)**, `GeographicRepresentationBiasIndex`.
KL divergence of corpus geography from disease burden (complements BEMI, which is
the total-variation distance):

$$\mathrm{GRBI} = D_{\mathrm{KL}}\!\left( P_C \,\|\, P_{\text{burden}} \right) = \sum_r P_C(r)\log\frac{P_C(r)}{P_{\text{burden}}(r)}, \qquad \mathrm{HIC\ ratio} = \frac{P_C(\mathrm{HIC})}{P_{\text{burden}}(\mathrm{HIC})}$$

**Healthcare System Stratified Fairness (HSSF)**, `HealthcareSystemStratifiedFairness`:

$$\Delta_s(g,g') = \bigl| \mathbb{E}[Y \mid g, s] - \mathbb{E}[Y \mid g', s] \bigr|, \qquad \mathrm{HSSF} = \sum_s P(s)\,\max_{g,g'} \Delta_s(g,g')$$

**Intersectional Shapley Fairness Value (ISFV)**, `IntersectionalShapleyFairnessValue`:

$$\phi_i = \sum_{S \subseteq A\setminus\{i\}} \frac{|S|!\,(m-|S|-1)!}{m!}\bigl( v(S\cup\{i\}) - v(S) \bigr), \qquad I(A_i,A_j) = v(\{i,j\}) - v(\{i\}) - v(\{j\}) + v(\varnothing)$$

**Semantic Robustness Parity Index (SRPI)**, `SemanticRobustnessParityIndex`:

$$R(g) = \frac{1}{n_g}\sum_{x} \frac{1}{|\mathrm{Para}(x)|^2}\sum_{i,j} \mathrm{sim}\!\left(f(x_i), f(x_j)\right), \qquad \mathrm{SRPI} = \frac{\min_g R(g)}{\max_g R(g)}$$

## Notes For Clinical Use

EquiMed-DSS is an assessment library. It does not determine whether a clinical model is safe, legal, or ready for deployment by itself. Use the metrics as structured evidence inside a broader review process that includes clinical validation, data governance, regulatory review, and stakeholder oversight.
