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

- A model output, such as `prediction`, `confidence`, or a binary decision.
- A reference outcome, such as `actual`.
- Demographic columns, such as `race`, `gender`, `age_group`, or `ses`.

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

## Notes For Clinical Use

EquiMed-DSS is an assessment library. It does not determine whether a clinical model is safe, legal, or ready for deployment by itself. Use the metrics as structured evidence inside a broader review process that includes clinical validation, data governance, regulatory review, and stakeholder oversight.
