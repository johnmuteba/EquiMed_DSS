#!/usr/bin/env python
"""
Comprehensive example demonstrating all 9 Advanced Metrics from EquiMed_DSS Appendix.

This example showcases each advanced metric with:
- Clear data setup
- Metric calculation
- Interpretation of results

Reference: Manuscript Equations 11-19
"""

import numpy as np

from equimed_dss.appendix import (
    BiasConcentrationIndex,
    BootstrapConfidenceIntervals,
    JensenShannonDivergence,
    MutualInformationContent,
    NetworkModularity,
    RobustnessCertificationScore,
    StatisticalPowerAnalysis,
    TransparencyScore,
    WassersteinDistance,
)
from equimed_dss.utils import SampleDataGenerator


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")


def print_result(result: dict, indent: int = 2):
    """Pretty print a result dictionary."""
    for key, value in result.items():
        if isinstance(value, dict):
            print(f"{' '*indent}{key}:")
            print_result(value, indent + 2)
        elif isinstance(value, float):
            print(f"{' '*indent}{key}: {value:.4f}")
        else:
            print(f"{' '*indent}{key}: {value}")


def main():
    print("=" * 60)
    print(" EquiMed_DSS: Advanced Metrics Demonstration")
    print(" 9 Novel Metrics for Clinical AI Fairness Assessment")
    print("=" * 60)

    # Initialize sample data generator
    generator = SampleDataGenerator(random_state=42)

    # ========================================================================
    # Metric 1: Bootstrap Confidence Intervals (BCI) - Equation 11
    # ========================================================================
    print_section("1. Bootstrap Confidence Intervals (BCI)")
    print("Purpose: Robust uncertainty estimation without distributional assumptions")
    print("Use case: Quantify reliability of performance metrics")
    print()

    # Generate sample performance data
    performance_data = np.random.RandomState(42).normal(0.85, 0.05, 100)

    bci = BootstrapConfidenceIntervals(n_bootstrap=1000, random_state=42)
    result = bci.calculate_bci(performance_data, statistic=np.mean, alpha=0.05)

    print("Input: 100 performance measurements (mean ~0.85, std ~0.05)")
    print()
    print("Results:")
    print(f"  Observed statistic: {result['observed_statistic']:.4f}")
    print(f"  95% CI: [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]")
    print(f"  CI width: {result['ci_width']:.4f}")
    print(f"  Stability: {result['interpretation']['stability']}")
    print(f"  Verdict: {result['interpretation']['verdict']}")

    # ========================================================================
    # Metric 2: Statistical Power Analysis (SPA) - Equation 12
    # ========================================================================
    print_section("2. Statistical Power Analysis (SPA)")
    print("Purpose: Determine sample size needed to detect meaningful differences")
    print("Use case: Ensure adequate power to detect demographic disparities")
    print()

    spa = StatisticalPowerAnalysis()

    # Calculate required sample size for medium effect
    result_size = spa.calculate_sample_size(effect_size=0.5, power=0.8, alpha=0.05)
    print("Scenario 1: Required sample size for medium effect (d=0.5)")
    print(f"  Required N per group: {result_size['n_per_group']}")
    print(f"  Total N needed: {result_size['total_n']}")
    print()

    # Calculate achieved power for given sample
    result_power = spa.calculate_power(n=50, effect_size=0.5, alpha=0.05)
    print("Scenario 2: Achieved power with N=50 per group")
    print(f"  Achieved power: {result_power['power']:.3f}")
    print(f"  Verdict: {result_power['interpretation']['verdict']}")

    # ========================================================================
    # Metric 3: Bias Concentration Index (BCI) - Equation 13
    # ========================================================================
    print_section("3. Bias Concentration Index")
    print("Purpose: Measure whether bias is concentrated in specific groups")
    print("Use case: Identify if disparities affect all groups or are targeted")
    print()

    bci_metric = BiasConcentrationIndex()

    # Scenario 1: Distributed bias (affects all groups somewhat equally)
    distributed_bias = [0.25, 0.25, 0.25, 0.25]
    result1 = bci_metric.calculate_bci(distributed_bias)
    print("Scenario 1: Evenly distributed bias [0.25, 0.25, 0.25, 0.25]")
    print(f"  BCI: {result1['bci']:.4f}")
    print(f"  Distribution: {result1['interpretation']['distribution']}")
    print()

    # Scenario 2: Concentrated bias (primarily affects one group)
    concentrated_bias = [0.7, 0.1, 0.1, 0.1]
    result2 = bci_metric.calculate_bci(concentrated_bias)
    print("Scenario 2: Concentrated bias [0.7, 0.1, 0.1, 0.1]")
    print(f"  BCI: {result2['bci']:.4f}")
    print(f"  Distribution: {result2['interpretation']['distribution']}")
    print(f"  Verdict: {result2['interpretation']['verdict']}")

    # ========================================================================
    # Metric 4: Mutual Information Content (MIC) - Equation 14
    # ========================================================================
    print_section("4. Mutual Information Content (MIC)")
    print("Purpose: Detect information leakage from demographics to outcomes")
    print("Use case: Check if protected attributes influence AI predictions")
    print()

    mic = MutualInformationContent()

    # Scenario 1: Independent (no leakage)
    rng = np.random.RandomState(42)
    demographics_independent = rng.randint(0, 3, 200)
    outcomes_independent = rng.randint(0, 2, 200)

    result1 = mic.calculate_mic(demographics_independent, outcomes_independent)
    print("Scenario 1: Demographics and outcomes are independent")
    print(f"  MIC: {result1['mic']:.4f}")
    print(f"  Leakage level: {result1['interpretation']['leakage_level']}")
    print()

    # Scenario 2: Correlated (potential leakage)
    demographics_correlated = rng.randint(0, 3, 200)
    outcomes_correlated = (demographics_correlated > 1).astype(int)  # Correlated

    result2 = mic.calculate_mic(demographics_correlated, outcomes_correlated)
    print("Scenario 2: Demographics predict outcomes (leakage)")
    print(f"  MIC: {result2['mic']:.4f}")
    print(f"  Leakage level: {result2['interpretation']['leakage_level']}")
    print(f"  Verdict: {result2['interpretation']['verdict']}")

    # ========================================================================
    # Metric 5: Jensen-Shannon Divergence (JSD) - Equation 15
    # ========================================================================
    print_section("5. Jensen-Shannon Divergence (JSD)")
    print("Purpose: Measure distributional differences between groups")
    print("Use case: Compare prediction distributions across demographics")
    print()

    jsd = JensenShannonDivergence()

    # Scenario 1: Similar distributions
    dist_a = np.array([0.3, 0.4, 0.3])
    dist_b = np.array([0.35, 0.35, 0.3])

    result1 = jsd.calculate_jsd(dist_a, dist_b)
    print("Scenario 1: Similar distributions")
    print(f"  Distribution A: {dist_a}")
    print(f"  Distribution B: {dist_b}")
    print(f"  JSD: {result1['jsd']:.4f}")
    print(f"  Similarity: {result1['interpretation']['similarity']}")
    print()

    # Scenario 2: Different distributions
    dist_c = np.array([0.1, 0.2, 0.7])
    dist_d = np.array([0.6, 0.3, 0.1])

    result2 = jsd.calculate_jsd(dist_c, dist_d)
    print("Scenario 2: Different distributions")
    print(f"  Distribution C: {dist_c}")
    print(f"  Distribution D: {dist_d}")
    print(f"  JSD: {result2['jsd']:.4f}")
    print(f"  Similarity: {result2['interpretation']['similarity']}")
    print(f"  Verdict: {result2['interpretation']['verdict']}")

    # ========================================================================
    # Metric 6: Wasserstein Distance (WD) - Equation 16
    # ========================================================================
    print_section("6. Wasserstein Distance (WD)")
    print("Purpose: Measure optimal transport distance between distributions")
    print("Use case: Robust comparison resistant to outliers")
    print()

    wd = WassersteinDistance()

    # Generate two prediction distributions
    group_a_preds = np.random.RandomState(42).normal(0.8, 0.1, 100)
    group_b_preds = np.random.RandomState(42).normal(0.7, 0.1, 100)

    result = wd.calculate_wd(group_a_preds, group_b_preds)
    print("Comparing prediction distributions between two groups:")
    print(f"  Group A: mean={np.mean(group_a_preds):.2f}")
    print(f"  Group B: mean={np.mean(group_b_preds):.2f}")
    print()
    print(f"  Wasserstein Distance: {result['wasserstein_distance']:.4f}")
    print(f"  Difference level: {result['interpretation']['difference_level']}")
    print(f"  Verdict: {result['interpretation']['verdict']}")

    # ========================================================================
    # Metric 7: Network Modularity (NM) - Equation 17
    # ========================================================================
    print_section("7. Network Modularity (NM)")
    print("Purpose: Identify clustered relationships among fairness metrics")
    print("Use case: Understand metric interdependencies")
    print()

    nm = NetworkModularity()

    # Generate a correlation matrix (adjacency) for metrics
    adjacency = generator.generate_network_data(n_nodes=8)

    result = nm.calculate_modularity(adjacency)
    print(f"Analyzing network with {adjacency.shape[0]} nodes (metrics)")
    print()
    print(f"  Modularity (Q): {result['modularity']:.4f}")
    print(f"  Number of communities: {result.get('n_communities', 'N/A')}")
    print(f"  Clustering strength: {result['interpretation']['clustering_strength']}")
    print(f"  Verdict: {result['interpretation']['verdict']}")

    # ========================================================================
    # Metric 8: Transparency Score (TS) - Equation 18
    # ========================================================================
    print_section("8. Transparency Score (TS)")
    print("Purpose: Evaluate AI explanation quality for clinicians")
    print("Use case: Assess readiness for clinical deployment")
    print()

    ts = TransparencyScore()

    # Generate explanation quality data
    explanations = generator.generate_explanation_data(
        n_decisions=50, quality_level=0.75
    )

    result = ts.calculate_ts(explanations)
    print(f"Evaluating {len(explanations)} AI decision explanations:")
    print()
    print(f"  Overall Transparency Score: {result['ts']:.4f}")
    print(f"  Mean explanation quality: {result['mean_explanation_quality']:.4f}")
    print(f"  Mean feature importance: {result['mean_feature_importance']:.4f}")
    print(f"  Mean interpretability: {result['mean_interpretability']:.4f}")
    print()
    print(f"  Transparency level: {result['interpretation']['transparency_level']}")
    print(f"  Verdict: {result['interpretation']['verdict']}")

    # ========================================================================
    # Metric 9: Robustness Certification Score (RCS) - Equation 19
    # ========================================================================
    print_section("9. Robustness Certification Score (RCS)")
    print("Purpose: Quantify model stability under input perturbations")
    print("Use case: Ensure predictions are robust to clinical variations")
    print()

    rcs = RobustnessCertificationScore()

    # Generate original and perturbed predictions
    original, perturbed = generator.generate_perturbation_data(
        n_samples=100, n_perturbations=10, robustness=0.85
    )

    result = rcs.calculate_rcs(original, perturbed, epsilon=0.1)
    print(f"Testing robustness with {len(perturbed)} perturbation sets:")
    print()
    print(f"  RCS: {result['rcs']:.4f}")
    print(f"  RCS std: {result['rcs_std']:.4f}")
    print(f"  Min consistency: {result['min_consistency']:.4f}")
    print(f"  Max consistency: {result['max_consistency']:.4f}")
    print()
    print(f"  Robustness level: {result['interpretation']['robustness_level']}")
    print(f"  Verdict: {result['interpretation']['verdict']}")

    # ========================================================================
    # Summary
    # ========================================================================
    print_section("Summary")
    print("All 9 advanced metrics successfully demonstrated!")
    print()
    print("Metric Categories:")
    print("  - Reliability: BCI, SPA, Bias Concentration Index")
    print("  - Information Theory: MIC, JSD, WD")
    print("  - Network & Governance: NM, TS, RCS")
    print()
    print("For more details, see the manuscript Appendix A (Equations 11-19)")
    print()
    print("Next steps:")
    print("  1. Apply these metrics to your clinical AI system")
    print("  2. Compare results against thresholds in documentation")
    print("  3. Address any concerning findings before deployment")


if __name__ == "__main__":
    main()
