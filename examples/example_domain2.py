import numpy as np

from equimed_dss.domain2 import (
    EthicalRiskIndex,
    HarmAdjustedFairnessGap,
    HierarchicalEquityRatio,
    IntersectionalBiasScore,
)
from equimed_dss.utils.data_loader import generate_synthetic_fairness_data
from equimed_dss.utils.visualization import plot_her_heatmap


def main():
    print("=== Domain 2: Fairness, Equity, and Ethics ===")

    # 4. HER
    print("\n4. Hierarchical Equity Ratio (HER)")
    her_metric = HierarchicalEquityRatio()
    scores = generate_synthetic_fairness_data()
    her_scores = her_metric.calculate_her(scores)
    print(f"HER Scores: {her_scores}")
    # Note: her_scores is now a dict of dicts, interpretation is inside each group's dict
    gini = her_metric.calculate_bias_gini(list(scores.values()))
    print(f"Bias-Gini Index: {gini:.4f}")

    # Visualization
    print("Generating HER Heatmap...")
    # Wrap in expected format for heatmap: {corpus: {group: score}}
    plot_data = {"Synthetic Corpus": {g: d["score"] for g, d in her_scores.items()}}
    plot_her_heatmap(
        plot_data, title="HER Heatmap (Synthetic Data)", save_path="her_heatmap.png"
    )
    print("Plot saved to her_heatmap.png")

    # 5. HAFG
    print("\n5. Harm-Adjusted Fairness Gap (HAFG)")
    hafg_metric = HarmAdjustedFairnessGap()
    g1_err = {"fn": 5, "fp": 10}
    g2_err = {"fn": 2, "fp": 5}
    hafg_results = hafg_metric.calculate_hafg(g1_err, g2_err)
    print(f"HAFG Results: {hafg_results['hafg']:.3f}")
    print(f"Interpretation: {hafg_results['interpretation']}")

    # 6. ERI
    print("\n6. Ethical Risk Index (ERI)")
    eri_metric = EthicalRiskIndex()
    violations = [{"severity": 2.5}, {"severity": 1.0}, {"severity": 5.0}]
    eri_results = eri_metric.calculate_eri(violations, n_total_outputs=100)
    print(f"ERI Results: {eri_results['eri']:.3f}")
    print(f"Interpretation: {eri_results['interpretation']}")

    # 7. IBS
    print("\n7. Intersectional Bias Score (IBS)")
    ibs_metric = IntersectionalBiasScore()
    vectors = {
        "GroupA": np.array([0.8, 0.7, 0.9]),
        "GroupB": np.array([0.75, 0.65, 0.85]),
        "GroupC": np.array([0.5, 0.4, 0.6]),  # Outlier
    }
    ibs_results = ibs_metric.calculate_subgroup_similarity(vectors)
    print(
        f"IBS Outlier: {ibs_results['outlier_subgroup']} (Dist: {ibs_results['outlier_distance']:.3f})"
    )
    print(f"Interpretation: {ibs_results['interpretation']}")


if __name__ == "__main__":
    main()
