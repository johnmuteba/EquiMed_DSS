import numpy as np

from equimed_dss.domain1 import (
    DecisionFlipRate,
    EmbeddingConsistencyScore,
    InterRaterReliability,
)
from equimed_dss.utils.data_loader import (
    generate_synthetic_embeddings,
    generate_synthetic_judge_data,
)
from equimed_dss.utils.visualization import plot_bland_altman


def main():
    print("=== Domain 1: Reliability and Robustness ===")

    # 1. ICC
    print("\n1. Inter-Rater Reliability (ICC)")
    icc_metric = InterRaterReliability()
    judge_data = generate_synthetic_judge_data()
    icc_results = icc_metric.calculate_icc_2_1(judge_data)
    print(f"ICC(2,1) Score: {icc_results['score']:.3f}")
    print(f"Interpretation: {icc_results['interpretation']}")

    ba_results = icc_metric.bland_altman_analysis(judge_data)
    print("Bland-Altman Results (Judge1 vs Judge2):")
    print(ba_results["Judge1-Judge2"])

    # Visualization
    print("Generating Bland-Altman Plot...")
    # Extract data for plot
    j1 = judge_data[:, 0]
    j2 = judge_data[:, 1]
    means = (j1 + j2) / 2
    diffs = j1 - j2
    limits = (
        ba_results["Judge1-Judge2"]["lower_loa"],
        ba_results["Judge1-Judge2"]["upper_loa"],
    )
    plot_bland_altman(
        means,
        diffs,
        limits,
        title="Bland-Altman: Judge 1 vs Judge 2",
        save_path="bland_altman.png",
    )
    print("Plot saved to bland_altman.png")

    # 2. ECS
    print("\n2. Embedding Consistency Score (ECS)")
    ecs_metric = EmbeddingConsistencyScore()
    orig_emb = generate_synthetic_embeddings()
    pert_emb = orig_emb + np.random.normal(0, 0.1, orig_emb.shape)
    ecs_results = ecs_metric.calculate_ecs(orig_emb, pert_emb)
    print(f"ECS Results: {ecs_results['mean_ecs']:.3f}")
    print(f"Interpretation: {ecs_results['interpretation']}")

    # 3. DFR
    print("\n3. Decision Flip Rate (DFR)")
    dfr_metric = DecisionFlipRate()
    orig_decisions = np.random.randint(0, 2, 100)
    # Flip 10%
    new_decisions = orig_decisions.copy()
    flip_indices = np.random.choice(100, 10, replace=False)
    new_decisions[flip_indices] = 1 - new_decisions[flip_indices]

    dfr_results = dfr_metric.calculate_dfr(orig_decisions, new_decisions)
    print(f"DFR Results: {dfr_results['flip_rate']:.3f}")
    print(f"Interpretation: {dfr_results['interpretation']}")


if __name__ == "__main__":
    main()
