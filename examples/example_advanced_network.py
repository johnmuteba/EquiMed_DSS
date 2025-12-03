import numpy as np
import pandas as pd

from equimed_dss.appendix import AdvancedNetworkAnalysis
from equimed_dss.utils import plot_correlation_matrix, plot_network_graph


def main():
    print("=== Advanced Network Analysis & Text Handling ===")

    ana = AdvancedNetworkAnalysis()

    # 1. Metric Correlation Network (RQ16)
    print("\n[1] Metric Correlation Network")
    # Simulate metric scores for 10 models
    metrics_data = {
        "ICC": np.random.uniform(0.5, 0.9, 10),
        "HER": np.random.uniform(0.8, 1.2, 10),
        "TFD": np.random.uniform(0, 0.2, 10),
        "GCI": np.random.uniform(0.6, 1.0, 10),
    }
    metric_df = pd.DataFrame(metrics_data)

    corr_res = ana.metric_correlation_network(metric_df, threshold=0.3)
    print(f"Interpretation: {corr_res['interpretation']}")
    plot_correlation_matrix(
        corr_res["correlation_matrix"], save_path="metric_correlation.png"
    )
    print("Plot saved to metric_correlation.png")
    plot_network_graph(
        corr_res["graph"],
        title="Metric Correlation Network",
        save_path="metric_network.png",
    )
    print("Plot saved to metric_network.png")

    # 2. Subgroup Similarity (RQ17)
    print("\n[2] Subgroup Similarity Graph")
    vectors = {
        "GroupA": np.array([1, 0, 1]),
        "GroupB": np.array([1, 0.1, 0.9]),
        "GroupC": np.array([0, 1, 0]),  # Dissimilar
    }
    sim_res = ana.subgroup_similarity_graph(vectors, threshold=0.5)
    print(f"Interpretation: {sim_res['interpretation']}")
    plot_network_graph(
        sim_res["graph"],
        title="Subgroup Similarity",
        save_path="subgroup_similarity.png",
    )
    print("Plot saved to subgroup_similarity.png")

    # 3. Text Data Handling (RQ18)
    print("\n[3] Text Data Handling (Concept Co-occurrence)")
    print("Note: EquiMed_DSS works with structured outputs (embeddings/counts).")
    print("For raw text, you first extract concepts or embeddings.")

    raw_texts = [
        "Patient has hypertension and diabetes.",
        "Diabetes management for elderly patient.",
        "Hypertension risk factors include smoking.",
        "Smoking cessation programs.",
    ]
    concepts = ["hypertension", "diabetes", "smoking", "elderly"]

    cooc_res = ana.concept_cooccurrence_network(raw_texts, concepts)
    print(f"Co-occurrence Matrix:\n{cooc_res['cooccurrence_matrix']}")
    print(f"Interpretation: {cooc_res['interpretation']}")
    plot_network_graph(
        cooc_res["graph"],
        title="Concept Co-occurrence",
        save_path="concept_cooccurrence.png",
    )
    print("Plot saved to concept_cooccurrence.png")


if __name__ == "__main__":
    main()
