"""Render hierarchical / mediation / network results as publication tables.

Uses bundled illustrative sample data. REAL-DATA HOOK: replace the sample
DataFrames with your real result frames (or load a saved result dict) to render
the actual manuscript tables for slides.
"""
import numpy as np
import pandas as pd

from equimed_dss.statistics import (
    HierarchicalLinearModeling,
    MediationAnalysis,
    NetworkStatistics,
)
from equimed_dss.reporting import (
    export_table,
    hierarchical_coefficients_table,
    mediation_effects_table,
    network_centrality_table,
)


def _sample_hierarchical_df(rng):
    rows = []
    for g in range(8):
        group_effect = rng.normal(0, 2)
        for _ in range(25):
            x = rng.normal(0, 1)
            rows.append(
                {"group": g, "x": x, "outcome": group_effect + 0.5 * x + rng.normal(0, 1)}
            )
    return pd.DataFrame(rows)


def main():
    rng = np.random.RandomState(42)

    # --- Hierarchical ---
    # fit_model returns the dict (icc, variance components, aic, bic); note
    # calculate_icc returns a bare float, so use fit_model here.
    hlm = HierarchicalLinearModeling()
    df = _sample_hierarchical_df(rng)
    hlm_res = hlm.fit_model(
        df, outcome_var="outcome", level1_predictors=["x"], level2_var="group"
    )
    print("Hierarchical (variance components):")
    print(export_table(hierarchical_coefficients_table(hlm_res), fmt="markdown"))

    # --- Mediation ---  (arg is treatment_var, not independent_var)
    n = 300
    x = rng.normal(0, 1, n)
    m = 0.5 * x + rng.normal(0, 1, n)
    y = 0.3 * x + 0.4 * m + rng.normal(0, 1, n)
    med = MediationAnalysis()
    med_res = med.analyze_mediation(
        pd.DataFrame({"X": x, "M": m, "Y": y}),
        treatment_var="X", mediator_var="M", outcome_var="Y",
    )
    print("\nMediation effects:")
    med_table = mediation_effects_table(med_res)
    print(export_table(med_table, fmt="markdown"))
    if bool(med_table["outside_bounds"].any()):
        print("NOTE: proportion_mediated falls outside [0, 1] "
              "(competitive/unstable mediation)")

    # --- Network ---  (takes a numpy adjacency matrix + node_labels)
    net = NetworkStatistics()
    labels = ["DFR", "ECS", "ICC", "HER"]
    adjacency = np.array(
        [
            [0.0, 0.6, 0.2, 0.0],
            [0.6, 0.0, 0.0, 0.5],
            [0.2, 0.0, 0.0, 0.0],
            [0.0, 0.5, 0.0, 0.0],
        ]
    )
    net_res = net.analyze_network(adjacency, node_labels=labels)
    print("\nNetwork centralities:")
    print(export_table(network_centrality_table(net_res), fmt="markdown"))


if __name__ == "__main__":
    main()
