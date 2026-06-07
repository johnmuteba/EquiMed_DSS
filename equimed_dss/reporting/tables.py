"""Convert EquiMed-DSS statistics result dicts into tidy DataFrames.

Pure presentation layer: these functions reshape the dicts already returned by
`equimed_dss.statistics` and `equimed_dss.geographic`; they do not recompute
any statistic.
"""
from typing import Any, Dict

import pandas as pd


def hierarchical_coefficients_table(results: Dict[str, Any]) -> pd.DataFrame:
    """Variance components and fit statistics from HierarchicalLinearModeling."""
    rows = [
        ("ICC", results.get("icc")),
        ("variance_between_groups", results.get("variance_between_groups")),
        ("variance_within_groups", results.get("variance_within_groups")),
        ("total_variance", results.get("total_variance")),
        ("r_squared_marginal", results.get("r_squared_marginal")),
        ("AIC", results.get("aic")),
        ("BIC", results.get("bic")),
        ("n_groups", results.get("n_groups")),
        ("n_observations", results.get("n_observations")),
    ]
    return pd.DataFrame(rows, columns=["term", "value"])


def mediation_effects_table(results: Dict[str, Any]) -> pd.DataFrame:
    """Total/direct/indirect effects from MediationAnalysis.

    proportion_mediated is reported as-is; a flag in df.attrs marks the case
    where it falls outside [0, 1] (competitive/unstable mediation).
    """
    pm = results.get("proportion_mediated", float("nan"))
    mtype = results.get("interpretation", {}).get("mediation_type", "")
    rows = [
        ("total", results.get("total_effect"), None, None, None, ""),
        ("direct", results.get("direct_effect"), None, None, None, ""),
        (
            "indirect",
            results.get("indirect_effect"),
            results.get("indirect_ci_lower"),
            results.get("indirect_ci_upper"),
            pm,
            mtype,
        ),
    ]
    df = pd.DataFrame(
        rows,
        columns=[
            "effect", "estimate", "ci_lower", "ci_upper",
            "proportion_mediated", "classification",
        ],
    )
    in_range = isinstance(pm, (int, float)) and 0.0 <= pm <= 1.0
    df.attrs["proportion_mediated_flag"] = (
        "" if in_range else "out-of-range (competitive/unstable mediation)"
    )
    return df


def network_centrality_table(results: Dict[str, Any]) -> pd.DataFrame:
    """One row per node with degree/betweenness/closeness/clustering."""
    deg = results.get("degree_centrality", {})
    bet = results.get("betweenness_centrality", {})
    clo = results.get("closeness_centrality", {})
    clu = results.get("clustering_coefficients", {})
    nodes = sorted(deg)
    return pd.DataFrame(
        {
            "node": nodes,
            "degree": [deg.get(n) for n in nodes],
            "betweenness": [bet.get(n) for n in nodes],
            "closeness": [clo.get(n) for n in nodes],
            "clustering": [clu.get(n) for n in nodes],
        }
    )


def geographic_table(result: Dict[str, Any]) -> pd.DataFrame:
    """Return the per-region table from a BEMI or GCC result."""
    return result["per_region"].copy()
