"""Convert EquiMed-DSS result dicts into tidy, publication-ready DataFrames.

Pure presentation layer: these functions reshape the dicts already returned by
``equimed_dss.statistics`` and ``equimed_dss.geographic``; they do not recompute
any statistic. Each accepts a ``decimals`` argument for rounding.
"""
from typing import Any, Dict

import pandas as pd


def _round(df: pd.DataFrame, decimals: int) -> pd.DataFrame:
    """Round numeric columns of a copy of ``df`` to ``decimals`` places."""
    out = df.copy()
    num_cols = out.select_dtypes(include="number").columns
    out[num_cols] = out[num_cols].round(decimals)
    return out


def hierarchical_coefficients_table(
    hlm_result: Dict[str, Any], decimals: int = 3
) -> pd.DataFrame:
    """Fixed-effect coefficient table from ``HierarchicalLinearModeling.fit_model``.

    One row per fixed effect with estimate, standard error, t-value, p-value and
    the 95% confidence interval. If the result lacks per-coefficient statistics
    (e.g. the simple-variance fallback path), falls back to a tidy variance-
    components table (term, value).
    """
    coeffs = hlm_result.get("coefficients")
    if coeffs:
        df = pd.DataFrame(
            coeffs,
            columns=[
                "term", "estimate", "std_err", "t", "p_value",
                "ci_lower", "ci_upper",
            ],
        )
        return _round(df, decimals)

    # Fallback: variance-components summary when no coefficient stats are present.
    rows = [
        ("ICC", hlm_result.get("icc")),
        ("variance_between_groups", hlm_result.get("variance_between_groups")),
        ("variance_within_groups", hlm_result.get("variance_within_groups")),
        ("total_variance", hlm_result.get("total_variance")),
        ("r_squared_marginal", hlm_result.get("r_squared_marginal")),
        ("AIC", hlm_result.get("aic")),
        ("BIC", hlm_result.get("bic")),
        ("n_groups", hlm_result.get("n_groups")),
        ("n_observations", hlm_result.get("n_observations")),
    ]
    return _round(pd.DataFrame(rows, columns=["term", "value"]), decimals)


def mediation_effects_table(
    mediation_result: Dict[str, Any], decimals: int = 3
) -> pd.DataFrame:
    """Direct / indirect / total effects from ``MediationAnalysis``.

    ``proportion_mediated`` is reported unclamped; the boolean column
    ``outside_bounds`` is True when it falls outside [0, 1] (competitive or
    unstable mediation).
    """
    pm = mediation_result.get("proportion_mediated", float("nan"))
    mtype = mediation_result.get("interpretation", {}).get("mediation_type", "")
    in_range = isinstance(pm, (int, float)) and 0.0 <= pm <= 1.0
    outside = not in_range

    rows = [
        ("direct", mediation_result.get("direct_effect"), None, None, None, False, ""),
        (
            "indirect",
            mediation_result.get("indirect_effect"),
            mediation_result.get("indirect_ci_lower"),
            mediation_result.get("indirect_ci_upper"),
            pm,
            outside,
            mtype,
        ),
        ("total", mediation_result.get("total_effect"), None, None, None, False, ""),
    ]
    df = pd.DataFrame(
        rows,
        columns=[
            "effect", "estimate", "ci_lower", "ci_upper",
            "proportion_mediated", "outside_bounds", "classification",
        ],
    )
    return _round(df, decimals)


def network_centrality_table(
    network_result: Dict[str, Any], decimals: int = 3
) -> pd.DataFrame:
    """One row per node: degree, betweenness, closeness, clustering."""
    deg = network_result.get("degree_centrality", {})
    bet = network_result.get("betweenness_centrality", {})
    clo = network_result.get("closeness_centrality", {})
    clu = network_result.get("clustering_coefficients", {})
    nodes = sorted(deg)
    df = pd.DataFrame(
        {
            "node": nodes,
            "degree": [deg.get(n) for n in nodes],
            "betweenness": [bet.get(n) for n in nodes],
            "closeness": [clo.get(n) for n in nodes],
            "clustering": [clu.get(n) for n in nodes],
        }
    )
    return _round(df, decimals)


def geographic_table(
    bemi_result: Dict[str, Any],
    gcc_result: Dict[str, Any],
    decimals: int = 3,
) -> pd.DataFrame:
    """Combine BEMI and GCC results into a single tidy summary DataFrame.

    One row per metric (metric, value), covering the burden-evidence mismatch
    and the two concentration descriptors, suitable for a manuscript table.
    """
    def fmt(v):
        # Round floats; leave ints/strings (e.g. region name) untouched. The
        # value column is mixed-type, so round per-value rather than per-column.
        return round(v, decimals) if isinstance(v, float) else v

    rows = [
        ("BEMI", fmt(bemi_result.get("bemi"))),
        ("Gini* (G*)", fmt(gcc_result.get("gini_corrected"))),
        ("H_norm", fmt(gcc_result.get("entropy_normalized"))),
        ("concentration (1 - H_norm)", fmt(gcc_result.get("concentration"))),
        ("most_underserved_region", bemi_result.get("most_underserved_region")),
        ("n_regions", gcc_result.get("n_regions")),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])
