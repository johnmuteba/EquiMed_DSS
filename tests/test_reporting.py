"""Tests for the reporting/table layer."""
import numpy as np
import pandas as pd
import pytest

from equimed_dss.reporting import (
    hierarchical_coefficients_table,
    mediation_effects_table,
    network_centrality_table,
    geographic_table,
)


def test_hierarchical_table_shape():
    results = {
        "icc": 0.12, "variance_between_groups": 0.5, "variance_within_groups": 3.5,
        "total_variance": 4.0, "r_squared_marginal": 0.05, "aic": 100.0,
        "bic": 110.0, "n_groups": 10, "n_observations": 200,
    }
    df = hierarchical_coefficients_table(results)
    assert list(df.columns) == ["term", "value"]
    assert "ICC" in set(df["term"])
    assert df.loc[df["term"] == "ICC", "value"].iloc[0] == 0.12


def test_mediation_table_rows_and_flag():
    results = {
        "total_effect": 1.0, "direct_effect": 0.4, "indirect_effect": 0.6,
        "proportion_mediated": 0.6, "indirect_ci_lower": 0.2,
        "indirect_ci_upper": 0.9,
        "interpretation": {"mediation_type": "Partial mediation (complementary)"},
    }
    df = mediation_effects_table(results)
    assert list(df["effect"]) == ["total", "direct", "indirect"]
    assert df.loc[df["effect"] == "indirect", "proportion_mediated"].iloc[0] == 0.6
    assert df.attrs["proportion_mediated_flag"] == ""


def test_mediation_table_flags_out_of_range_pm():
    results = {
        "total_effect": 0.1, "direct_effect": -0.5, "indirect_effect": 0.6,
        "proportion_mediated": 6.0, "indirect_ci_lower": 0.2,
        "indirect_ci_upper": 0.9,
        "interpretation": {"mediation_type": "Partial mediation (competitive)"},
    }
    df = mediation_effects_table(results)
    assert "out-of-range" in df.attrs["proportion_mediated_flag"]


def test_network_table_one_row_per_node():
    results = {
        "degree_centrality": {"a": 1.0, "b": 0.5},
        "betweenness_centrality": {"a": 0.0, "b": 0.0},
        "closeness_centrality": {"a": 1.0, "b": 0.66},
        "clustering_coefficients": {"a": 0.0, "b": 0.0},
    }
    df = network_centrality_table(results)
    assert list(df.columns) == ["node", "degree", "betweenness", "closeness", "clustering"]
    assert len(df) == 2


def test_geographic_table_passthrough():
    per_region = pd.DataFrame({"region": ["A", "B"], "evidence_share": [0.6, 0.4]})
    df = geographic_table({"per_region": per_region})
    assert df.equals(per_region)
