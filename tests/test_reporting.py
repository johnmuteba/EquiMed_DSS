"""Tests for the reporting/table layer."""
import numpy as np
import pandas as pd
import pytest

from equimed_dss.reporting import (
    hierarchical_coefficients_table,
    mediation_effects_table,
    network_centrality_table,
    geographic_table,
    export_table,
)


def test_hierarchical_coefficient_table_shape():
    result = {
        "coefficients": [
            {"term": "Intercept", "estimate": -0.21, "std_err": 0.56,
             "t": -0.37, "p_value": 0.71, "ci_lower": -1.31, "ci_upper": 0.90},
            {"term": "x", "estimate": 0.40, "std_err": 0.07,
             "t": 5.99, "p_value": 0.0, "ci_lower": 0.27, "ci_upper": 0.53},
        ]
    }
    df = hierarchical_coefficients_table(result)
    assert list(df.columns) == [
        "term", "estimate", "std_err", "t", "p_value", "ci_lower", "ci_upper"
    ]
    assert set(df["term"]) == {"Intercept", "x"}


def test_hierarchical_table_fallback_to_variance_components():
    # No "coefficients" key -> variance-components fallback.
    df = hierarchical_coefficients_table({"icc": 0.12, "aic": 100.0, "bic": 110.0})
    assert list(df.columns) == ["term", "value"]
    assert "ICC" in set(df["term"])


def test_mediation_table_outside_bounds_false():
    result = {
        "total_effect": 1.0, "direct_effect": 0.4, "indirect_effect": 0.6,
        "proportion_mediated": 0.6, "indirect_ci_lower": 0.2,
        "indirect_ci_upper": 0.9,
        "interpretation": {"mediation_type": "Partial mediation (complementary)"},
    }
    df = mediation_effects_table(result)
    assert list(df["effect"]) == ["direct", "indirect", "total"]
    assert "outside_bounds" in df.columns
    assert df.loc[df["effect"] == "indirect", "outside_bounds"].iloc[0] == False  # noqa: E712


def test_mediation_table_outside_bounds_true():
    result = {
        "total_effect": 0.1, "direct_effect": -0.5, "indirect_effect": 0.6,
        "proportion_mediated": 6.0, "indirect_ci_lower": 0.2,
        "indirect_ci_upper": 0.9,
        "interpretation": {"mediation_type": "Partial mediation (competitive)"},
    }
    df = mediation_effects_table(result)
    assert df.loc[df["effect"] == "indirect", "outside_bounds"].iloc[0] == True  # noqa: E712


def test_network_table_one_row_per_node():
    result = {
        "degree_centrality": {"a": 1.0, "b": 0.5},
        "betweenness_centrality": {"a": 0.0, "b": 0.0},
        "closeness_centrality": {"a": 1.0, "b": 0.66},
        "clustering_coefficients": {"a": 0.0, "b": 0.0},
    }
    df = network_centrality_table(result)
    assert list(df.columns) == ["node", "degree", "betweenness", "closeness", "clustering"]
    assert len(df) == 2


def test_geographic_table_combines_bemi_and_gcc():
    bemi_result = {"bemi": 0.42, "most_underserved_region": "SEARO"}
    gcc_result = {"gini_corrected": 0.51, "entropy_normalized": 0.84,
                  "concentration": 0.16, "n_regions": 6}
    df = geographic_table(bemi_result, gcc_result)
    assert list(df.columns) == ["metric", "value"]
    metrics = set(df["metric"])
    assert "BEMI" in metrics and "Gini* (G*)" in metrics and "H_norm" in metrics


def test_export_markdown_nonempty():
    df = pd.DataFrame({"term": ["ICC"], "value": [0.123456]})
    out = export_table(df, fmt="markdown", decimals=3)
    assert "ICC" in out and "0.123" in out


def test_export_latex_nonempty():
    df = pd.DataFrame({"term": ["ICC"], "value": [0.123456]})
    out = export_table(df, fmt="latex", decimals=2)
    assert "tabular" in out and "0.12" in out


def test_export_writes_file(tmp_path):
    df = pd.DataFrame({"term": ["ICC"], "value": [0.5]})
    p = tmp_path / "t.md"
    export_table(df, fmt="markdown", path=str(p))
    assert p.read_text().strip() != ""


def test_export_bad_format_raises():
    with pytest.raises(ValueError):
        export_table(pd.DataFrame({"a": [1]}), fmt="csv")
