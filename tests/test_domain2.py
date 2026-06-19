"""Tests for Domain 2 metrics."""

import numpy as np
import pytest

from equimed_dss.domain2 import (
    EthicalRiskIndex,
    HarmAdjustedFairnessGap,
    HierarchicalEquityRatio,
    IntersectionalBiasScore,
)


class TestHierarchicalEquityRatio:
    """Test suite for HierarchicalEquityRatio."""

    def test_calculate_her_basic(self):
        """Test basic HER calculation."""
        her = HierarchicalEquityRatio()
        scores = {"White": 0.85, "Black": 0.75, "Hispanic": 0.80}

        result = her.calculate_her(scores)

        assert "White" in result
        assert "Black" in result
        assert "Hispanic" in result
        assert result["White"]["score"] == 1.0
        assert 0 < result["Black"]["score"] < 1

    def test_calculate_her_missing_reference(self):
        """Test that missing reference group raises error."""
        her = HierarchicalEquityRatio()
        scores = {"Black": 0.75, "Hispanic": 0.80}

        with pytest.raises(ValueError):
            her.calculate_her(scores, reference_group="White")

    def test_calculate_bias_gini(self):
        """Test Bias-Gini calculation."""
        her = HierarchicalEquityRatio()
        scores = [0.85, 0.75, 0.80, 0.82]

        result = her.calculate_bias_gini(scores)

        assert 0 <= result["bias_gini"] <= 1
        assert isinstance(result["bias_gini"], float)
        # Printing shows the value alongside a 95% CI (v1.9.0 contract).
        assert "95% CI" in str(result)

    def test_calculate_bias_gini_empty(self):
        """Test Bias-Gini with empty scores."""
        her = HierarchicalEquityRatio()
        result = her.calculate_bias_gini([])

        assert result["bias_gini"] == 0.0

    def test_bias_gini_behaves_like_a_scalar(self):
        """Regression: a CI-carrying result still supports scalar-style usage
        (format spec, round, float, comparison) as documented in the vignette."""
        her = HierarchicalEquityRatio()
        gini = her.calculate_bias_gini([0.85, 0.75, 0.80, 0.82])
        # f"{gini:.4f}" must not raise and must format the point value.
        assert f"{gini:.4f}" == f"{gini['bias_gini']:.4f}"
        assert round(gini, 3) == round(gini["bias_gini"], 3)
        assert float(gini) == gini["bias_gini"]
        assert (gini < 0.2) == (gini["bias_gini"] < 0.2)

    def test_her_returns_pure_group_mapping(self):
        """Regression: calculate_her must yield only per-group entries so
        ``for g, r in result.items(): r['score']`` works; the gap/CI are carried
        as the printable point/CI, not as extra dict keys."""
        her = HierarchicalEquityRatio()
        scores = {"White": 0.85, "Black": 0.78, "Hispanic": 0.80, "Asian": 0.87}
        result = her.calculate_her(scores)
        assert set(result.keys()) == set(scores.keys())
        for group, entry in result.items():       # must not raise
            assert isinstance(entry["score"], float)
            assert "verdict" in entry["interpretation"]
        assert "95% CI" in str(result)            # printed scalar = HER gap


class TestHarmAdjustedFairnessGap:
    """Test suite for HarmAdjustedFairnessGap."""

    def test_calculate_hafg_basic(self):
        """Test basic HAFG calculation."""
        hafg = HarmAdjustedFairnessGap()
        group1_errors = {"fn": 5, "fp": 10}
        group2_errors = {"fn": 2, "fp": 5}

        result = hafg.calculate_hafg(group1_errors, group2_errors)

        assert "hafg" in result
        assert "interpretation" in result
        assert isinstance(result["hafg"], float)
        assert result["hafg"] >= 0


class TestEthicalRiskIndex:
    """Test suite for EthicalRiskIndex."""

    def test_calculate_eri_basic(self):
        """Test basic ERI calculation."""
        eri = EthicalRiskIndex()
        violations = [{"severity": 2.5}, {"severity": 1.0}, {"severity": 5.0}]

        result = eri.calculate_eri(violations, n_total_outputs=100)

        assert "eri" in result
        assert "interpretation" in result
        assert isinstance(result["eri"], float)
        assert result["eri"] >= 0

    def test_calculate_eri_no_violations(self):
        """Test ERI with no violations."""
        eri = EthicalRiskIndex()

        result = eri.calculate_eri([], n_total_outputs=100)

        assert result["eri"] == 0.0


class TestIntersectionalBiasScore:
    """Test suite for IntersectionalBiasScore."""

    def test_calculate_subgroup_similarity_basic(self):
        """Test basic IBS calculation."""
        ibs = IntersectionalBiasScore()
        vectors = {
            "GroupA": np.array([0.8, 0.7, 0.9]),
            "GroupB": np.array([0.75, 0.65, 0.85]),
            "GroupC": np.array([0.5, 0.4, 0.6]),
        }

        result = ibs.calculate_subgroup_similarity(vectors)

        assert "outlier_subgroup" in result
        assert "outlier_distance" in result
        assert result["outlier_subgroup"] in vectors.keys()
