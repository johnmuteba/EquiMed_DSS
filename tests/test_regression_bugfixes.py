"""Regression tests for the four v1.5.0 correctness fixes.

Each test pins the *corrected* behaviour so the bug cannot silently return:

1. DecisionFlipRate  - Wilson 95% CI (not a percentile of the 0/1 vector).
2. JensenShannonDivergence - both implementations return the SAME quantity
   (JS divergence, base 2, range [0, 1]).
3. HarmAdjustedFairnessGap - `hafg` is normalized to [0, 1] (|H1-H2|/max).
4. IntersectionalBiasScore.interaction_analysis - does not mutate the caller's
   DataFrame; plus the ddof=1 sample-SD fix in Bland-Altman and TFD.
"""
import numpy as np
import pandas as pd
import pytest

from equimed_dss.domain1 import DecisionFlipRate, InterRaterReliability
from equimed_dss.domain2 import HarmAdjustedFairnessGap, IntersectionalBiasScore
from equimed_dss.domain3 import TemporalFairnessDrift
from equimed_dss.appendix.advanced_metrics import JensenShannonDivergence
from equimed_dss.appendix.info_theory import AdvancedInfoTheoryMetrics


# ----------------------------------------------------------------------
# Bug 1: DecisionFlipRate confidence interval
# ----------------------------------------------------------------------
class TestDFRWilsonInterval:
    def test_ci_is_not_the_degenerate_zero_one_interval(self):
        # 2 flips in 10. The OLD bug returned np.percentile(flips, 2.5/97.5),
        # i.e. (0.0, 1.0). The Wilson interval must be strictly inside (0, 1).
        res = DecisionFlipRate().calculate_dfr([0] * 8 + [1, 1], [0] * 10)
        assert res["flip_rate"] == pytest.approx(0.20)
        assert res["n_flipped"] == 2 and res["n_samples"] == 10
        assert res["ci_lower"] > 0.01, "lower bound collapsed to 0 (old bug)"
        assert res["ci_upper"] < 0.99, "upper bound collapsed to 1 (old bug)"
        assert res["ci_lower"] < res["flip_rate"] < res["ci_upper"]

    def test_wilson_values_match_closed_form(self):
        res = DecisionFlipRate().calculate_dfr([0] * 8 + [1, 1], [0] * 10)
        # Closed-form Wilson for x=2, n=10, z=1.96
        x, n, z = 2, 10, 1.96
        p = x / n
        denom = 1 + z**2 / n
        centre = (p + z**2 / (2 * n)) / denom
        half = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
        assert res["ci_lower"] == pytest.approx(centre - half, abs=1e-9)
        assert res["ci_upper"] == pytest.approx(centre + half, abs=1e-9)

    def test_boundary_all_flips(self):
        res = DecisionFlipRate().calculate_dfr([0] * 10, [1] * 10)
        assert res["flip_rate"] == 1.0
        assert res["ci_upper"] == pytest.approx(1.0)
        assert res["ci_lower"] < 1.0  # not degenerate


# ----------------------------------------------------------------------
# Bug 2: JSD consistency (divergence, base 2, [0, 1])
# ----------------------------------------------------------------------
class TestJSDConsistency:
    @pytest.mark.parametrize(
        "p,q",
        [([0.7, 0.2, 0.1], [0.1, 0.2, 0.7]), ([0.5, 0.5], [0.9, 0.1])],
    )
    def test_both_implementations_agree(self, p, q):
        a = JensenShannonDivergence().calculate_jsd(np.array(p), np.array(q))["jsd"]
        b = AdvancedInfoTheoryMetrics().calculate_jsd(p, q)
        assert a == pytest.approx(b, abs=1e-12)

    def test_divergence_base2_endpoints(self):
        jsd = JensenShannonDivergence()
        assert jsd.calculate_jsd([0.25] * 4, [0.25] * 4)["jsd"] == pytest.approx(0.0, abs=1e-9)
        # Orthogonal distributions -> JS divergence (base 2) == 1.0
        orth = jsd.calculate_jsd([1.0, 0.0], [0.0, 1.0])
        assert orth["jsd"] == pytest.approx(1.0, abs=1e-9)
        assert 0.0 <= orth["jsd"] <= 1.0

    def test_distance_is_sqrt_of_divergence(self):
        r = JensenShannonDivergence().calculate_jsd([0.7, 0.2, 0.1], [0.1, 0.2, 0.7])
        assert r["jsd_distance"] == pytest.approx(np.sqrt(r["jsd"]), abs=1e-12)


# ----------------------------------------------------------------------
# Bug 3: HAFG normalization
# ----------------------------------------------------------------------
class TestHAFGNormalized:
    def test_hafg_in_unit_interval_and_matches_formula(self):
        hafg = HarmAdjustedFairnessGap(cost_fn=10.0, cost_fp=3.0)
        r = hafg.calculate_hafg({"fn": 5, "fp": 10}, {"fn": 2, "fp": 5})
        h1, h2 = 5 * 10 + 10 * 3, 2 * 10 + 5 * 3  # 80, 35
        assert 0.0 <= r["hafg"] <= 1.0
        assert r["hafg"] == pytest.approx(abs(h1 - h2) / max(h1, h2))  # 45/80 = 0.5625
        assert r["absolute_harm_gap"] == pytest.approx(abs(h1 - h2))   # raw gap retained

    def test_equal_harm_is_zero(self):
        r = HarmAdjustedFairnessGap().calculate_hafg({"fn": 3, "fp": 4}, {"fn": 3, "fp": 4})
        assert r["hafg"] == pytest.approx(0.0)


# ----------------------------------------------------------------------
# Bug 4: IBS does not mutate input; ddof=1 sample SD
# ----------------------------------------------------------------------
class TestIBSAndSampleSD:
    def test_interaction_analysis_does_not_mutate_input(self):
        df = pd.DataFrame(
            {
                "race": ["A", "A", "B", "B"],
                "gender": ["M", "F", "M", "F"],
                "score": [0.8, 0.7, 0.5, 0.6],
            }
        )
        cols_before = list(df.columns)
        IntersectionalBiasScore().interaction_analysis(df)
        assert list(df.columns) == cols_before
        assert "race_gender" not in df.columns  # temporary column did not leak

    def test_bland_altman_uses_sample_sd(self):
        # Two raters, differences = [1, 1, 4] -> sample SD (ddof=1), not population.
        matrix = np.array([[3.0, 2.0], [5.0, 4.0], [6.0, 2.0]])
        res = InterRaterReliability().bland_altman_analysis(matrix)["Judge1-Judge2"]
        diffs = np.array([1.0, 1.0, 4.0])
        assert res["std_difference"] == pytest.approx(np.std(diffs, ddof=1))

    def test_tfd_uses_sample_sd(self):
        series = [0.85, 0.84, 0.86, 0.83, 0.75]
        res = TemporalFairnessDrift().calculate_drift(series)
        expected_sd = np.std(np.array(series), ddof=1)
        assert res["std_pdi"] == pytest.approx(expected_sd)
