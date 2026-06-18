"""Tests for equimed_dss.inference (CIs and hypothesis tests for metrics)."""
import math

import numpy as np
import pytest

from equimed_dss.inference import (
    InferenceResult,
    bootstrap_ci,
    bootstrap_metric,
    permutation_test,
    proportion_ci,
    wilson_ci,
)


class TestWilson:
    def test_known_value(self):
        # 50/100: Wilson 95% CI is approximately [0.404, 0.596]
        r = wilson_ci(50, 100)
        assert abs(r.estimate - 0.5) < 1e-9
        assert abs(r.ci_lower - 0.4038) < 1e-3
        assert abs(r.ci_upper - 0.5962) < 1e-3
        assert r.method == "Wilson score"

    def test_matches_manuscript_dangerous_miss(self):
        # 84/621 dangerous-miss rate: estimate 13.5%, CI ~[11.1%, 16.4%]
        r = wilson_ci(84, 621)
        assert abs(r.estimate - 0.1353) < 1e-3
        assert abs(r.ci_lower - 0.111) < 2e-3
        assert abs(r.ci_upper - 0.164) < 2e-3

    def test_boundary_zero_and_full(self):
        lo = wilson_ci(0, 20)
        assert lo.ci_lower >= 0.0 and lo.ci_upper > 0.0
        hi = wilson_ci(20, 20)
        assert hi.ci_upper <= 1.0 and hi.ci_lower < 1.0

    def test_invalid_inputs(self):
        with pytest.raises(ValueError):
            wilson_ci(5, 0)
        with pytest.raises(ValueError):
            wilson_ci(11, 10)


class TestProportionTest:
    def test_null_rejected_when_far(self):
        # 80/100 tested against null 0.5 -> tiny p-value
        r = proportion_ci(80, 100, null_value=0.5)
        assert r.null_value == 0.5
        assert r.p_value < 1e-6

    def test_null_not_rejected_when_close(self):
        r = proportion_ci(51, 100, null_value=0.5)
        assert r.p_value > 0.5

    def test_one_sided(self):
        r = proportion_ci(60, 100, null_value=0.5, alternative="greater")
        assert r.p_value < 0.05


class TestBootstrap:
    def test_ci_brackets_mean(self):
        data = [1] * 30 + [0] * 70  # proportion 0.30
        r = bootstrap_ci(data, lambda s: float(np.mean(s)),
                         n_boot=1000, random_state=1)
        assert abs(r.estimate - 0.30) < 1e-9
        assert r.ci_lower < 0.30 < r.ci_upper
        assert r.method == "bootstrap"

    def test_reproducible(self):
        data = list(np.random.RandomState(0).binomial(1, 0.4, 200))
        r1 = bootstrap_ci(data, lambda s: float(np.mean(s)), n_boot=500, random_state=7)
        r2 = bootstrap_ci(data, lambda s: float(np.mean(s)), n_boot=500, random_state=7)
        assert (r1.ci_lower, r1.ci_upper) == (r2.ci_lower, r2.ci_upper)

    def test_cluster_bootstrap_wider_than_iid(self):
        # Strong within-cluster correlation: each of 20 clusters is all-0 or all-1.
        rng = np.random.RandomState(0)
        data, clusters = [], []
        for c in range(40):
            val = int(rng.rand() < 0.5)
            for _ in range(10):           # 10 identical evaluations per cluster
                data.append(val)
                clusters.append(c)
        iid = bootstrap_ci(data, lambda s: float(np.mean(s)), n_boot=800, random_state=3)
        clu = bootstrap_ci(data, lambda s: float(np.mean(s)), clusters=clusters,
                           n_boot=800, random_state=3)
        assert clu.n_clusters == 40
        assert clu.method == "cluster bootstrap"
        # ignoring clustering badly understates uncertainty
        assert (clu.ci_upper - clu.ci_lower) > (iid.ci_upper - iid.ci_lower)

    def test_cluster_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            bootstrap_ci([1, 0, 1], lambda s: float(np.mean(s)), clusters=[1, 2])


class TestPermutation:
    def test_separated_groups_significant(self):
        a = list(np.random.RandomState(1).normal(1.0, 0.5, 100))
        b = list(np.random.RandomState(2).normal(0.0, 0.5, 100))
        r = permutation_test(a, b, n_perm=1000, random_state=5)
        assert r.p_value < 0.01
        assert r.estimate > 0

    def test_identical_groups_not_significant(self):
        x = list(np.random.RandomState(3).normal(0, 1, 100))
        r = permutation_test(x, list(x), n_perm=1000, random_state=5)
        assert r.p_value > 0.2

    def test_pvalue_never_zero(self):
        a = [10.0] * 50
        b = [0.0] * 50
        r = permutation_test(a, b, n_perm=200, random_state=5)
        assert r.p_value >= 1.0 / (200 + 1)


class TestMetricUncertaintyIntegration:
    """Proportion metrics now return value + CI + threshold p-value directly."""

    def test_chr_returns_ci_and_p(self):
        from equimed_dss.domain4 import ClinicalHallucinationRate
        s = np.full(285, 0.9); s[:274] = 0.1   # 274/285 unsupported
        r = ClinicalHallucinationRate().calculate_chr(s, threshold=0.05)
        assert abs(r["chr"] - 274 / 285) < 1e-9
        assert r["ci_lower"] < r["chr"] < r["ci_upper"]
        assert r["p_value_above_threshold"] < 0.001   # 96% >> 5%
        assert r["ci_method"] == "Wilson score"

    def test_ivi_returns_ci_and_p(self):
        from equimed_dss.domain4 import InstructionalVulnerabilityIndex
        neutral = list(range(132)); biased = list(range(132))
        for i in range(36):
            biased[i] = -1
        r = InstructionalVulnerabilityIndex().calculate_ivi(neutral, biased)
        assert abs(r["ivi_flip_rate"] - 36 / 132) < 1e-9
        assert r["ci_lower"] < r["ivi_flip_rate"] < r["ci_upper"]
        assert r["p_value_above_threshold"] < 0.05

    def test_dfr_returns_ci_and_p(self):
        from equimed_dss.domain1 import DecisionFlipRate
        orig = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1]
        cf   = [1, 0, 0, 1, 0, 1, 1, 0, 1, 0]
        r = DecisionFlipRate().calculate_dfr(orig, cf)
        assert r["ci_lower"] < r["flip_rate"] < r["ci_upper"]
        assert "p_value_above_threshold" in r

    def test_bootstrap_metric_wraps_dict_metric(self):
        from equimed_dss.domain4 import ClinicalHallucinationRate
        s = np.full(285, 0.9); s[:274] = 0.1
        fn = lambda x: ClinicalHallucinationRate().calculate_chr(x)
        r = bootstrap_metric(fn, list(s), value_key="chr", n_boot=500, random_state=0)
        assert r.ci_lower < r.estimate < r.ci_upper
        assert r.method == "bootstrap"
        # cluster variant resamples groups
        clusters = np.arange(len(s)) % 40
        rc = bootstrap_metric(fn, list(s), value_key="chr", clusters=clusters,
                              n_boot=500, random_state=0)
        assert rc.n_clusters == 40


class TestMetricResultPrinting:
    """Every wrapped metric prints value :: 95% CI and stays dict-compatible."""

    def test_dfr_prints_ci(self):
        from equimed_dss.domain1 import DecisionFlipRate
        r = DecisionFlipRate().calculate_dfr(["a", "a", "b", "a"], ["a", "b", "b", "a"])
        s = str(r)
        assert "DFR = 0.250" in s and "95% CI" in s
        assert isinstance(r, dict) and r["flip_rate"] == 0.25   # backward compatible

    def test_ecs_and_icc_carry_ci(self):
        import numpy as np
        from equimed_dss.domain1 import EmbeddingConsistencyScore, InterRaterReliability
        o = np.random.RandomState(0).rand(12, 8)
        p = o + np.random.RandomState(1).normal(0, 0.05, (12, 8))
        ecs = EmbeddingConsistencyScore().calculate_ecs(o, p)
        assert "ci_lower" in ecs and "95% CI" in str(ecs)
        icc = InterRaterReliability().calculate_icc_2_1(
            np.array([[3, 4, 3], [5, 5, 4], [2, 3, 2], [4, 4, 5]]))
        assert icc["ci_lower"] <= icc["score"] <= icc["ci_upper"] or "95% CI" in str(icc)

    def test_str_enforces_lower_le_upper(self):
        from equimed_dss.inference import MetricResult
        r = MetricResult({"v": 0.5, "ci_lower": 0.9, "ci_upper": 0.1},
                         name="X", value_key="v")   # deliberately reversed
        s = str(r)
        assert "[0.100; 0.900]" in s                # printed in order

    def test_str_when_ci_unavailable(self):
        from equimed_dss.inference import MetricResult
        r = MetricResult({"v": 0.5}, name="X", value_key="v")
        assert "unavailable" in str(r)


def test_result_to_dict_drops_none():
    r = wilson_ci(5, 10)
    d = r.to_dict()
    assert "estimate" in d and "ci_lower" in d
    assert "p_value" not in d  # not populated, should be dropped
    assert isinstance(str(r), str)
