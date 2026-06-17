"""Tests for equimed_dss.inference (CIs and hypothesis tests for metrics)."""
import math

import numpy as np
import pytest

from equimed_dss.inference import (
    InferenceResult,
    bootstrap_ci,
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


def test_result_to_dict_drops_none():
    r = wilson_ci(5, 10)
    d = r.to_dict()
    assert "estimate" in d and "ci_lower" in d
    assert "p_value" not in d  # not populated, should be dropped
    assert isinstance(str(r), str)
