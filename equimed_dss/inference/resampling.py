"""Uncertainty quantification for EquiMed-DSS metrics.

Most fairness, reliability, and geographic metrics in this library are reported
as single point estimates (a "value-at-risk"-style number). This module turns
any of them into an *interval* estimate with an explicit method, sample size,
and -- optionally -- a hypothesis test, so a metric can be reported the way an
effect size is in clinical research: estimate, 95% CI, and a p-value against a
pre-specified null or acceptability threshold.

Design goals
------------
* numpy + standard-library only (no scipy dependency).
* A single result schema (:class:`InferenceResult`) for every method.
* Honour non-independence: :func:`bootstrap_ci` with ``clusters=...`` resamples
  whole clusters (e.g. patients / ED visits), not individual rows, so repeated
  evaluations of the same patient do not produce falsely narrow intervals.

Examples
--------
>>> from equimed_dss.inference import wilson_ci, bootstrap_ci, permutation_test
>>> r = wilson_ci(84, 621)                 # dangerous-miss rate 84/621
>>> round(r.estimate, 3), round(r.ci_lower, 3), round(r.ci_upper, 3)
(0.135, 0.111, 0.164)
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from statistics import NormalDist
from typing import Callable, Optional, Sequence

import numpy as np

__all__ = [
    "InferenceResult",
    "wilson_ci",
    "proportion_ci",
    "bootstrap_ci",
    "permutation_test",
]


def _z(conf: float) -> float:
    """Two-sided standard-normal quantile for a confidence level (e.g. 0.95)."""
    if not 0 < conf < 1:
        raise ValueError("conf must be in (0, 1)")
    return NormalDist().inv_cdf(0.5 + conf / 2.0)


@dataclass
class InferenceResult:
    """A metric point estimate together with its uncertainty.

    Only ``estimate``, ``method`` and ``n`` are always present; interval and
    test fields are populated by whichever routine produced the result.
    """

    estimate: float
    method: str
    n: int
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    conf_level: Optional[float] = 0.95
    se: Optional[float] = None
    p_value: Optional[float] = None
    null_value: Optional[float] = None
    n_boot: Optional[int] = None
    n_clusters: Optional[int] = None

    def to_dict(self) -> dict:
        """Drop unpopulated (None) fields for compact JSON export."""
        return {k: v for k, v in asdict(self).items() if v is not None}

    def __str__(self) -> str:
        if self.ci_lower is None:
            ci = ""
        else:
            pct = int(round((self.conf_level or 0.95) * 100))
            ci = f" ({pct}% CI {self.ci_lower:.4g} to {self.ci_upper:.4g})"
        p = "" if self.p_value is None else f", p={self.p_value:.3g}"
        return f"{self.estimate:.4g}{ci} [{self.method}, n={self.n}{p}]"


def wilson_ci(k: int, n: int, conf: float = 0.95) -> InferenceResult:
    """Wilson score interval for a binomial proportion ``k / n``.

    Preferred over the normal (Wald) interval because it is well-behaved near
    0 and 1 and for small ``n``.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= k <= n:
        raise ValueError("require 0 <= k <= n")
    z = _z(conf)
    p = k / n
    den = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / den
    return InferenceResult(
        estimate=p,
        method="Wilson score",
        n=n,
        ci_lower=center - half,
        ci_upper=center + half,
        conf_level=conf,
        se=math.sqrt(p * (1 - p) / n),
    )


def _prop_ztest(k: int, n: int, p0: float, alternative: str) -> float:
    """One-proportion score (z) test p-value against null ``p0``.

    Robust for all ``n`` (no factorial overflow); standard for moderate-to-large
    samples. For very small ``n`` an exact binomial test would be preferable.
    """
    se0 = math.sqrt(p0 * (1 - p0) / n)
    if se0 == 0:
        return float("nan")
    z = (k / n - p0) / se0
    nd = NormalDist()
    if alternative == "greater":
        return 1.0 - nd.cdf(z)
    if alternative == "less":
        return nd.cdf(z)
    return 2.0 * (1.0 - nd.cdf(abs(z)))


def proportion_ci(
    k: int,
    n: int,
    conf: float = 0.95,
    null_value: Optional[float] = None,
    alternative: str = "two-sided",
) -> InferenceResult:
    """Wilson CI for ``k / n`` plus an optional score test against ``null_value``.

    Use ``null_value`` to test a metric against a pre-specified acceptability
    threshold (e.g. an instruction-vulnerability rate against a tolerated 5%).
    """
    res = wilson_ci(k, n, conf)
    if null_value is not None:
        res.null_value = null_value
        res.p_value = _prop_ztest(k, n, null_value, alternative)
    return res


def bootstrap_ci(
    data: Sequence,
    statistic: Callable[[Sequence], float],
    conf: float = 0.95,
    n_boot: int = 2000,
    clusters: Optional[Sequence] = None,
    random_state: Optional[int] = None,
) -> InferenceResult:
    """Percentile bootstrap CI for ``statistic(data)``.

    Parameters
    ----------
    data : sequence of records (list / array / list of dicts).
    statistic : callable mapping a resampled subset of ``data`` to a float.
    clusters : optional labels (same length as ``data``); when supplied, whole
        clusters are resampled with replacement (cluster / visit bootstrap),
        giving honest intervals under within-cluster correlation.
    random_state : seed for reproducibility.
    """
    rng = np.random.default_rng(random_state)
    data = list(data)
    n = len(data)
    if n == 0:
        raise ValueError("data is empty")
    est = float(statistic(data))
    boots = []
    n_clusters = None
    if clusters is not None:
        clusters = list(clusters)
        if len(clusters) != n:
            raise ValueError("clusters must have the same length as data")
        groups: dict = {}
        for i, c in enumerate(clusters):
            groups.setdefault(c, []).append(i)
        keys = list(groups.keys())
        n_clusters = len(keys)
        for _ in range(n_boot):
            chosen = rng.integers(0, n_clusters, size=n_clusters)
            idx = [i for c in chosen for i in groups[keys[c]]]
            boots.append(statistic([data[i] for i in idx]))
        method = "cluster bootstrap"
    else:
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            boots.append(statistic([data[i] for i in idx]))
        method = "bootstrap"
    a = (1 - conf) / 2
    lo, hi = np.percentile(boots, [100 * a, 100 * (1 - a)])
    return InferenceResult(
        estimate=est,
        method=method,
        n=n,
        ci_lower=float(lo),
        ci_upper=float(hi),
        conf_level=conf,
        se=float(np.std(boots, ddof=1)),
        n_boot=n_boot,
        n_clusters=n_clusters,
    )


def permutation_test(
    a: Sequence[float],
    b: Sequence[float],
    statistic: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    n_perm: int = 2000,
    alternative: str = "two-sided",
    random_state: Optional[int] = None,
) -> InferenceResult:
    """Permutation test for a difference in ``statistic`` between groups ``a`` and ``b``.

    Default statistic is the difference in means (``mean(a) - mean(b)``). The
    p-value uses the add-one estimator ``(count + 1) / (n_perm + 1)`` so it is
    never exactly zero. Use this for fairness gaps between demographic groups.
    """
    rng = np.random.default_rng(random_state)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size == 0 or b.size == 0:
        raise ValueError("both groups must be non-empty")
    if statistic is None:
        statistic = lambda x, y: float(np.mean(x) - np.mean(y))
    obs = float(statistic(a, b))
    pooled = np.concatenate([a, b])
    na = a.size
    count = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        d = statistic(pooled[:na], pooled[na:])
        if alternative == "two-sided":
            count += abs(d) >= abs(obs) - 1e-12
        elif alternative == "greater":
            count += d >= obs - 1e-12
        else:
            count += d <= obs + 1e-12
    p = (count + 1) / (n_perm + 1)
    return InferenceResult(
        estimate=obs,
        method="permutation test",
        n=int(a.size + b.size),
        p_value=float(p),
        n_boot=n_perm,
    )
