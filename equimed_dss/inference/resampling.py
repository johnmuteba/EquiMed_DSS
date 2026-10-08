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
import warnings
from dataclasses import asdict, dataclass
from statistics import NormalDist
from typing import Any, Callable, Dict, Optional, Sequence

import numpy as np

__all__ = [
    "InferenceResult",
    "MetricResult",
    "wilson_ci",
    "proportion_ci",
    "bootstrap_ci",
    "bootstrap_metric",
    "stratified_bootstrap_ci",
    "block_bootstrap_ci",
    "permutation_test",
]


class MetricResult(dict):
    """A metric result that always prints its value with a 95% CI.

    Behaves exactly like the ``dict`` it wraps (so ``result['flip_rate']`` and
    every other key keep working, and it is JSON-serialisable), but its string
    form always shows the point estimate and, when available, the
    $\\alpha=0.05$ confidence interval. Printed bounds are ordered so the
    interval always has lower $\\le$ upper.

    Where a metric is conceptually a single number (e.g. a flip rate or a Gini
    index), the result also behaves like that number in numeric and formatting
    contexts: ``f"{result:.4f}"``, ``round(result, 3)``, ``float(result)`` and
    ``result < 0.2`` all use the point estimate, so existing scalar-style usage
    keeps working even though the object is a dict that prints its CI.

    Parameters
    ----------
    data : the metric's result mapping.
    name : short label shown when printing (e.g. ``"DFR"``).
    value_key : key holding the point estimate (e.g. ``"flip_rate"``).
    point : explicit point estimate, used when the headline value is *not* a
        top-level key (e.g. a mapping-valued result such as HER, whose dict holds
        per-group entries while the printed scalar is the across-group gap).
    ci : explicit ``(lower, upper, method)`` tuple, used the same way as
        ``point`` when the CI is not stored as ``ci_lower`` / ``ci_upper`` keys.
    """

    def __init__(
        self, data=None, *, name="metric", value_key=None, point=None, ci=None
    ):
        super().__init__(data or {})
        self._name = name
        self._value_key = value_key
        self._point_override = point
        self._ci_override = ci

    def _point(self):
        if self._point_override is not None:
            return self._point_override
        return self.get(self._value_key) if self._value_key else None

    def _ci(self):
        """Return ``(lower, upper, method)`` or ``None``."""
        if self._ci_override is not None:
            return self._ci_override
        lo, hi = self.get("ci_lower"), self.get("ci_upper")
        if lo is not None and hi is not None:
            return (lo, hi, self.get("ci_method"))
        return None

    def __str__(self):
        v = self._point()
        head = (
            f"{self._name} = {v:.3f}"
            if isinstance(v, (int, float))
            else f"{self._name} = {v}"
        )
        ci = self._ci()
        if ci is not None:
            lo, hi, method = ci
            lo, hi = sorted((float(lo), float(hi)))  # guarantee lower <= upper
            tail = f"95% CI [{lo:.3f}; {hi:.3f}]"
            if method:
                tail += f" ({method})"
            return f"{head} :: {tail}"
        return f"{head} :: 95% CI unavailable (needs observation-level input)"

    __repr__ = __str__

    # --- graceful scalar behaviour ------------------------------------------
    # A metric whose headline value is a single number should still work in
    # numeric/formatting contexts, so documented scalar-style usage (e.g.
    # ``f"{gini:.4f}"`` or ``round(gini, 3)``) does not break when the result is
    # a CI-carrying dict.
    def __float__(self):
        v = self._point()
        if isinstance(v, (int, float)):
            return float(v)
        raise TypeError(
            f"{self._name} result has no scalar point value to convert to float"
        )

    def __format__(self, spec):
        v = self._point()
        if spec and isinstance(v, (int, float)):
            return format(v, spec)  # e.g. f"{gini:.4f}" -> "0.0247"
        if spec:
            return format(str(self), spec)
        return str(self)

    def __round__(self, ndigits=None):
        return round(float(self), ndigits)

    def __lt__(self, other):
        return float(self) < float(other)

    def __le__(self, other):
        return float(self) <= float(other)

    def __gt__(self, other):
        return float(self) > float(other)

    def __ge__(self, other):
        return float(self) >= float(other)


# Accepted values of ``alternative``; anything else raises (up to 1.9.5 an
# unrecognised value such as "two_sided" was silently treated as "less" by
# permutation_test and as "two-sided" by the score test).
_ALTERNATIVES = ("two-sided", "greater", "less")


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
    n_failed: Optional[int] = None

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
        # The Wilson bounds are exactly 0 at k = 0 and exactly 1 at k = n; set
        # them so, rather than leaving floating-point residue (about 1e-17).
        ci_lower=0.0 if k == 0 else max(0.0, center - half),
        ci_upper=1.0 if k == n else min(1.0, center + half),
        conf_level=conf,
        se=math.sqrt(p * (1 - p) / n),
    )


def _prop_ztest(k: int, n: int, p0: float, alternative: str) -> float:
    """One-proportion score (z) test p-value against null ``p0``.

    Robust for all ``n`` (no factorial overflow); standard for moderate-to-large
    samples. For very small ``n`` an exact binomial test would be preferable.
    """
    if alternative not in _ALTERNATIVES:
        raise ValueError(f"alternative must be one of {_ALTERNATIVES}")
    if not 0 < p0 < 1:
        raise ValueError("the null proportion must lie strictly between 0 and 1")
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
    _check_resampling(conf, n_boot)
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


def _check_resampling(conf: float, n_boot: int) -> None:
    if not 0 < conf < 1:
        raise ValueError("conf must be in (0, 1)")
    if int(n_boot) < 2:
        raise ValueError("n_boot must be at least 2")


def _percentile_result(est, boots, n, conf, n_boot, method, n_failed=None):
    a = (1 - conf) / 2
    lo, hi = np.percentile(boots, [100 * a, 100 * (1 - a)])
    return InferenceResult(
        estimate=float(est),
        method=method,
        n=n,
        ci_lower=float(lo),
        ci_upper=float(hi),
        conf_level=conf,
        se=float(np.std(boots, ddof=1)) if len(boots) > 1 else None,
        n_boot=n_boot,
        n_failed=n_failed,
    )


def stratified_bootstrap_ci(
    strata: Dict[Any, Sequence],
    statistic: Callable[[Dict[Any, Sequence]], float],
    conf: float = 0.95,
    n_boot: int = 1000,
    random_state: Optional[int] = 0,
    min_stratum_n: int = 5,
    label: str = "group",
) -> InferenceResult:
    """Percentile bootstrap that resamples WITHIN each stratum (e.g. group).

    Every stratum keeps its observed size in every replicate, so the interval
    describes the same set of groups as the point estimate. A pooled bootstrap
    instead lets small groups vanish from some replicates (a group of one is
    absent from about 37% of them) and then compares whichever groups remain.

    Parameters
    ----------
    strata : mapping stratum -> its observations (list or array). Every
        stratum must be non-empty.
    statistic : callable mapping a dict of the same keys (resampled
        observations) to a float.
    min_stratum_n : strata smaller than this trigger a warning, because their
        contribution to the interval is unreliable.

    Replicates whose statistic is not finite (for example a ratio whose
    denominator is 0 in that replicate) are dropped and counted in
    ``n_failed``; a warning is raised if more than 5% are dropped.
    """
    _check_resampling(conf, n_boot)
    keys = list(strata)
    if not keys:
        raise ValueError("strata is empty")
    data = {}
    for k in keys:
        v = strata[k]
        v = v if isinstance(v, np.ndarray) else list(v)
        if len(v) == 0:
            raise ValueError(f"{label} {k!r} has no observations")
        data[k] = v
    small = sorted(str(k) for k in keys if len(data[k]) < min_stratum_n)
    if small:
        warnings.warn(
            f"{label}s with fewer than {min_stratum_n} observations: {small}; their "
            "contribution to the confidence interval is unreliable.",
            UserWarning,
            stacklevel=3,
        )
    rng = np.random.default_rng(random_state)
    est = float(statistic(data))
    boots, failed = [], 0
    for _ in range(int(n_boot)):
        sample = {}
        for k in keys:
            v = data[k]
            idx = rng.integers(0, len(v), size=len(v))
            sample[k] = v[idx] if isinstance(v, np.ndarray) else [v[i] for i in idx]
        b = float(statistic(sample))
        if np.isfinite(b):
            boots.append(b)
        else:
            failed += 1
    if len(boots) < 2:
        raise ValueError("too few finite bootstrap replicates to form an interval")
    if failed > 0.05 * n_boot:
        warnings.warn(
            f"{failed} of {n_boot} bootstrap replicates gave an undefined statistic "
            "and were dropped; the interval may be unreliable.",
            UserWarning,
            stacklevel=3,
        )
    n = sum(len(data[k]) for k in keys)
    return _percentile_result(
        est, boots, n, conf, int(n_boot), f"bootstrap (stratified by {label})", failed
    )


def block_bootstrap_ci(
    series: Sequence[float],
    statistic: Callable[[np.ndarray], float],
    block_length: Optional[int] = None,
    conf: float = 0.95,
    n_boot: int = 1000,
    random_state: Optional[int] = 0,
) -> InferenceResult:
    """Moving-block bootstrap CI for a statistic of a time series.

    Resamples overlapping blocks of consecutive points, which keeps short-range
    serial dependence that an ordinary bootstrap destroys (Kunsch, Ann Stat
    1989). The default block length is round(n ** (1/3)), at least 1.
    """
    _check_resampling(conf, n_boot)
    x = np.asarray(series, dtype=float)
    n = len(x)
    if n == 0:
        raise ValueError("series is empty")
    L = int(block_length) if block_length else max(1, int(round(n ** (1 / 3))))
    if not 1 <= L <= n:
        raise ValueError("block_length must be between 1 and the series length")
    rng = np.random.default_rng(random_state)
    n_blocks = int(np.ceil(n / L))
    starts_max = n - L + 1
    boots = []
    for _ in range(int(n_boot)):
        starts = rng.integers(0, starts_max, size=n_blocks)
        sample = np.concatenate([x[s : s + L] for s in starts])[:n]
        boots.append(float(statistic(sample)))
    return _percentile_result(
        statistic(x), boots, n, conf, int(n_boot), f"moving-block bootstrap (block {L})"
    )


def bootstrap_metric(
    metric_fn: Callable[[Sequence], "float | dict"],
    data: Sequence,
    value_key: Optional[str] = None,
    conf: float = 0.95,
    n_boot: int = 2000,
    clusters: Optional[Sequence] = None,
    random_state: Optional[int] = None,
) -> InferenceResult:
    """Bootstrap CI for *any* EquiMed-DSS metric over its observation sample.

    Wraps :func:`bootstrap_ci` so a metric that maps a subset of observations to
    a value (or to a result dict, with ``value_key`` selecting the scalar) gains
    a confidence interval without changing the metric itself.

    Examples
    --------
    >>> from equimed_dss.domain4 import ClinicalHallucinationRate
    >>> chr_fn = lambda s: ClinicalHallucinationRate().calculate_chr(s)
    >>> r = bootstrap_metric(chr_fn, support_scores, value_key="chr",
    ...                      random_state=0)
    """

    def stat(subset):
        out = metric_fn(subset)
        if value_key is not None:
            out = out[value_key]
        return float(out)

    return bootstrap_ci(
        data,
        stat,
        conf=conf,
        n_boot=n_boot,
        clusters=clusters,
        random_state=random_state,
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
    if alternative not in _ALTERNATIVES:
        raise ValueError(f"alternative must be one of {_ALTERNATIVES}")
    if int(n_perm) < 1:
        raise ValueError("n_perm must be at least 1")
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
