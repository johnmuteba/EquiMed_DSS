"""Small input checks shared by several metrics (private module)."""
import math
import warnings
from typing import Any, Dict, Iterable, Sequence

import numpy as np


def is_missing(value: Any) -> bool:
    """True for None and for floating-point NaN (Python or NumPy)."""
    if value is None:
        return True
    if isinstance(value, (float, np.floating)):
        return math.isnan(value)
    return False


def check_no_missing(values: Iterable[Any], name: str) -> None:
    """Raise ValueError if ``values`` contains None or NaN.

    Missing outputs must be dropped or imputed by the caller before a metric is
    computed: NaN compares unequal to itself, so it would otherwise count as a
    change, and a missing score would otherwise count as a valid one.
    """
    n_missing = sum(1 for v in values if is_missing(v))
    if n_missing:
        raise ValueError(
            f"{name} contains {n_missing} missing value(s) (None or NaN); drop or "
            "impute them before computing the metric."
        )


def check_records(
    records: Sequence[str],
    known: Iterable[str],
    point_shares: Dict[str, float],
    name: str,
    tolerance: float = 0.005,
) -> None:
    """Check per-record region labels used to bootstrap an aggregate metric.

    Every label must be one of the known regions (otherwise it would be dropped
    silently from the resamples). A warning is raised when the shares implied by
    the records differ from the shares behind the point estimate, because the
    interval would then not describe the reported value.
    """
    known = set(known)
    unknown = sorted({r for r in records if r not in known})
    if unknown:
        raise ValueError(
            f"{name} has labels that are not among the regions of the point "
            f"estimate: {unknown}."
        )
    n = len(records)
    if n == 0:
        return
    counts: Dict[str, int] = {}
    for r in records:
        counts[r] = counts.get(r, 0) + 1
    worst = max(abs(counts.get(r, 0) / n - s) for r, s in point_shares.items())
    if worst > tolerance:
        warnings.warn(
            f"The region shares implied by {name} differ from those of the point "
            f"estimate by up to {worst:.3f}; the confidence interval may not "
            "describe the reported value.",
            UserWarning,
            stacklevel=3,
        )
