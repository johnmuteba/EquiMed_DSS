from typing import Any, Dict, List, Optional, Union

import numpy as np


class TemporalFairnessDrift:
    """
    Domain 3: Governance and Transparency Assessment
    Metric 8: Temporal Fairness Drift (TFD)

    Tracks fairness degradation over time using Process Drift Index (PDI) and Control Charts.
    """

    def __init__(self):
        pass

    def calculate_drift(
        self,
        time_series_metrics: List[float],
        sigma_method: str = "sd",
        baseline_n: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Calculate drift metrics and control limits for a time series of fairness metrics.

        Args:
            time_series_metrics: List of fairness metric values over time.
            sigma_method: how the control-chart sigma is estimated. ``"sd"``
                (default, unchanged since 1.0) uses the sample SD; a drift inside
                the estimation window inflates it and widens the limits.
                ``"moving_range"`` uses the average moving range divided by
                d2 = 1.128, the standard estimate for an individuals (I-MR) chart
                (Montgomery, Introduction to Statistical Quality Control).
            baseline_n: number of initial points that form a fixed baseline. When
                given, the centre and limits are estimated from those points only
                and applied prospectively to the later points, which is how a
                control chart should monitor drift. Without it (default), the
                limits come from the whole series being monitored, so a shift can
                move the centre or widen the limits and go undetected.

        Returns:
            Dictionary with the centre (``mean_pdi``: the mean of the points used
            for the limits), ``series_mean``, sigma, limits, the indices of points
            outside the limits, ``phase`` and a 95% moving-block bootstrap CI for
            the centre (blocks keep short-range serial dependence, which an
            ordinary bootstrap would ignore).
        """
        from equimed_dss.inference import MetricResult, block_bootstrap_ci

        if sigma_method not in ("sd", "moving_range"):
            raise ValueError("sigma_method must be 'sd' or 'moving_range'.")
        metrics = np.array(time_series_metrics, dtype=float)
        if metrics.size == 0:
            raise ValueError("time_series_metrics must be non-empty.")
        if not np.all(np.isfinite(metrics)):
            raise ValueError("time_series_metrics must be finite.")
        if baseline_n is not None:
            if int(baseline_n) != baseline_n or not 2 <= baseline_n < len(metrics):
                raise ValueError(
                    "baseline_n must be an integer, at least 2 and smaller than the "
                    "series length."
                )
            base = metrics[: int(baseline_n)]
            first_monitored = int(baseline_n)
            phase = f"prospective (limits from the first {int(baseline_n)} points)"
        else:
            base = metrics
            first_monitored = 0
            phase = "retrospective (limits from the whole series)"

        mean_val = np.mean(base)
        if len(base) < 2:
            std_val = 0.0
        elif sigma_method == "moving_range":
            # Average moving range / d2 (d2 = 1.128 for ranges of 2 points).
            std_val = float(np.mean(np.abs(np.diff(base)))) / 1.128
        else:
            # Sample SD (ddof=1) for the control-chart sigma estimate.
            std_val = np.std(base, ddof=1)

        # Control limits (3-sigma)
        ucl = mean_val + 3 * std_val
        lcl = mean_val - 3 * std_val

        # Points outside the limits (monitored points only)
        out_of_control = [
            i
            for i in range(first_monitored, len(metrics))
            if metrics[i] > ucl or metrics[i] < lcl
        ]

        out = {
            "mean_pdi": float(mean_val),
            "series_mean": float(np.mean(metrics)),
            "std_pdi": float(std_val),
            "ucl": float(ucl),
            "lcl": float(lcl),
            "out_of_control_indices": out_of_control,
            "drift_detected": len(out_of_control) > 0,
            "sigma_method": sigma_method,
            "phase": phase,
            "interpretation": {
                "range": "N/A (Process Control)",
                "ideal": "No points outside the limits",
                "verdict": (
                    f"{len(out_of_control)} point(s) outside the 3-sigma limits"
                    if out_of_control
                    else "No points outside the 3-sigma limits"
                ),
            },
        }

        # Moving-block bootstrap CI for the centre (the points behind the limits).
        if len(base) >= 2:
            ci = block_bootstrap_ci(base, np.mean, n_boot=1000, random_state=0)
            out["ci_lower"] = ci.ci_lower
            out["ci_upper"] = ci.ci_upper
            out["ci_method"] = ci.method

        return MetricResult(out, name="TFD", value_key="mean_pdi")
