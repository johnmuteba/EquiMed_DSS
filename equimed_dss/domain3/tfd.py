from typing import Any, Dict, List, Union

import numpy as np


class TemporalFairnessDrift:
    """
    Domain 3: Governance and Transparency Assessment
    Metric 8: Temporal Fairness Drift (TFD)

    Tracks fairness degradation over time using Process Drift Index (PDI) and Control Charts.
    """

    def __init__(self):
        pass

    def calculate_drift(self, time_series_metrics: List[float]) -> Dict[str, Any]:
        """
        Calculate drift metrics and control limits for a time series of fairness metrics.

        Args:
            time_series_metrics: List of fairness metric values over time.

        Returns:
            Dictionary containing PDI stats and out-of-control points.
        """
        from equimed_dss.inference import MetricResult, bootstrap_ci

        if not time_series_metrics:
            return MetricResult({"mean_pdi": 0.0}, name="TFD", value_key="mean_pdi")

        metrics = np.array(time_series_metrics)
        mean_val = np.mean(metrics)
        # Sample SD (ddof=1) for the control-chart sigma estimate.
        std_val = np.std(metrics, ddof=1) if len(metrics) > 1 else 0.0

        # Control limits (3-sigma)
        ucl = mean_val + 3 * std_val
        lcl = mean_val - 3 * std_val

        # Detect out-of-control points
        out_of_control = []
        for i, val in enumerate(metrics):
            if val > ucl or val < lcl:
                out_of_control.append(i)

        out = {
            "mean_pdi": float(mean_val),
            "std_pdi": float(std_val),
            "ucl": float(ucl),
            "lcl": float(lcl),
            "out_of_control_indices": out_of_control,
            "drift_detected": len(out_of_control) > 0,
            "interpretation": {
                "range": "N/A (Process Control)",
                "ideal": "No drift detected",
                "verdict": (
                    "Unstable Process" if len(out_of_control) > 0 else "Stable Process"
                ),
            },
        }

        # Percentile bootstrap CI for the process mean (mean PDI) over the
        # observed time series of fairness-metric values.
        if len(metrics) >= 2:
            ci = bootstrap_ci(metrics.tolist(), lambda s: float(np.mean(s)),
                              n_boot=1000, random_state=0)
            out["ci_lower"] = ci.ci_lower
            out["ci_upper"] = ci.ci_upper
            out["ci_method"] = ci.method

        return MetricResult(out, name="TFD", value_key="mean_pdi")
