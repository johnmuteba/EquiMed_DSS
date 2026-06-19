from typing import Dict, List, Union

import numpy as np


class EthicalRiskIndex:
    """
    Domain 2: Fairness, Equity, and Ethics Assessment
    Metric 6: Ethical Risk Index (ERI)

    Quantifies ethical violations by severity and calculates Safety Violation Rate (SVR).
    """

    def __init__(self):
        pass

    def calculate_eri(
        self, violations: List[Dict[str, float]], n_total_outputs: int
    ) -> Dict[str, float]:
        """
        Calculate ERI based on a list of violations with severity scores.

        Args:
            violations: List of dicts, each containing 'severity' (float).
            n_total_outputs: Total number of outputs evaluated.

        Returns:
            MetricResult with ERI (mean severity per output) and SVR (violation rate
            per 1000). ERI carries a 95% percentile-bootstrap CI computed over the
            full per-output severity vector (severity for violating outputs, 0 for
            the rest); SVR is a proportion, so its rate carries a Wilson CI. Printing
            shows ERI with its CI.
        """
        from equimed_dss.inference import MetricResult, bootstrap_ci, proportion_ci

        if n_total_outputs == 0:
            return MetricResult(
                {"eri": 0.0, "svr": 0.0, "n_violations": 0, "total_severity": 0.0},
                name="ERI",
                value_key="eri",
            )

        severities = [float(v.get("severity", 0)) for v in violations]
        total_severity = float(sum(severities))
        n_violations = len(violations)

        eri = total_severity / n_total_outputs
        svr = (n_violations / n_total_outputs) * 1000  # Rate per 1000

        # Reconstruct the per-output severity vector: each violation keeps its
        # severity, every non-violating output contributes 0. ERI is the mean of
        # this vector, so a bootstrap over it gives an honest CI.
        n_clean = max(0, n_total_outputs - n_violations)
        per_output = severities + [0.0] * n_clean

        out = {
            "eri": float(eri),
            "svr": float(svr),
            "n_violations": n_violations,
            "n_total_outputs": int(n_total_outputs),
            "total_severity": total_severity,
            "interpretation": {
                "range": "[0, inf)",
                "ideal": "Lower is better (0 is perfect)",
                "verdict": "High Risk" if eri > 1.0 else "Low Risk",
            },
        }

        if len(per_output) >= 2:
            ci = bootstrap_ci(per_output, lambda s: float(np.mean(s)),
                              n_boot=1000, random_state=0)
            out["ci_lower"] = ci.ci_lower
            out["ci_upper"] = ci.ci_upper
            out["ci_method"] = ci.method

        # Wilson CI for the violation proportion (SVR is this proportion x 1000).
        svr_inf = proportion_ci(min(n_violations, n_total_outputs), n_total_outputs)
        out["svr_ci_lower"] = float(svr_inf.ci_lower * 1000)
        out["svr_ci_upper"] = float(svr_inf.ci_upper * 1000)
        out["svr_ci_method"] = svr_inf.method

        return MetricResult(out, name="ERI", value_key="eri")
