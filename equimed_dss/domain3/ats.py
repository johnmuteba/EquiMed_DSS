from typing import Any, Dict

import numpy as np


class AuditTraceabilityScore:
    """
    Domain 3: Governance and Transparency Assessment
    Metric 9: Audit Traceability Score (ATS)

    Measures decision traceability to specific sources using Wilson score interval.
    """

    def __init__(self):
        pass

    def calculate_ats(self, n_traceable: int, n_total: int) -> Dict[str, float]:
        """
        Calculate ATS and its confidence interval.

        Args:
            n_traceable: Number of decisions that are traceable.
            n_total: Total number of decisions audited.

        Returns:
            Dictionary containing ATS score and confidence interval.
        """
        from equimed_dss.inference import MetricResult

        if n_total == 0:
            return MetricResult(
                {"ats_score": 0.0, "ci_lower": 0.0, "ci_upper": 0.0,
                 "ci_method": "Wilson score"},
                name="ATS", value_key="ats_score",
            )

        p = n_traceable / n_total

        # Wilson score interval (z=1.96 for 95% CI)
        z = 1.96
        p_tilde = (n_traceable + z**2 / 2) / (n_total + z**2)
        se = np.sqrt(p_tilde * (1 - p_tilde) / (n_total + z**2))

        ci_lower = max(0.0, p_tilde - z * se)
        ci_upper = min(1.0, p_tilde + z * se)

        return MetricResult({
            "ats_score": float(p),
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
            "ci_method": "Wilson score",
            "meets_95_standard": bool(p >= 0.95),
            "interpretation": {
                "range": "[0, 1]",
                "ideal": "Higher is better (target >= 0.95)",
                "verdict": "Compliant" if p >= 0.95 else "Non-Compliant",
            },
        }, name="ATS", value_key="ats_score")
