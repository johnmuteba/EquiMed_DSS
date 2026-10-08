from typing import Any, Dict


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
            Dictionary containing ATS score and its 95% Wilson score interval.

        Raises:
            ValueError: if ``n_traceable`` is not between 0 and ``n_total``.
        """
        from equimed_dss.inference import MetricResult, wilson_ci

        if n_total == 0:
            return MetricResult(
                {
                    "ats_score": 0.0,
                    "ci_lower": 0.0,
                    "ci_upper": 0.0,
                    "ci_method": "Wilson score",
                },
                name="ATS",
                value_key="ats_score",
            )

        # Wilson score interval (95%), shared with the rest of the library.
        # Versions up to 1.9.5 labelled this interval "Wilson score" but
        # computed the Agresti-Coull interval, which is slightly wider.
        inf = wilson_ci(int(n_traceable), int(n_total))
        p = inf.estimate
        ci_lower = max(0.0, inf.ci_lower)
        ci_upper = min(1.0, inf.ci_upper)

        return MetricResult(
            {
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
            },
            name="ATS",
            value_key="ats_score",
        )
