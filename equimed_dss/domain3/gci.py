from typing import Any, Dict, List


class GovernanceComplianceIndex:
    """
    Domain 3: Governance and Transparency Assessment
    Metric 10: Governance Compliance Index (GCI)

    Quantifies adherence to regulatory requirements.
    """

    def __init__(self):
        pass

    def calculate_gci(self, policy_compliance: Dict[str, bool]) -> Dict[str, Any]:
        """
        Calculate GCI based on a dictionary of policy compliance statuses.

        Args:
            policy_compliance: Dictionary mapping policy names to boolean status (True=Compliant).

        Returns:
            Dictionary containing GCI score and details.
        """
        from equimed_dss.inference import MetricResult, proportion_ci

        if not policy_compliance:
            return MetricResult({"gci": 0.0}, name="GCI", value_key="gci")

        n_mandated = len(policy_compliance)
        n_enforced = sum(1 for status in policy_compliance.values() if status)

        gci = n_enforced / n_mandated

        gaps = [policy for policy, status in policy_compliance.items() if not status]

        # GCI is the proportion of enforced policies, so a Wilson score interval
        # is its natural 95% CI.
        inf = proportion_ci(n_enforced, n_mandated)

        return MetricResult({
            "gci": float(gci),
            "policies_enforced": n_enforced,
            "policies_mandated": n_mandated,
            "compliance_gaps": gaps,
            "ci_lower": inf.ci_lower,
            "ci_upper": inf.ci_upper,
            "ci_method": inf.method,
            "interpretation": {
                "range": "[0, 1]",
                "ideal": "1.0 (Full Compliance)",
                "verdict": (
                    "Fully Compliant"
                    if gci == 1.0
                    else f"Partial Compliance ({gci:.0%})"
                ),
            },
        }, name="GCI", value_key="gci")
