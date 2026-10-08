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
            policy_compliance: Dictionary mapping each listed check (policy) to
                True when it is met.

        Returns:
            Dictionary containing GCI (the share of listed checks met) and details.
            GCI describes the checks given; it does not establish regulatory
            compliance. Its Wilson interval has a sampling meaning only if the
            checks are a sample from a larger defined set; for a complete,
            fixed list, report GCI itself.

        Raises:
            ValueError: if no checks are given (GCI undefined; up to 1.9.5 it was
                reported as 0).
        """
        from equimed_dss.inference import MetricResult, proportion_ci

        if not policy_compliance:
            raise ValueError("policy_compliance is empty, so GCI is undefined.")

        n_mandated = len(policy_compliance)
        n_enforced = sum(1 for status in policy_compliance.values() if status)

        gci = n_enforced / n_mandated

        gaps = [policy for policy, status in policy_compliance.items() if not status]

        # GCI is the proportion of enforced policies, so a Wilson score interval
        # is its natural 95% CI. It treats the audited policies as a sample of
        # the policies that could have been audited; for a complete census of a
        # fixed policy list, report GCI itself.
        inf = proportion_ci(n_enforced, n_mandated)

        return MetricResult(
            {
                "gci": float(gci),
                "policies_enforced": n_enforced,
                "policies_mandated": n_mandated,
                "compliance_gaps": gaps,
                "ci_lower": inf.ci_lower,
                "ci_upper": inf.ci_upper,
                "ci_method": inf.method,
                "interpretation": {
                    "range": "[0, 1]",
                    "ideal": "1.0 (all listed checks met)",
                    "verdict": (
                        "All listed checks met"
                        if gci == 1.0
                        else f"{gci:.0%} of listed checks met"
                    ),
                },
            },
            name="GCI",
            value_key="gci",
        )
