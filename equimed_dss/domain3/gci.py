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
        if not policy_compliance:
            return {"gci": 0.0}

        n_mandated = len(policy_compliance)
        n_enforced = sum(1 for status in policy_compliance.values() if status)

        gci = n_enforced / n_mandated

        gaps = [policy for policy, status in policy_compliance.items() if not status]

        return {
            "gci": float(gci),
            "policies_enforced": n_enforced,
            "policies_mandated": n_mandated,
            "compliance_gaps": gaps,
            "interpretation": {
                "range": "[0, 1]",
                "ideal": "1.0 (Full Compliance)",
                "verdict": (
                    "Fully Compliant"
                    if gci == 1.0
                    else f"Partial Compliance ({gci:.0%})"
                ),
            },
        }
