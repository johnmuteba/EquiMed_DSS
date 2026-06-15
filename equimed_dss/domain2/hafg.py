from typing import Dict, Union

import numpy as np


class HarmAdjustedFairnessGap:
    """
    Domain 2: Fairness, Equity, and Ethics Assessment
    Metric 5: Harm-Adjusted Fairness Gap (HAFG)

    Quantifies fairness weighted by potential clinical harm (cost of errors).
    """

    def __init__(self, cost_fn: float = 10.0, cost_fp: float = 3.0):
        """
        Initialize with costs for False Negatives and False Positives.

        Args:
            cost_fn: Cost of a false negative (default: 10).
            cost_fp: Cost of a false positive (default: 3).
        """
        self.cost_fn = cost_fn
        self.cost_fp = cost_fp

    def calculate_hafg(
        self, group1_errors: Dict[str, int], group2_errors: Dict[str, int]
    ) -> Dict[str, float]:
        """
        Calculate HAFG between two groups (e.g., Marginalized vs Privileged).

        Args:
            group1_errors: Dict with 'fn' (count) and 'fp' (count) for group 1.
            group2_errors: Dict with 'fn' (count) and 'fp' (count) for group 2.

        Returns:
            Dictionary containing harm for each group and the gap.
        """
        harm1 = (
            group1_errors.get("fn", 0) * self.cost_fn
            + group1_errors.get("fp", 0) * self.cost_fp
        )
        harm2 = (
            group2_errors.get("fn", 0) * self.cost_fn
            + group2_errors.get("fp", 0) * self.cost_fp
        )

        gap = abs(harm1 - harm2)
        # HAFG is normalized by the larger harm so it lies in [0, 1] and is
        # comparable across datasets: HAFG = |H1 - H2| / max(H1, H2).
        denom = max(harm1, harm2)
        hafg = float(gap / denom) if denom > 0 else 0.0

        if hafg < 0.1:
            verdict = "Minimal harm disparity"
        elif hafg < 0.2:
            verdict = "Moderate harm disparity"
        else:
            verdict = "Significant harm disparity"

        return {
            "harm_group1": float(harm1),
            "harm_group2": float(harm2),
            "hafg": hafg,
            "absolute_harm_gap": float(gap),
            "ratio": float(harm1 / harm2) if harm2 > 0 else float("inf"),
            "interpretation": {
                "range": "[0, 1]",
                "ideal": "Lower is better (close to 0)",
                "verdict": verdict,
            },
        }
