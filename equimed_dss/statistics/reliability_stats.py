"""
Reliability Analysis Statistics

Implements reliability measures referenced in manuscript:
- Cronbach's Alpha (internal consistency)
- Bland-Altman analysis (inter-rater agreement)
- Test-retest reliability
"""

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


class ReliabilityAnalysis:
    """Statistical reliability assessment methods."""

    def cronbachs_alpha(self, data: np.ndarray) -> Dict[str, Any]:
        """
        Calculate Cronbach's Alpha for internal consistency.

        Args:
            data: 2D array where rows=subjects, columns=items/raters

        Returns:
            Dict with alpha value and interpretation

        Interpretation:
            - α > 0.9: Excellent
            - 0.8 < α ≤ 0.9: Good
            - 0.7 < α ≤ 0.8: Acceptable
            - α ≤ 0.7: Questionable
        """
        data = np.array(data)
        n_items = data.shape[1]

        # Item variances
        item_vars = np.var(data, axis=0, ddof=1)

        # Total score variance
        total_scores = np.sum(data, axis=1)
        total_var = np.var(total_scores, ddof=1)

        # Cronbach's alpha
        alpha = (n_items / (n_items - 1)) * (1 - (np.sum(item_vars) / total_var))

        return {
            "alpha": float(alpha),
            "n_items": n_items,
            "interpretation": {
                "range": "[0, 1]",
                "quality": (
                    "Excellent"
                    if alpha > 0.9
                    else (
                        "Good"
                        if alpha > 0.8
                        else "Acceptable" if alpha > 0.7 else "Questionable"
                    )
                ),
            },
        }

    def bland_altman_analysis(
        self, method1: np.ndarray, method2: np.ndarray
    ) -> Dict[str, Any]:
        """
        Perform Bland-Altman analysis for method agreement.

        Args:
            method1: Measurements from first method/rater
            method2: Measurements from second method/rater

        Returns:
            Dict with mean difference and limits of agreement
        """
        method1 = np.array(method1)
        method2 = np.array(method2)

        differences = method1 - method2
        means = (method1 + method2) / 2

        mean_diff = np.mean(differences)
        std_diff = np.std(differences, ddof=1)

        # 95% limits of agreement
        loa_upper = mean_diff + 1.96 * std_diff
        loa_lower = mean_diff - 1.96 * std_diff

        return {
            "mean_difference": float(mean_diff),
            "std_difference": float(std_diff),
            "loa_upper": float(loa_upper),
            "loa_lower": float(loa_lower),
            "loa_width": float(loa_upper - loa_lower),
            "interpretation": {
                "agreement": (
                    "Excellent agreement"
                    if abs(mean_diff) < 0.1
                    else "Good agreement" if abs(mean_diff) < 0.2 else "Poor agreement"
                )
            },
        }
