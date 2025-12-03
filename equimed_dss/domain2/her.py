from typing import Dict, List, Union

import numpy as np


class HierarchicalEquityRatio:
    """
    Domain 2: Fairness, Equity, and Ethics Assessment
    Metric 4: Hierarchical Equity Ratio (HER)

    Calculates HER (ratio of metric for group vs reference) and Bias-Gini Index
    to assess multi-level inequities.
    """

    def __init__(self):
        pass

    def calculate_her(
        self, group_scores: Dict[str, float], reference_group: str = "White"
    ) -> Dict[str, float]:
        """
        Calculate Hierarchical Equity Ratio for each group relative to a reference group.

        Args:
            group_scores: Dictionary mapping group names to their performance scores.
            reference_group: Name of the reference group (default: 'White').

        Returns:
            Dictionary mapping group names to their HER.
        """
        if reference_group not in group_scores:
            raise ValueError(f"Reference group '{reference_group}' not found in scores")

        reference_score = group_scores[reference_group]
        her_scores = {}

        for group, score in group_scores.items():
            if reference_score == 0:
                val = 0.0
            else:
                val = score / reference_score

            # Interpretation
            if 0.8 <= val <= 1.25:  # 4/5ths rule approximation
                verdict = "Equitable"
            else:
                verdict = "Disparity Detected"

            her_scores[group] = {
                "score": float(val),
                "interpretation": {
                    "range": "[0, inf)",
                    "ideal": "Close to 1 (0.8 - 1.25)",
                    "verdict": verdict,
                },
            }

        return her_scores

    def calculate_bias_gini(self, scores: List[float]) -> float:
        """
        Calculate Bias-Gini Index to measure dispersion of fairness metrics.

        Args:
            scores: List of performance scores across groups.

        Returns:
            float: Bias-Gini Index.
        """
        if not scores:
            return 0.0

        n = len(scores)
        mean_score = np.mean(scores)

        if mean_score == 0:
            return 0.0

        gini_sum = 0
        for i in range(n):
            for j in range(n):
                gini_sum += abs(scores[i] - scores[j])

        gini = gini_sum / (2 * n * n * mean_score)
        return float(gini)
