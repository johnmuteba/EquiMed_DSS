from typing import Dict, List, Tuple, Union

import numpy as np


class InterRaterReliability:
    """
    Domain 1: Reliability and Robustness Assessment
    Metric 1: Inter-Rater Reliability (ICC)

    Calculates Intraclass Correlation Coefficient (ICC 2,1) and performs Bland-Altman analysis
    to assess consistency among clinical evaluators or model outputs.
    """

    def __init__(self):
        pass

    def calculate_icc_2_1(
        self, judge_matrix: np.ndarray
    ) -> Dict[str, Union[float, Dict]]:
        """
        Calculate ICC(2,1) using ANOVA approach.

        Args:
            judge_matrix: numpy array of shape (n_items, n_judges) containing scores.

        Returns:
            Dictionary with ICC score and interpretation.
        """
        n_items, n_judges = judge_matrix.shape

        # Mean squares calculation
        grand_mean = np.mean(judge_matrix)
        item_means = np.mean(judge_matrix, axis=1)
        judge_means = np.mean(judge_matrix, axis=0)

        # Sum of squares
        ss_total = np.sum((judge_matrix - grand_mean) ** 2)
        ss_items = n_judges * np.sum((item_means - grand_mean) ** 2)
        ss_judges = n_items * np.sum((judge_means - grand_mean) ** 2)
        ss_error = ss_total - ss_items - ss_judges

        # Mean squares
        ms_items = ss_items / (n_items - 1)
        ms_judges = ss_judges / (n_judges - 1)
        ms_error = ss_error / ((n_items - 1) * (n_judges - 1))

        # ICC(2,1) calculation
        numerator = ms_items - ms_error
        denominator = (
            ms_items
            + (n_judges - 1) * ms_error
            + (n_judges * (ms_judges - ms_error) / n_items)
        )

        icc_2_1 = numerator / denominator

        # Interpretation
        if icc_2_1 >= 0.75:
            verdict = "Excellent"
        elif icc_2_1 >= 0.60:
            verdict = "Good"
        elif icc_2_1 >= 0.40:
            verdict = "Fair"
        else:
            verdict = "Poor"

        return {
            "score": float(icc_2_1),
            "interpretation": {
                "range": "[0, 1]",
                "ideal": "Higher is better (close to 1)",
                "verdict": verdict,
                "thresholds": ">0.75 (Exc), >0.6 (Good), >0.4 (Fair)",
            },
        }

    def bland_altman_analysis(
        self, judge_matrix: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """
        Perform Bland-Altman analysis for pairs of judges.

        Args:
            judge_matrix: numpy array of shape (n_items, n_judges).

        Returns:
            Dictionary containing mean difference and limits of agreement for each pair.
        """
        n_judges = judge_matrix.shape[1]
        results = {}

        for i in range(n_judges):
            for j in range(i + 1, n_judges):
                pair_name = f"Judge{i+1}-Judge{j+1}"
                means = (judge_matrix[:, i] + judge_matrix[:, j]) / 2
                differences = judge_matrix[:, i] - judge_matrix[:, j]
                mean_diff = np.mean(differences)
                std_diff = np.std(differences)

                results[pair_name] = {
                    "mean_difference": float(mean_diff),
                    "std_difference": float(std_diff),
                    "upper_loa": float(mean_diff + 1.96 * std_diff),
                    "lower_loa": float(mean_diff - 1.96 * std_diff),
                }
        return results

    def interpret_score(self, icc_score: float) -> str:
        """Interpret the ICC score."""
        if icc_score > 0.75:
            return "Excellent"
        elif icc_score > 0.60:
            return "Good"
        elif icc_score > 0.40:
            return "Fair"
        else:
            return "Poor"
