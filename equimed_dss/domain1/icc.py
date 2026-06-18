from typing import Dict, List, Tuple, Union

import numpy as np


def _icc_2_1_score(judge_matrix: np.ndarray) -> float:
    """ICC(2,1) point estimate via the two-way ANOVA decomposition.

    Returns 0.0 for degenerate resamples (no between-item variance), which keeps
    the bootstrap distribution finite.
    """
    m = np.asarray(judge_matrix, dtype=float)
    n_items, n_judges = m.shape
    if n_items < 2 or n_judges < 2:
        return 0.0
    grand = m.mean()
    item_means = m.mean(axis=1)
    judge_means = m.mean(axis=0)
    ss_total = np.sum((m - grand) ** 2)
    ss_items = n_judges * np.sum((item_means - grand) ** 2)
    ss_judges = n_items * np.sum((judge_means - grand) ** 2)
    ss_error = ss_total - ss_items - ss_judges
    ms_items = ss_items / (n_items - 1)
    ms_judges = ss_judges / (n_judges - 1)
    ms_error = ss_error / ((n_items - 1) * (n_judges - 1))
    denom = ms_items + (n_judges - 1) * ms_error + n_judges * (ms_judges - ms_error) / n_items
    if not np.isfinite(denom) or denom == 0:
        return 0.0
    val = (ms_items - ms_error) / denom
    return float(val) if np.isfinite(val) else 0.0


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

        # 95% CI by bootstrapping over items (rows). Small item counts give a
        # wide, honestly unstable interval.
        from equimed_dss.inference import MetricResult, bootstrap_ci

        ci = bootstrap_ci(list(judge_matrix), lambda rows: _icc_2_1_score(np.asarray(rows)),
                          n_boot=1000, random_state=0)
        return MetricResult({
            "score": float(icc_2_1),
            "ci_lower": ci.ci_lower,
            "ci_upper": ci.ci_upper,
            "ci_method": "bootstrap (over items)",
            "interpretation": {
                "range": "[0, 1]",
                "ideal": "Higher is better (close to 1)",
                "verdict": verdict,
                "thresholds": ">0.75 (Exc), >0.6 (Good), >0.4 (Fair)",
            },
        }, name="ICC(2,1)", value_key="score")

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
                # Sample SD (ddof=1) is the convention for Bland-Altman limits.
                std_diff = np.std(differences, ddof=1) if len(differences) > 1 else 0.0

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
