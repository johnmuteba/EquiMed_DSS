from typing import Any, Dict, List

import numpy as np
from scipy import stats


class AdvancedReliabilityMetrics:
    """
    Appendix A.1: Advanced Reliability Metrics

    Includes:
    11. Bootstrap Confidence Intervals (BCI)
    12. Statistical Power Analysis (SPA)
    13. Bias Concentration Index (BCI - distinct from Bootstrap CI)
    """

    def __init__(self):
        pass

    def calculate_bootstrap_ci(
        self, data: List[float], n_bootstrap: int = 1000, alpha: float = 0.05
    ) -> Dict[str, float]:
        """
        Calculate Bootstrap Confidence Intervals.
        """
        if not data:
            return {}

        data_np = np.array(data)
        means = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(data_np, size=len(data_np), replace=True)
            means.append(np.mean(sample))

        return {
            "mean": float(np.mean(data_np)),
            "ci_lower": float(np.percentile(means, 100 * (alpha / 2))),
            "ci_upper": float(np.percentile(means, 100 * (1 - alpha / 2))),
        }

    def calculate_power_analysis(
        self, effect_size: float, alpha: float = 0.05, power: float = 0.8
    ) -> Dict[str, Any]:
        """
        Calculate sample size requirements for detecting bias (Two-sample Z-test approximation).
        """
        # Z-scores
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = stats.norm.ppf(power)

        n_per_group = ((z_alpha + z_beta) / effect_size) ** 2

        return {
            "required_n_per_group": int(np.ceil(n_per_group)),
            "total_n": int(np.ceil(n_per_group * 2)),
            "parameters": {"alpha": alpha, "power": power, "effect_size": effect_size},
        }

    def calculate_bias_concentration(
        self, population_share: np.ndarray, health_variable: np.ndarray
    ) -> float:
        """
        Calculate Bias Concentration Index (Concentration Index).
        Assumes data is sorted by the ranking variable (e.g., income).
        """
        # Simplified calculation based on area between curve and diagonal
        # CI = 2 * cov(h, r) / mean(h) where r is fractional rank

        n = len(health_variable)
        if n == 0:
            return 0.0

        fractional_rank = (np.arange(1, n + 1) - 0.5) / n
        mean_h = np.mean(health_variable)

        if mean_h == 0:
            return 0.0

        cov = np.cov(health_variable, fractional_rank)[0, 1]
        ci = 2 * cov / mean_h
        return float(ci)
