from typing import Any, Dict, List, Optional

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
        self,
        data: List[float],
        n_bootstrap: int = 1000,
        alpha: float = 0.05,
        random_state: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Percentile bootstrap confidence interval for the mean.

        Args:
            data: observations.
            n_bootstrap: number of bootstrap resamples.
            alpha: 1 - confidence level (0.05 gives a 95% interval).
            random_state: seed; pass an integer for a reproducible interval
                (up to 1.9.5 the global NumPy generator was used, so results
                could not be reproduced).
        """
        if not data:
            return {}

        rng = np.random.default_rng(random_state)
        data_np = np.array(data, dtype=float)
        means = []
        for _ in range(n_bootstrap):
            sample = data_np[rng.integers(0, len(data_np), size=len(data_np))]
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
        Sample size per group to detect a standardised mean difference.

        Two-sample z-test approximation with equal groups and a two-sided test:
        n per group = 2 (z_{1-alpha/2} + z_{power})^2 / d^2, where d is Cohen's d.
        For d = 0.5, alpha = 0.05 and power = 0.8 this gives 63 per group; the
        t-test solution in ``StatisticalPowerAnalysis`` gives 64. Up to 1.9.5
        the factor 2 was missing, so the sample size was half what is needed.
        """
        if effect_size == 0:
            raise ValueError("effect_size must be non-zero.")
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = stats.norm.ppf(power)

        n_per_group = 2 * ((z_alpha + z_beta) / effect_size) ** 2

        return {
            "required_n_per_group": int(np.ceil(n_per_group)),
            "total_n": 2 * int(np.ceil(n_per_group)),
            "parameters": {"alpha": alpha, "power": power, "effect_size": effect_size},
        }

    def calculate_bias_concentration(
        self, population_share: np.ndarray, health_variable: np.ndarray
    ) -> float:
        """
        Concentration index of a health (or bias) variable over a ranking.

        Rows must be sorted by the ranking variable (e.g. income, poorest
        first). For grouped data, ``population_share`` gives each row's share of
        the population; the fractional rank of row i is the cumulative share of
        the rows before it plus half its own, and

            C = (2 / mu) * sum_i w_i (h_i - mu) (R_i - 1/2),  mu = sum_i w_i h_i,

        which for equal shares equals 2 cov(h, R) / mu with the population
        (1/n) covariance (Kakwani, Wagstaff and van Doorslaer, J Econometrics
        1997; O'Donnell et al., Analyzing Health Equity Using Household Survey
        Data, World Bank 2008). Pass ``None`` or an empty array for equal
        shares. Up to 1.9.5 ``population_share`` was ignored and the sample
        (1/(n-1)) covariance was used, which overstated C by n/(n-1).
        """
        h = np.asarray(health_variable, dtype=float)
        n = len(h)
        if n == 0:
            return 0.0
        if population_share is None or len(population_share) == 0:
            w = np.full(n, 1.0 / n)
        else:
            w = np.asarray(population_share, dtype=float)
            if w.shape != h.shape:
                raise ValueError(
                    "population_share must have one entry per row of health_variable."
                )
            if (w < 0).any() or w.sum() <= 0:
                raise ValueError(
                    "population_share must be non-negative with a positive sum."
                )
            w = w / w.sum()

        mean_h = float(np.sum(w * h))
        if mean_h == 0:
            return 0.0

        fractional_rank = np.cumsum(w) - w / 2
        ci = 2.0 * np.sum(w * (h - mean_h) * (fractional_rank - 0.5)) / mean_h
        return float(ci)
