from typing import Any, Dict, List

from scipy.stats import entropy, wasserstein_distance


class AdvancedInfoTheoryMetrics:
    """
    Appendix A.2: Advanced Information-Theoretic Metrics

    Includes:
    14. Mutual Information Content (MIC)
    15. Jensen-Shannon Divergence (JSD)
    16. Wasserstein Distance (WD)
    """

    def __init__(self):
        pass

    def calculate_mic(self, x: List[Any], y: List[Any]) -> float:
        """
        Calculate Mutual Information between two discrete variables.
        """
        from sklearn.metrics import mutual_info_score

        return float(mutual_info_score(x, y))

    def calculate_jsd(self, p: List[float], q: List[float]) -> float:
        """
        Calculate the Jensen-Shannon Divergence (base 2, range [0, 1]) between
        two probability distributions (or counts) over the same categories, in
        the same order; histogram raw samples on common bins first. Consistent with
        ``advanced_metrics.JensenShannonDivergence`` (both return the divergence,
        not the distance).
        """
        from equimed_dss.appendix.advanced_metrics import JensenShannonDivergence

        # Same validation and value as the canonical implementation.
        return float(JensenShannonDivergence().calculate_jsd(p, q)["jsd"])

    def calculate_wasserstein(
        self, u_values: List[float], v_values: List[float]
    ) -> float:
        """
        Calculate Wasserstein Distance (Earth Mover's Distance) between two distributions.
        """
        return float(wasserstein_distance(u_values, v_values))
