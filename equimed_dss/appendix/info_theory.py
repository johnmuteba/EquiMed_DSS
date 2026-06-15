from typing import Any, Dict, List

import numpy as np
from scipy.spatial.distance import jensenshannon
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
        two probability distributions. Consistent with
        ``advanced_metrics.JensenShannonDivergence`` (both return the divergence,
        not the distance).
        """
        # Normalize if needed
        p = np.array(p) / np.sum(p)
        q = np.array(q) / np.sum(q)

        # jensenshannon returns the distance (sqrt of divergence) in the given
        # base; square it for the divergence, base 2 so the range is [0, 1].
        return float(jensenshannon(p, q, base=2) ** 2)

    def calculate_wasserstein(
        self, u_values: List[float], v_values: List[float]
    ) -> float:
        """
        Calculate Wasserstein Distance (Earth Mover's Distance) between two distributions.
        """
        return float(wasserstein_distance(u_values, v_values))
