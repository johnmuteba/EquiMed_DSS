import numpy as np
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
from typing import Dict, List, Any

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
        Calculate Jensen-Shannon Divergence between two probability distributions.
        """
        # Normalize if needed
        p = np.array(p) / np.sum(p)
        q = np.array(q) / np.sum(q)
        
        # jensenshannon returns distance (sqrt of divergence), square it for divergence
        return float(jensenshannon(p, q) ** 2)

    def calculate_wasserstein(self, u_values: List[float], v_values: List[float]) -> float:
        """
        Calculate Wasserstein Distance (Earth Mover's Distance) between two distributions.
        """
        return float(wasserstein_distance(u_values, v_values))
