from typing import Any, Dict, List

import networkx as nx
import numpy as np


class AdvancedNetworkMetrics:
    """
    Appendix A.3: Advanced Network and Governance Metrics

    Includes:
    17. Network Modularity (NM)
    18. Transparency Score (TS) - (Note: TS is also in Domain 3 as part of RAMS, implemented here as standalone)
    19. Robustness Certification Score (RCS)
    """

    def __init__(self):
        pass

    def calculate_modularity(self, adjacency_matrix: np.ndarray) -> float:
        """
        Calculate Network Modularity using NetworkX (greedy modularity).

        Absolute weights are used and the diagonal is ignored (a correlation
        matrix's 1s are not edges); communities are found and scored with the
        same weights. A network without edges has modularity 0. Up to 1.9.5
        the diagonal was kept, the community search ignored the weights, and
        any error was silently turned into 0.0.
        """
        from networkx.algorithms.community import (
            greedy_modularity_communities,
            modularity,
        )

        A = np.abs(np.asarray(adjacency_matrix, dtype=float))
        np.fill_diagonal(A, 0.0)
        G = nx.from_numpy_array(A)
        if G.number_of_edges() == 0:
            return 0.0
        communities = greedy_modularity_communities(G, weight="weight")
        return float(modularity(G, communities, weight="weight"))

    def calculate_transparency_score(self, n_explained: int, n_total: int) -> float:
        """
        Calculate Transparency Score (clinician ability to understand AI reasoning).
        """
        if n_total == 0:
            return 0.0
        return float(n_explained / n_total)

    def calculate_rcs(
        self, stability_scores: List[float], threshold: float = 0.8
    ) -> Dict[str, Any]:
        """
        Calculate Robustness Certification Score based on stability under variations.
        """
        if not stability_scores:
            return {}

        pass_rate = sum(1 for s in stability_scores if s >= threshold) / len(
            stability_scores
        )

        return {
            "rcs_score": float(pass_rate),
            "certified": bool(pass_rate >= 0.95),
            "threshold": threshold,
        }
