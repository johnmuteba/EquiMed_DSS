import warnings
from typing import Any, Dict, List

import networkx as nx
import numpy as np


class AdvancedNetworkMetrics:
    """
    Appendix A.3: network and governance helpers.

    Includes:
    - calculate_modularity: Newman modularity of a weighted network.
    - calculate_explained_fraction: share of decisions that came with an
      explanation (a count; different from TransparencyScore, which averages
      three ratings).
    - calculate_stability_pass_rate: share of stability scores at or above a
      threshold (different from ObservedPerturbationAgreement, which averages
      prediction agreement).

    The older names calculate_transparency_score and calculate_rcs are kept as
    deprecated aliases: they reused the names of different metrics, and the
    "certified" flag of calculate_rcs certified nothing.
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

    def calculate_explained_fraction(self, n_explained: int, n_total: int) -> float:
        """Share of decisions that came with an explanation, n_explained / n_total."""
        if n_total <= 0:
            raise ValueError("n_total must be positive.")
        if not 0 <= n_explained <= n_total:
            raise ValueError("n_explained must lie between 0 and n_total.")
        return float(n_explained / n_total)

    def calculate_stability_pass_rate(
        self,
        stability_scores: List[float],
        threshold: float = 0.8,
        target: float = 0.95,
    ) -> Dict[str, Any]:
        """Share of stability scores at or above ``threshold``.

        Returns the pass rate and whether it reaches ``target``. This describes
        the scores given; it is not a robustness certification.
        """
        x = np.asarray(stability_scores, dtype=float)
        if x.size == 0:
            raise ValueError(
                "stability_scores is empty, so the pass rate is undefined."
            )
        if not np.all(np.isfinite(x)):
            raise ValueError("stability_scores must be finite.")
        rate = float(np.mean(x >= threshold))
        return {
            "pass_rate": rate,
            "threshold": threshold,
            "target": target,
            "meets_target": bool(rate >= target),
        }

    def calculate_transparency_score(self, n_explained: int, n_total: int) -> float:
        """Deprecated: use :meth:`calculate_explained_fraction` (same value)."""
        warnings.warn(
            "AdvancedNetworkMetrics.calculate_transparency_score is deprecated (it is "
            "not TransparencyScore); use calculate_explained_fraction.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.calculate_explained_fraction(n_explained, n_total)

    def calculate_rcs(
        self, stability_scores: List[float], threshold: float = 0.8
    ) -> Dict[str, Any]:
        """Deprecated: use :meth:`calculate_stability_pass_rate`.

        Returns the old keys; ``certified`` only means pass rate >= 0.95.
        """
        warnings.warn(
            "AdvancedNetworkMetrics.calculate_rcs is deprecated (it is not the RCS "
            "of ObservedPerturbationAgreement, and 'certified' certifies nothing); "
            "use calculate_stability_pass_rate.",
            DeprecationWarning,
            stacklevel=2,
        )
        r = self.calculate_stability_pass_rate(stability_scores, threshold)
        return {
            "rcs_score": r["pass_rate"],
            "certified": r["meets_target"],
            "threshold": threshold,
        }
