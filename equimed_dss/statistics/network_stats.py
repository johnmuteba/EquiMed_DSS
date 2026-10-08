"""
Network Analysis Statistics

Implements centrality measures and network analysis as described in
Manuscript Section 2.4:
- Degree centrality: deg(v)/(n-1)
- Betweenness centrality: Σ σst(v)/σst
"""

from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np


class NetworkStatistics:
    """Network analysis for metric correlation structures."""

    def analyze_network(
        self,
        adjacency_matrix: np.ndarray,
        node_labels: List[str] = None,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Comprehensive network analysis of metric correlations.

        Args:
            adjacency_matrix: Correlation or adjacency matrix. Absolute values
                are used and the diagonal is ignored (a correlation matrix's 1s
                are not edges; up to 1.9.5 they became self-loops and inflated
                degree centrality above 1).
            node_labels: Optional labels for nodes
            threshold: optional minimum absolute weight for an edge. The
                centralities are unweighted, so a dense correlation matrix
                without a threshold gives a complete graph in which every node
                has the same degree; set a threshold (e.g. 0.3) to analyse the
                structure of the stronger correlations.

        Returns:
            Dict with centrality measures and network properties
        """
        A = np.abs(np.asarray(adjacency_matrix, dtype=float))
        np.fill_diagonal(A, 0.0)
        if threshold is not None:
            A[A < threshold] = 0.0
        G = nx.from_numpy_array(A)

        if node_labels:
            mapping = {i: label for i, label in enumerate(node_labels)}
            G = nx.relabel_nodes(G, mapping)

        # Calculate centrality measures
        degree_cent = nx.degree_centrality(G)
        betweenness_cent = nx.betweenness_centrality(G)
        closeness_cent = nx.closeness_centrality(G)

        # Network properties
        clustering = nx.clustering(G)
        avg_clustering = nx.average_clustering(G)

        return {
            "degree_centrality": degree_cent,
            "betweenness_centrality": betweenness_cent,
            "closeness_centrality": closeness_cent,
            "clustering_coefficients": clustering,
            "average_clustering": float(avg_clustering),
            "n_nodes": G.number_of_nodes(),
            "n_edges": G.number_of_edges(),
            "density": float(nx.density(G)),
        }
