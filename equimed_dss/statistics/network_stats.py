"""
Network Analysis Statistics

Implements centrality measures and network analysis as described in
Manuscript Section 2.4:
- Degree centrality: deg(v)/(n-1)
- Betweenness centrality: Σ σst(v)/σst
"""

from typing import Any, Dict, List, Tuple

import networkx as nx
import numpy as np


class NetworkStatistics:
    """Network analysis for metric correlation structures."""

    def analyze_network(
        self,
        adjacency_matrix: np.ndarray,
        node_labels: List[str] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive network analysis of metric correlations.

        Args:
            adjacency_matrix: Correlation or adjacency matrix
            node_labels: Optional labels for nodes

        Returns:
            Dict with centrality measures and network properties
        """
        G = nx.from_numpy_array(np.abs(adjacency_matrix))

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
            "density": float(nx.density(G))
        }
