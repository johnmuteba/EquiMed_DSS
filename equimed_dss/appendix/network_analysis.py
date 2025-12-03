import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Any, Tuple
from scipy.spatial.distance import pdist, squareform

class AdvancedNetworkAnalysis:
    """
    Advanced Network Analysis for EquiMed_DSS.
    
    Implements complex network-based fairness metrics:
    - RQ16: Metric Correlation Network
    - RQ17: Subgroup Similarity Graph (Enhanced)
    - RQ18: Concept Co-occurrence Network
    - RQ19: Temporal Fairness Dynamics
    """
    
    def __init__(self):
        pass
        
    def metric_correlation_network(self, metric_df: pd.DataFrame, threshold: float = 0.5) -> Dict[str, Any]:
        """
        RQ16: Analyze correlations between different fairness metrics.
        
        Args:
            metric_df: DataFrame where columns are metrics and rows are observations (e.g., models, timepoints).
            threshold: Correlation threshold to draw edges.
            
        Returns:
            NetworkX graph stats and correlation matrix.
        """
        corr_matrix = metric_df.corr()
        
        # Build graph
        G = nx.Graph()
        for i, metric1 in enumerate(corr_matrix.columns):
            for j, metric2 in enumerate(corr_matrix.columns):
                if i < j:
                    weight = corr_matrix.iloc[i, j]
                    if abs(weight) >= threshold:
                        G.add_edge(metric1, metric2, weight=weight)
                        
        # Network metrics
        centrality = nx.degree_centrality(G)
        modularity = 0.0
        try:
            from networkx.algorithms.community import greedy_modularity_communities, modularity as calc_modularity
            communities = greedy_modularity_communities(G)
            modularity = calc_modularity(G, communities)
        except:
            pass
            
        return {
            'correlation_matrix': corr_matrix,
            'graph': G,
            'num_nodes': G.number_of_nodes(),
            'num_edges': G.number_of_edges(),
            'centrality': centrality,
            'modularity': modularity,
            'interpretation': {
                'verdict': f"Found {G.number_of_edges()} strong correlations (>{threshold})",
                'ideal': "Depends on context (trade-offs vs synergies)"
            }
        }

    def subgroup_similarity_graph(self, subgroup_vectors: Dict[str, np.ndarray], threshold: float = 0.5) -> Dict[str, Any]:
        """
        RQ17: Enhanced Subgroup Similarity Graph.
        
        Args:
            subgroup_vectors: Dict mapping subgroup names to metric vectors.
            threshold: Similarity threshold for edges.
        """
        subgroups = list(subgroup_vectors.keys())
        vectors = np.array([subgroup_vectors[g] for g in subgroups])
        
        # Cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity
        sim_matrix = cosine_similarity(vectors)
        
        G = nx.Graph()
        for i, g1 in enumerate(subgroups):
            G.add_node(g1)
            for j, g2 in enumerate(subgroups):
                if i < j:
                    sim = sim_matrix[i, j]
                    if sim >= threshold:
                        G.add_edge(g1, g2, weight=sim)
                        
        # Detect isolates (outliers)
        isolates = list(nx.isolates(G))
        
        return {
            'similarity_matrix': sim_matrix,
            'graph': G,
            'isolates': isolates,
            'interpretation': {
                'verdict': f"{len(isolates)} outlier subgroups detected",
                'ideal': "No isolates (all groups similar)"
            }
        }

    def concept_cooccurrence_network(self, texts: List[str], concepts: List[str]) -> Dict[str, Any]:
        """
        RQ18: Concept Co-occurrence Network.
        
        Args:
            texts: List of text documents.
            concepts: List of concepts to track.
        """
        # Simple co-occurrence
        cooc_matrix = pd.DataFrame(0, index=concepts, columns=concepts)
        
        for text in texts:
            text_lower = text.lower()
            present = [c for c in concepts if c.lower() in text_lower]
            for i, c1 in enumerate(present):
                for j, c2 in enumerate(present):
                    if i < j:
                        cooc_matrix.loc[c1, c2] += 1
                        cooc_matrix.loc[c2, c1] += 1
                        
        G = nx.from_pandas_adjacency(cooc_matrix)
        
        return {
            'cooccurrence_matrix': cooc_matrix,
            'graph': G,
            'density': nx.density(G),
            'interpretation': {
                'verdict': f"Network density: {nx.density(G):.2f}",
                'ideal': "Higher density implies stronger conceptual linkage"
            }
        }

    def temporal_fairness_dynamics(self, time_series_graphs: List[nx.Graph]) -> Dict[str, Any]:
        """
        RQ19: Temporal Fairness Dynamics (Network Evolution).
        
        Args:
            time_series_graphs: List of NetworkX graphs at sequential timepoints.
        """
        # Track edge stability (Jaccard index of edges between steps)
        stability_scores = []
        for i in range(len(time_series_graphs) - 1):
            edges1 = set(time_series_graphs[i].edges())
            edges2 = set(time_series_graphs[i+1].edges())
            
            if not edges1 and not edges2:
                jaccard = 1.0
            else:
                jaccard = len(edges1.intersection(edges2)) / len(edges1.union(edges2))
            stability_scores.append(jaccard)
            
        return {
            'stability_scores': stability_scores,
            'mean_stability': float(np.mean(stability_scores)) if stability_scores else 0.0,
            'interpretation': {
                'verdict': f"Mean Stability: {np.mean(stability_scores):.2f}" if stability_scores else "N/A",
                'ideal': "High stability (consistent fairness relations)"
            }
        }
