import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import pandas as pd
import seaborn as sns
import networkx as nx

def plot_network_graph(G: nx.Graph, title: str = "Network Graph", node_color: str = 'skyblue', save_path: Optional[str] = None):
    """Plot a NetworkX graph."""
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_color=node_color, 
            node_size=1500, font_size=10, font_weight='bold', 
            edge_color='gray', alpha=0.7)
    labels = nx.get_edge_attributes(G, 'weight')
    # Format weights
    labels = {k: f"{v:.2f}" for k, v in labels.items()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.title(title)
    plt.axis('off')
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_correlation_matrix(corr_matrix: pd.DataFrame, title: str = "Correlation Matrix", save_path: Optional[str] = None):
    """Plot correlation matrix heatmap."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_metric_distribution(data: List[float], title: str = "Metric Distribution", save_path: Optional[str] = None):
    """Plot distribution of a metric (e.g., bootstrap results)."""
    plt.figure(figsize=(10, 6))
    sns.histplot(data, kde=True, color='blue', alpha=0.3)
    plt.axvline(np.mean(data), color='red', linestyle='--', label=f'Mean: {np.mean(data):.2f}')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_bland_altman(means: np.ndarray, diffs: np.ndarray, limits: Tuple[float, float], title: str = "Bland-Altman Plot", save_path: Optional[str] = None):
    """Plot Bland-Altman analysis."""
    plt.figure(figsize=(8, 6))
    plt.scatter(means, diffs, alpha=0.5)
    plt.axhline(np.mean(diffs), color='black', linestyle='-')
    plt.axhline(limits[0], color='red', linestyle='--')
    plt.axhline(limits[1], color='red', linestyle='--')
    plt.title(title)
    plt.xlabel("Mean")
    plt.ylabel("Difference")
    plt.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_her_heatmap(her_scores: Dict[str, Dict[str, float]], title: str = "HER Heatmap", save_path: Optional[str] = None):
    """Plot HER scores as a heatmap."""
    groups = list(next(iter(her_scores.values())).keys())
    corpora = list(her_scores.keys())
    
    matrix = np.zeros((len(corpora), len(groups)))
    for i, corpus in enumerate(corpora):
        for j, group in enumerate(groups):
            matrix[i, j] = her_scores[corpus].get(group, 0)
            
    plt.figure(figsize=(10, 6))
    plt.imshow(matrix, cmap='RdYlGn', vmin=0.5, vmax=1.5, aspect='auto')
    plt.colorbar(label='HER')
    plt.yticks(range(len(corpora)), corpora)
    plt.xticks(range(len(groups)), groups, rotation=45)
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_control_chart(data: List[float], ucl: float, lcl: float, title: str = "Control Chart", save_path: Optional[str] = None):
    """Plot process control chart."""
    plt.figure(figsize=(10, 5))
    plt.plot(data, marker='o')
    plt.axhline(np.mean(data), color='green', linestyle='-')
    plt.axhline(ucl, color='red', linestyle='--')
    plt.axhline(lcl, color='red', linestyle='--')
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Metric Value")
    plt.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
