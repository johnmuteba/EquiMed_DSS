from typing import Dict, List

import numpy as np


def generate_synthetic_judge_data(n_items: int = 100, n_judges: int = 3) -> np.ndarray:
    """Generate synthetic judge scores."""
    base_scores = np.random.normal(7.5, 1.0, n_items)
    judge_matrix = np.zeros((n_items, n_judges))
    for j in range(n_judges):
        judge_matrix[:, j] = base_scores + np.random.normal(0, 0.5, n_items)
    return judge_matrix


def generate_synthetic_embeddings(n_samples: int = 100, dim: int = 768) -> np.ndarray:
    """Generate synthetic embeddings."""
    return np.random.randn(n_samples, dim)


def generate_synthetic_fairness_data(
    groups: List[str] = ["White", "Black", "Asian"]
) -> Dict[str, float]:
    """Generate synthetic fairness scores."""
    return {g: np.random.uniform(0.7, 0.9) for g in groups}
