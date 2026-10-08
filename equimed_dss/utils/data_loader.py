"""Synthetic inputs for the examples (never study data).

Each generator takes ``random_state``; pass an integer for reproducible output.
"""

from typing import Dict, List, Optional

import numpy as np


def generate_synthetic_judge_data(
    n_items: int = 100, n_judges: int = 3, random_state: Optional[int] = None
) -> np.ndarray:
    """Generate synthetic judge scores."""
    rng = np.random.default_rng(random_state)
    base_scores = rng.normal(7.5, 1.0, n_items)
    judge_matrix = np.zeros((n_items, n_judges))
    for j in range(n_judges):
        judge_matrix[:, j] = base_scores + rng.normal(0, 0.5, n_items)
    return judge_matrix


def generate_synthetic_embeddings(
    n_samples: int = 100, dim: int = 768, random_state: Optional[int] = None
) -> np.ndarray:
    """Generate synthetic embeddings."""
    return np.random.default_rng(random_state).standard_normal((n_samples, dim))


def generate_synthetic_fairness_data(
    groups: Optional[List[str]] = None, random_state: Optional[int] = None
) -> Dict[str, float]:
    """Generate synthetic fairness scores."""
    rng = np.random.default_rng(random_state)
    groups = ["White", "Black", "Asian"] if groups is None else groups
    return {g: float(rng.uniform(0.7, 0.9)) for g in groups}
