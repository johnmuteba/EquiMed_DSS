from typing import Dict, List, Union

import numpy as np


class EmbeddingConsistencyScore:
    """
    Domain 1: Reliability and Robustness Assessment
    Metric 2: Embedding Consistency Score (ECS)

    Measures semantic consistency of embeddings under perturbations using cosine similarity.
    """

    def __init__(self):
        pass

    def calculate_ecs(
        self, original_embeddings: np.ndarray, perturbed_embeddings: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate ECS between original and perturbed embeddings.

        Args:
            original_embeddings: numpy array of shape (n_samples, embedding_dim).
            perturbed_embeddings: numpy array of shape (n_samples, embedding_dim).

        Returns:
            Dictionary containing mean, std, and median ECS (cosine distance,
            1 - cosine similarity; 0 = unchanged, higher = less consistent). A
            pair in which either vector is all zeros has distance 1.

        Raises:
            ValueError: if the two arrays are not 2D with the same shape.
        """
        original_embeddings = np.asarray(original_embeddings, dtype=float)
        perturbed_embeddings = np.asarray(perturbed_embeddings, dtype=float)
        if original_embeddings.ndim != 2 or perturbed_embeddings.ndim != 2:
            raise ValueError("Embeddings must be 2D arrays (n_samples, embedding_dim).")
        if original_embeddings.shape != perturbed_embeddings.shape:
            raise ValueError(
                "original_embeddings and perturbed_embeddings must have the same "
                f"shape; got {original_embeddings.shape} and "
                f"{perturbed_embeddings.shape}."
            )
        n_samples = original_embeddings.shape[0]
        if n_samples == 0:
            raise ValueError("Embeddings must be non-empty.")
        cosine_distances = []

        for i in range(n_samples):
            # Cosine similarity
            norm_orig = np.linalg.norm(original_embeddings[i])
            norm_pert = np.linalg.norm(perturbed_embeddings[i])

            if norm_orig == 0 or norm_pert == 0:
                cos_sim = 0.0
            else:
                cos_sim = np.dot(original_embeddings[i], perturbed_embeddings[i]) / (
                    norm_orig * norm_pert
                )

            # ECS is reported as a distance (1 - similarity): higher means the
            # embedding moved more under the perturbation.
            cosine_distances.append(1 - cos_sim)

        mean_ecs = float(np.mean(cosine_distances))

        # Interpretation
        # Cosine distance is [0, 2], but usually [0, 1] for embeddings
        if mean_ecs < 0.1:
            verdict = "Excellent Consistency"
        elif mean_ecs < 0.2:
            verdict = "Good Consistency"
        else:
            verdict = "Poor Consistency (High Sensitivity)"

        # 95% CI for the mean ECS by bootstrapping over the per-pair distances.
        from equimed_dss.inference import MetricResult, bootstrap_ci

        ci = bootstrap_ci(cosine_distances, lambda s: float(np.mean(s)),
                          n_boot=1000, random_state=0)
        return MetricResult({
            "mean_ecs": mean_ecs,
            "std_ecs": float(np.std(cosine_distances)),
            "median_ecs": float(np.median(cosine_distances)),
            "ci_lower": ci.ci_lower,
            "ci_upper": ci.ci_upper,
            "ci_method": "bootstrap",
            "interpretation": {
                "range": "[0, 2] (typically [0, 1])",
                "ideal": "Lower is better (close to 0)",
                "verdict": verdict,
            },
        }, name="ECS", value_key="mean_ecs")
