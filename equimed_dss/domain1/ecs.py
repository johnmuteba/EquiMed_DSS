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
            Dictionary containing mean, std, and median ECS (cosine distance).
        """
        n_samples = original_embeddings.shape[0]
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

            # ECS is defined as distance (1 - similarity) in the script context for "consistency gap"
            # But "Consistency Score" usually implies higher is better.
            # The script calculates `1 - cos_sim` and calls it ECS, implying it's a "Consistency Error" or "Inconsistency".
            # However, to align with the script's output "Most sensitive corpus... ECS = ...",
            # where higher ECS meant more sensitive (less consistent), we will return the distance.

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

        return {
            "mean_ecs": mean_ecs,
            "std_ecs": float(np.std(cosine_distances)),
            "median_ecs": float(np.median(cosine_distances)),
            "interpretation": {
                "range": "[0, 2] (typically [0, 1])",
                "ideal": "Lower is better (close to 0)",
                "verdict": verdict,
            },
        }
