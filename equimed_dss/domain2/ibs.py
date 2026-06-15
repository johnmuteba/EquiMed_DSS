from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform


class IntersectionalBiasScore:
    """
    Domain 2: Fairness, Equity, and Ethics Assessment
    Metric 7: Intersectional Bias Score (IBS)

    Assesses complex bias patterns across demographic dimensions using subgroup similarity
    and multi-way interaction analysis.
    """

    def __init__(self):
        pass

    def calculate_subgroup_similarity(
        self, subgroup_vectors: Dict[str, np.ndarray]
    ) -> Dict[str, Any]:
        """
        Calculate similarity matrix between subgroups based on metric vectors.

        Args:
            subgroup_vectors: Dictionary mapping subgroup names to metric vectors (numpy arrays).

        Returns:
            Dictionary containing similarity matrix and outlier analysis.
        """
        subgroups = list(subgroup_vectors.keys())
        vectors = np.array([subgroup_vectors[g] for g in subgroups])

        # Pairwise Euclidean distances
        distances = squareform(pdist(vectors, metric="euclidean"))

        # Convert to similarity (inverse distance)
        similarity_matrix = 1 / (1 + distances)

        # Find outliers (highest average distance to others)
        avg_distances = distances.mean(axis=1)
        outlier_idx = np.argmax(avg_distances)

        return {
            "similarity_matrix": similarity_matrix.tolist(),
            "subgroups": subgroups,
            "outlier_subgroup": subgroups[outlier_idx],
            "outlier_distance": float(avg_distances[outlier_idx]),
            "mean_similarity": float(similarity_matrix[similarity_matrix < 1].mean()),
            "interpretation": {
                "range": "Similarity [0, 1], Distance [0, inf)",
                "ideal": "High similarity, Low outlier distance",
                "verdict": f"Outlier detected: {subgroups[outlier_idx]}",
            },
        }

    def interaction_analysis(
        self, df: pd.DataFrame, formula: str = "score ~ C(race) * C(gender) * C(ses)"
    ) -> Dict[str, float]:
        """
        Perform simplified interaction analysis (ANOVA-like) to detect intersectional effects.
        Note: Full ANOVA requires statsmodels, here we implement a simplified variance analysis
        if statsmodels is not available or for lightweight usage.

        Args:
            df: DataFrame containing columns for demographics and 'score'.
            formula: Formula string (informational here, logic assumes race/gender/ses columns).

        Returns:
            Dictionary of interaction effect sizes (eta-squared proxies).
        """
        # Work on a copy so the caller's DataFrame is never mutated (a temporary
        # 'race_gender' column is added below for the interaction term).
        df = df.copy()

        # Simplified approach: Compare variance of subgroups vs total variance
        total_var = df["score"].var()

        # Main effects
        effects = {}
        for col in ["race", "gender", "ses"]:
            if col in df.columns:
                group_means = df.groupby(col)["score"].mean()
                # Weighted variance of means
                grand_mean = df["score"].mean()
                ss_between = sum(
                    df[df[col] == g].shape[0] * (mean - grand_mean) ** 2
                    for g, mean in group_means.items()
                )
                ss_total = sum((df["score"] - grand_mean) ** 2)
                effects[col] = ss_between / ss_total if ss_total > 0 else 0

        # Interaction (Race x Gender)
        if "race" in df.columns and "gender" in df.columns:
            df["race_gender"] = df["race"].astype(str) + "_" + df["gender"].astype(str)
            group_means = df.groupby("race_gender")["score"].mean()
            grand_mean = df["score"].mean()
            ss_between = sum(
                df[df["race_gender"] == g].shape[0] * (mean - grand_mean) ** 2
                for g, mean in group_means.items()
            )
            ss_total = sum((df["score"] - grand_mean) ** 2)
            effects["race_gender_interaction"] = (
                (ss_between / ss_total)
                - effects.get("race", 0)
                - effects.get("gender", 0)
            )

        return effects
