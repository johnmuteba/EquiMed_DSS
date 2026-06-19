"""Semantic Parity Gap (SPG).

SPG measures latent bias in an LLM's representation space as the geometric
distance between the embedding centroids of clinical scenarios that differ only
by a protected demographic attribute.

Euclidean centroid distance:
    SPG = || mean_i E(x_p,i) - mean_j E(x_m,j) ||_2
Cosine (orientation) variant, with mean vectors v_p, v_m:
    SPG_cos = 1 - (v_p . v_m) / (||v_p|| ||v_m||)

A larger SPG means the model's internal representation of an identical clinical
case is more strongly altered by patient identity.
"""
from typing import Any, Dict

import numpy as np


class SemanticParityGap:
    """Semantic Parity Gap (SPG) between two demographic embedding clusters."""

    def __init__(self):
        pass

    def calculate_spg(
        self,
        privileged_embeddings,
        marginalized_embeddings,
    ) -> Dict[str, Any]:
        """Compute the Semantic Parity Gap between two embedding clusters.

        Args:
            privileged_embeddings: array-like of shape (n, d), embeddings of the
                clinical prompt for the privileged group.
            marginalized_embeddings: array-like of shape (m, d), embeddings of the
                identical prompt for the marginalized group.

        Returns:
            Dict with spg_euclidean, spg_cosine, embedding_dim, n_privileged,
            n_marginalized, and interpretation.
        """
        p = np.asarray(privileged_embeddings, dtype=float)
        m = np.asarray(marginalized_embeddings, dtype=float)
        if p.ndim != 2 or m.ndim != 2:
            raise ValueError("Embeddings must be 2D arrays of shape (n, d).")
        if p.shape[0] == 0 or m.shape[0] == 0:
            raise ValueError("Both embedding clusters must be non-empty.")
        if p.shape[1] != m.shape[1]:
            raise ValueError(
                f"Embedding dimensions differ: {p.shape[1]} vs {m.shape[1]}."
            )

        cp = p.mean(axis=0)
        cm = m.mean(axis=0)
        spg_euclidean = float(np.linalg.norm(cp - cm))
        denom = float(np.linalg.norm(cp) * np.linalg.norm(cm))
        spg_cosine = float(1.0 - (cp @ cm) / denom) if denom > 0 else 0.0

        from equimed_dss.inference import MetricResult, bootstrap_ci

        out = {
            "spg_euclidean": spg_euclidean,
            "spg_cosine": spg_cosine,
            "embedding_dim": int(p.shape[1]),
            "n_privileged": int(p.shape[0]),
            "n_marginalized": int(m.shape[0]),
            "interpretation": (
                f"SPG (Euclidean centroid distance) = {spg_euclidean:.4f}; "
                f"cosine variant = {spg_cosine:.4f}. Larger values mean the "
                "model's representation of an identical case shifts more with "
                "patient identity (latent demographic bias)."
            ),
        }

        # Bootstrap the centroid-distance SPG by resampling embedding rows within
        # each group (each group resampled independently to its own size), which
        # propagates the sampling variability of both centroids into the CI.
        if p.shape[0] >= 2 and m.shape[0] >= 2:
            rng = np.random.default_rng(0)
            n_boot = 1000
            boots = []
            np_, nm_ = p.shape[0], m.shape[0]
            for _ in range(n_boot):
                ip = rng.integers(0, np_, size=np_)
                im = rng.integers(0, nm_, size=nm_)
                boots.append(float(np.linalg.norm(p[ip].mean(axis=0) - m[im].mean(axis=0))))
            lo, hi = np.percentile(boots, [2.5, 97.5])
            out["ci_lower"] = float(lo)
            out["ci_upper"] = float(hi)
            out["ci_method"] = "bootstrap"

        return MetricResult(out, name="SPG", value_key="spg_euclidean")
