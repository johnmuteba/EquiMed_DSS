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

A centroid distance is positive even when both groups come from the same
distribution (sampling noise alone separates two sample means), so its
bootstrap interval never contains 0 and cannot show "no gap". The permutation
p-value tests the null that the group labels are exchangeable: it compares the
observed distance with the distances obtained after shuffling the labels.
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
        paired: bool = False,
    ) -> Dict[str, Any]:
        """Compute the Semantic Parity Gap between two embedding clusters.

        Args:
            privileged_embeddings: array-like of shape (n, d), embeddings of the
                clinical prompt for the privileged group.
            marginalized_embeddings: array-like of shape (m, d), embeddings of the
                identical prompt for the marginalized group.
            paired: set True when row i of both arrays is the SAME clinical case
                with only the protected attribute changed (n == m). The interval
                then resamples cases (keeping each pair together) and the
                permutation test swaps the two members within randomly chosen
                pairs, so the pairing is respected. With the default (False) the
                two groups are treated as independent samples.

        Returns:
            Dict with spg_euclidean, spg_cosine, embedding_dim, n_privileged,
            n_marginalized, paired, interpretation and, when each group has at
            least two rows, a bootstrap CI and ``p_value_permutation`` (1000
            permutations, add-one estimator, seed 0).

        A shift in the model's internal representation does not by itself show
        clinically harmful bias; relate it to differences in the outputs (for
        example DecisionFlipRate or CounterfactualParityScore on the same pairs).
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
        if not (np.all(np.isfinite(p)) and np.all(np.isfinite(m))):
            raise ValueError("Embeddings must be finite.")
        if paired and p.shape[0] != m.shape[0]:
            raise ValueError(
                "paired=True needs the same number of rows in both arrays."
            )

        cp = p.mean(axis=0)
        cm = m.mean(axis=0)
        spg_euclidean = float(np.linalg.norm(cp - cm))
        denom = float(np.linalg.norm(cp) * np.linalg.norm(cm))
        spg_cosine = float(1.0 - (cp @ cm) / denom) if denom > 0 else 0.0

        from equimed_dss.inference import MetricResult

        out = {
            "spg_euclidean": spg_euclidean,
            "spg_cosine": spg_cosine,
            "embedding_dim": int(p.shape[1]),
            "n_privileged": int(p.shape[0]),
            "n_marginalized": int(m.shape[0]),
            "paired": bool(paired),
            "interpretation": (
                f"SPG (Euclidean centroid distance) = {spg_euclidean:.4f}; "
                f"cosine variant = {spg_cosine:.4f}. Larger values mean the "
                "model's representation of an identical case shifts more with "
                "patient identity; this alone does not show harm to patients."
            ),
        }

        if p.shape[0] >= 2 and m.shape[0] >= 2:
            rng = np.random.default_rng(0)
            n_boot = 1000
            boots = []
            np_, nm_ = p.shape[0], m.shape[0]
            if paired:
                # Resample cases; each case keeps both of its embeddings.
                diff = p - m
                for _ in range(n_boot):
                    ic = rng.integers(0, np_, size=np_)
                    boots.append(float(np.linalg.norm(diff[ic].mean(axis=0))))
            else:
                # Resample each group independently to its own size.
                for _ in range(n_boot):
                    ip = rng.integers(0, np_, size=np_)
                    im = rng.integers(0, nm_, size=nm_)
                    boots.append(
                        float(np.linalg.norm(p[ip].mean(axis=0) - m[im].mean(axis=0)))
                    )
            lo, hi = np.percentile(boots, [2.5, 97.5])
            out["ci_lower"] = float(lo)
            out["ci_upper"] = float(hi)
            out["ci_method"] = (
                "bootstrap (cases, pairs kept)" if paired else "bootstrap"
            )

            n_perm = 1000
            exceed = 0
            if paired:
                # Under the null the attribute label within a pair is arbitrary:
                # flip the sign of randomly chosen within-pair differences.
                diff = p - m
                for _ in range(n_perm):
                    signs = rng.choice((-1.0, 1.0), size=np_)[:, None]
                    d = np.linalg.norm((signs * diff).mean(axis=0))
                    exceed += d >= spg_euclidean - 1e-12
                null_txt = "within-pair label swaps"
            else:
                pooled = np.vstack([p, m])
                for _ in range(n_perm):
                    idx = rng.permutation(pooled.shape[0])
                    d = np.linalg.norm(
                        pooled[idx[:np_]].mean(axis=0) - pooled[idx[np_:]].mean(axis=0)
                    )
                    exceed += d >= spg_euclidean - 1e-12
                null_txt = "exchangeable group labels"
            p_perm = (exceed + 1) / (n_perm + 1)
            out["p_value_permutation"] = float(p_perm)
            out["n_permutations"] = n_perm
            out[
                "interpretation"
            ] += f" Permutation p = {p_perm:.3g} against {null_txt}."

        return MetricResult(out, name="SPG", value_key="spg_euclidean")
