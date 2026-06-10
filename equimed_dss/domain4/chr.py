"""Clinical Hallucination Rate (CHR).

In a retrieval-augmented system, a hallucination is a generated clinical claim
that is not supported by the retrieved context. Given per-claim support scores
S(c, K) in [0, 1] (for example from a natural-language-inference / entailment
model) and an entailment threshold tau, CHR is the proportion of unsupported
claims:
    CHR = (1/|C|) sum_c I(S(c, K) < tau)
A severity-weighted variant uses per-claim weights w_c:
    CHR_weighted = sum_c w_c I(S(c, K) < tau) / sum_c w_c

This function takes the precomputed support scores; computing entailment itself
(claim extraction, NLI) is the caller's responsibility.
"""
from typing import Any, Dict, Optional, Sequence

import numpy as np


class ClinicalHallucinationRate:
    """Clinical Hallucination Rate (CHR) from per-claim support scores."""

    def __init__(self):
        pass

    def calculate_chr(
        self,
        support_scores: Sequence[float],
        tau: float = 0.5,
        weights: Optional[Sequence[float]] = None,
    ) -> Dict[str, Any]:
        """Compute the Clinical Hallucination Rate.

        Args:
            support_scores: per-claim support scores S(c, K) in [0, 1]; a claim is
                a hallucination when its score is below ``tau``.
            tau: entailment threshold in [0, 1] (default 0.5).
            weights: optional per-claim clinical-severity weights for the weighted
                variant; must match the length of ``support_scores``.

        Returns:
            Dict with chr, chr_weighted, n_claims, n_unsupported, tau, and
            interpretation.
        """
        s = np.asarray(support_scores, dtype=float)
        if s.size == 0:
            raise ValueError("support_scores must be non-empty.")
        if not 0.0 <= tau <= 1.0:
            raise ValueError("tau must be in [0, 1].")

        unsupported = (s < tau).astype(float)
        chr_value = float(unsupported.mean())

        if weights is not None:
            w = np.asarray(weights, dtype=float)
            if w.shape != s.shape:
                raise ValueError("weights must match the length of support_scores.")
            if w.sum() <= 0:
                raise ValueError("weights must have a positive total.")
            chr_weighted = float((w * unsupported).sum() / w.sum())
        else:
            chr_weighted = chr_value

        return {
            "chr": chr_value,
            "chr_weighted": chr_weighted,
            "n_claims": int(s.size),
            "n_unsupported": int(unsupported.sum()),
            "tau": float(tau),
            "interpretation": (
                f"CHR = {chr_value:.3f} ({int(unsupported.sum())} of {s.size} "
                f"claims unsupported at tau={tau}); severity-weighted "
                f"CHR = {chr_weighted:.3f}. Higher is worse."
            ),
        }
