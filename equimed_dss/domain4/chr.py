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
        threshold: float = 0.05,
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

        # Uncertainty: CHR is a binomial proportion, so report a Wilson 95% CI
        # and a one-sided score test that the true rate exceeds an acceptable
        # threshold (default 5%).
        from equimed_dss.inference import MetricResult, proportion_ci

        inf = proportion_ci(int(unsupported.sum()), int(s.size),
                            null_value=threshold, alternative="greater")
        p_txt = "<0.001" if inf.p_value < 0.001 else f"{inf.p_value:.3g}"

        return MetricResult({
            "chr": chr_value,
            "chr_weighted": chr_weighted,
            "n_claims": int(s.size),
            "n_unsupported": int(unsupported.sum()),
            "tau": float(tau),
            "ci_lower": inf.ci_lower,
            "ci_upper": inf.ci_upper,
            "ci_method": inf.method,
            "threshold": float(threshold),
            "p_value_above_threshold": inf.p_value,
            "interpretation": (
                f"CHR = {chr_value:.3f} (95% CI {inf.ci_lower:.3f} to "
                f"{inf.ci_upper:.3f}; {int(unsupported.sum())} of {s.size} "
                f"claims unsupported at tau={tau}); severity-weighted "
                f"CHR = {chr_weighted:.3f}. One-sided p={p_txt} that "
                f"the true rate exceeds {threshold:.0%}. Higher is worse."
            ),
        }, name="CHR", value_key="chr")
