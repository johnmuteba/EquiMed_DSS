"""Geographic Representation Index (GRI).

GRI quantifies the spatial variety of a knowledge base. Let L be the set of
unique locations in the corpus and W the subset identified as Western /
high-income:
    GRI = (|L| - |W|) / |L|
GRI -> 0 means an almost entirely Western-centric set of locations; GRI -> 1
means none of the represented locations are Western. GRI is set-based (variety),
complementary to volume-based geographic measures.

Geographic Bias (GB) is the correlation between the GRI of retrieved documents
and the error rate for non-Western patients across queries:
    GB = corr( GRI(K), error_rate )
"""
from typing import Any, Dict, Optional, Sequence

import numpy as np


class GeographicRepresentationIndex:
    """Geographic Representation Index (GRI) and Geographic Bias (GB)."""

    def __init__(self):
        pass

    def calculate_gri(
        self,
        locations: Sequence[str],
        western_locations: Sequence[str],
    ) -> Dict[str, Any]:
        """Compute the Geographic Representation Index.

        Args:
            locations: locations represented in the corpus (duplicates allowed;
                only the unique set is used).
            western_locations: locations counted as Western / high-income.

        Returns:
            Dict with gri, n_locations, n_western, n_non_western,
            non_western_locations, and interpretation.
        """
        L = set(locations)
        if not L:
            raise ValueError("locations must be non-empty.")
        W = set(western_locations) & L
        non_western = sorted(L - W)
        gri = float((len(L) - len(W)) / len(L))
        return {
            "gri": gri,
            "n_locations": len(L),
            "n_western": len(W),
            "n_non_western": len(non_western),
            "non_western_locations": non_western,
            "interpretation": (
                f"GRI = {gri:.3f}; {len(non_western)} of {len(L)} represented "
                "locations are non-Western. Values near 0 indicate a "
                "Western-centric knowledge base (by variety of locations)."
            ),
        }

    def calculate_geographic_bias(
        self,
        gri_values: Sequence[float],
        error_rates: Sequence[float],
        method: str = "pearson",
    ) -> Dict[str, Any]:
        """Compute Geographic Bias (GB): correlation of GRI with error rate.

        Args:
            gri_values: per-query GRI of the retrieved documents.
            error_rates: per-query error rate for non-Western patients (paired).
            method: "pearson" or "spearman".

        Returns:
            Dict with gb (correlation), p_value, n, method, and interpretation.
        """
        x = np.asarray(gri_values, dtype=float)
        y = np.asarray(error_rates, dtype=float)
        if x.shape != y.shape:
            raise ValueError("gri_values and error_rates must be paired.")
        if x.size < 3:
            raise ValueError("Need at least 3 paired points for a correlation.")
        if method == "pearson":
            from scipy.stats import pearsonr

            r, p = pearsonr(x, y)
        elif method == "spearman":
            from scipy.stats import spearmanr

            r, p = spearmanr(x, y)
        else:
            raise ValueError("method must be 'pearson' or 'spearman'.")
        return {
            "gb": float(r),
            "p_value": float(p),
            "n": int(x.size),
            "method": method,
            "interpretation": (
                f"Geographic Bias = {float(r):+.3f} ({method}, p={float(p):.3g}). "
                "A negative correlation means lower geographic representation "
                "tracks higher error for non-Western patients."
            ),
        }
