"""Geographic Concentration of Coverage (GCC).

Two complementary descriptors of how a corpus's evidence is spread across
regions:

- Gini (sample-corrected): G* = (R / (R-1)) * G_raw, range [0, 1].
  0 = perfectly even, 1 = all evidence in one region. The R/(R-1) factor is
  required because raw Gini for R categories maxes out at (R-1)/R.
- Normalized Shannon entropy: H_norm = -sum_r p_r ln p_r / ln R, range [0, 1].
  1 = even coverage, 0 = single-region concentration.

Gini and entropy run in opposite directions; `concentration = 1 - H_norm` is
exposed so a single "higher = more concentrated" reading is available.
Verified 2026-06-07.
"""
from typing import Any, Dict

import numpy as np
import pandas as pd


class GeographicConcentration:
    """Geographic Concentration of Coverage (GCC)."""

    def __init__(self):
        pass

    def calculate_gcc(self, evidence: Dict[str, float]) -> Dict[str, Any]:
        """Compute concentration descriptors for a geographic evidence vector.

        Args:
            evidence: region -> evidence count or share (non-negative).

        Returns:
            Dict with gini, normalized_entropy, concentration, n_regions,
            per_region (DataFrame), and interpretation.
        """
        regions = sorted(evidence)
        x = np.array([float(evidence[r]) for r in regions])
        if np.any(x < 0):
            raise ValueError("Evidence values must be non-negative.")
        if x.sum() <= 0:
            raise ValueError("Evidence total must be positive.")
        R = len(x)
        if R < 2:
            raise ValueError("Need at least 2 regions to measure concentration.")

        gini_raw = np.abs(x[:, None] - x[None, :]).sum() / (2 * R * x.sum())
        gini = float(gini_raw * R / (R - 1))

        p = x / x.sum()
        nz = p[p > 0]
        entropy = float(-(nz * np.log(nz)).sum() / np.log(R))
        concentration = float(1.0 - entropy)

        table = (
            pd.DataFrame({"region": regions, "evidence_share": p})
            .sort_values("evidence_share", ascending=False)
            .reset_index(drop=True)
        )

        return {
            "gini": gini,
            "normalized_entropy": entropy,
            "concentration": concentration,
            "n_regions": R,
            "per_region": table,
            "interpretation": {
                "gini": "0 = even coverage, 1 = single-region (sample-corrected)",
                "entropy": "1 = even coverage, 0 = single-region concentration",
            },
        }
