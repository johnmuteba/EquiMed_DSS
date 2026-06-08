"""Geographic Concentration of Coverage (GCC).

Two complementary descriptors of how a corpus's evidence (e.g. included
studies) is spread across regions:

- Sample-corrected Gini, G* = (R / (R-1)) * G_raw, range [0, 1].
  0 = perfectly even coverage, 1 = all evidence in one region. The R/(R-1)
  factor is required because the raw Gini for R categories maxes out at
  (R-1)/R, so without it the index could not reach 1.
- Normalized Shannon entropy, H_norm = -sum_r p_r ln(p_r) / ln(R), range [0, 1].
  1 = perfectly even, 0 = single-region concentration.

G* and H_norm run in opposite directions; ``concentration = 1 - H_norm`` is
exposed so a single "higher = more concentrated" reading is available.
"""
from typing import Any, Dict

import numpy as np
import pandas as pd


class GeographicConcentration:
    """Geographic Concentration of Coverage (GCC)."""

    def __init__(self):
        pass

    def calculate_gcc(self, region_counts: Dict[str, float]) -> Dict[str, Any]:
        """Calculate the Geographic Concentration of Coverage (GCC).

        Args:
            region_counts: region -> number of studies (or cases) per region
                (non-negative; raw counts or shares).

        Returns:
            Dict with keys:
              - ``gini_corrected`` (float): sample-corrected Gini G* in [0, 1];
                0 = even, 1 = single-region.
              - ``entropy_normalized`` (float): normalized Shannon entropy H_norm
                in [0, 1]; 1 = even, 0 = single-region (opposite to G*).
              - ``concentration`` (float): 1 - H_norm; higher = more concentrated.
              - ``n_regions`` (int): number of regions.
              - ``per_region`` (pd.DataFrame): region, evidence_share, sorted
                descending by share.
              - ``interpretation`` (str): human-readable summary.
        """
        if not region_counts:
            raise ValueError("region_counts must be a non-empty mapping.")
        regions = sorted(region_counts)
        x = np.array([float(region_counts[r]) for r in regions])
        if np.any(x < 0):
            raise ValueError("region_counts must be non-negative.")
        if x.sum() <= 0:
            raise ValueError("region_counts total must be positive.")
        R = len(x)
        if R < 2:
            raise ValueError("Need at least 2 regions to measure concentration.")

        gini_raw = np.abs(x[:, None] - x[None, :]).sum() / (2 * R * x.sum())
        gini_corrected = float(gini_raw * R / (R - 1))

        p = x / x.sum()
        nz = p[p > 0]
        entropy_normalized = float(-(nz * np.log(nz)).sum() / np.log(R))
        concentration = float(1.0 - entropy_normalized)

        per_region = (
            pd.DataFrame({"region": regions, "evidence_share": p})
            .sort_values("evidence_share", ascending=False)
            .reset_index(drop=True)
        )

        return {
            "gini_corrected": gini_corrected,
            "entropy_normalized": entropy_normalized,
            "concentration": concentration,
            "n_regions": R,
            "per_region": per_region,
            "interpretation": (
                f"G* = {gini_corrected:.3f} (0 = even, 1 = single-region); "
                f"H_norm = {entropy_normalized:.3f} (1 = even, 0 = single-region); "
                f"concentration = {concentration:.3f} (higher = more concentrated)."
            ),
        }
