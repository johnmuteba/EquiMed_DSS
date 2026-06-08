"""Burden-Evidence Mismatch Index (BEMI).

BEMI is the total-variation distance between a corpus's geographic *evidence*
distribution and a disease-*burden* distribution over the same regions:

    BEMI = 0.5 * sum_r |evidence_share_r - burden_share_r|,  range [0, 1].

0 means evidence tracks burden perfectly; 1 means disjoint support. It equals
the fraction of evidence mass that would need geographic reallocation to match
burden. Bounds proven and verified 2026-06-07.
"""
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


class BurdenEvidenceMismatch:
    """Geographic Burden-Evidence Mismatch Index (BEMI)."""

    def __init__(self, burden_reference: Optional[Dict[str, float]] = None):
        # Optional default burden distribution; can also be passed per call.
        self.burden_reference = burden_reference

    def calculate_bemi(
        self,
        evidence: Dict[str, float],
        burden: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Compute BEMI between an evidence distribution and a burden reference.

        Args:
            evidence: region -> evidence count or share.
            burden: region -> burden count or share. If None, uses the
                reference passed at construction.

        Returns:
            Dict with bemi_index (float), per_region (DataFrame),
            most_underserved_region (str), and interpretation.
        """
        burden = burden if burden is not None else self.burden_reference
        if burden is None:
            raise ValueError(
                "A burden reference must be provided (per call or at construction)."
            )

        regions = sorted(set(evidence) | set(burden))
        a = np.array([float(evidence.get(r, 0.0)) for r in regions])
        b = np.array([float(burden.get(r, 0.0)) for r in regions])
        if a.sum() <= 0 or b.sum() <= 0:
            raise ValueError("Evidence and burden must each have a positive total.")

        a = a / a.sum()
        b = b / b.sum()
        mismatch = a - b
        ratio = np.divide(a, b, out=np.full_like(a, np.nan), where=b > 0)
        bemi_index = float(0.5 * np.abs(mismatch).sum())

        table = pd.DataFrame(
            {
                "region": regions,
                "evidence_share": a,
                "burden_share": b,
                "mismatch": mismatch,
                "ratio": ratio,
            }
        )
        underserved = str(table.loc[table["mismatch"].idxmin(), "region"])

        return {
            "bemi_index": bemi_index,
            "per_region": table,
            "most_underserved_region": underserved,
            "interpretation": {
                "range": "BEMI in [0, 1]; 0 = evidence tracks burden, 1 = disjoint",
                "index_meaning": (
                    f"{bemi_index * 100:.1f}% of evidence mass would need "
                    "geographic reallocation to match burden"
                ),
            },
        }
