"""Burden-Evidence Mismatch Index (BEMI).

BEMI is the total-variation distance between a corpus's geographic *evidence*
distribution and a disease-*burden* distribution over the same regions:

    BEMI = 0.5 * sum_r |evidence_share_r - burden_share_r|,  range [0, 1].

0 means evidence tracks burden perfectly; 1 means the two distributions are
completely disjoint. It equals the fraction of evidence that would need
geographic reallocation to match burden. Bounds proven and verified.
"""
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd


class BurdenEvidenceMismatch:
    """Geographic Burden-Evidence Mismatch Index (BEMI)."""

    def __init__(self):
        pass

    def calculate_bemi(
        self,
        evidence_counts: Dict[str, float],
        burden_shares: Dict[str, float],
        evidence_records: "Optional[Sequence[str]]" = None,
    ) -> Dict[str, Any]:
        """Calculate the Burden-Evidence Mismatch Index (BEMI).

        Args:
            evidence_counts: region -> number of studies (or cases) per region.
                Raw counts or shares; normalized to a distribution internally.
            burden_shares: region -> disease-burden share per region. Should sum
                to 1.0 (normalized internally if not). Use
                ``WHO_REGION_IHD_BURDEN`` for IHD DALY shares (Roth GA et al., 2020).

        Returns:
            Dict with keys:
              - ``bemi`` (float): total-variation distance in [0, 1]; 0 = evidence
                mirrors burden, 1 = completely disjoint.
              - ``evidence_shares`` (Dict[str, float]): normalized evidence shares.
              - ``burden_shares`` (Dict[str, float]): normalized burden shares.
              - ``per_region`` (pd.DataFrame): region, evidence_share, burden_share,
                mismatch (evidence - burden), ratio (evidence / burden).
              - ``most_underserved_region`` (str): region with the most negative
                mismatch (most under-represented relative to burden).
              - ``interpretation`` (str): human-readable verdict.
        """
        if not evidence_counts:
            raise ValueError("evidence_counts must be a non-empty mapping.")
        if not burden_shares:
            raise ValueError("burden_shares must be a non-empty mapping.")

        regions = sorted(set(evidence_counts) | set(burden_shares))
        a = np.array([float(evidence_counts.get(r, 0.0)) for r in regions])
        b = np.array([float(burden_shares.get(r, 0.0)) for r in regions])
        if np.any(a < 0) or np.any(b < 0):
            raise ValueError("evidence_counts and burden_shares must be non-negative.")
        if a.sum() <= 0 or b.sum() <= 0:
            raise ValueError(
                "evidence_counts and burden_shares must each have a positive total."
            )

        b_norm = b / b.sum()
        a = a / a.sum()
        b = b_norm
        mismatch = a - b
        ratio = np.divide(a, b, out=np.full_like(a, np.nan), where=b > 0)
        bemi = float(0.5 * np.abs(mismatch).sum())

        per_region = pd.DataFrame(
            {
                "region": regions,
                "evidence_share": a,
                "burden_share": b,
                "mismatch": mismatch,
                "ratio": ratio,
            }
        )
        underserved = str(per_region.loc[per_region["mismatch"].idxmin(), "region"])

        out = {
            "bemi": bemi,
            "evidence_shares": dict(zip(regions, a.tolist())),
            "burden_shares": dict(zip(regions, b.tolist())),
            "per_region": per_region,
            "most_underserved_region": underserved,
            "interpretation": (
                f"BEMI = {bemi:.3f} (range [0, 1]; 0 = evidence tracks burden, "
                f"1 = disjoint). About {bemi * 100:.1f}% of the evidence would need "
                f"geographic reallocation to match disease burden; the most "
                f"under-served region is {underserved}."
            ),
        }

        from equimed_dss.inference import MetricResult, bootstrap_ci

        # A CI cannot be computed honestly from aggregate shares. When the caller
        # supplies per-evidence region labels, resample those records and recompute
        # BEMI against the fixed burden distribution.
        if evidence_records is not None:
            recs = [str(r) for r in evidence_records]
            if len(recs) >= 2:
                def _bemi(sample):
                    counts: Dict[str, float] = {}
                    for r in sample:
                        counts[r] = counts.get(r, 0.0) + 1.0
                    av = np.array([counts.get(r, 0.0) for r in regions])
                    if av.sum() <= 0:
                        return 0.0
                    av = av / av.sum()
                    return float(0.5 * np.abs(av - b_norm).sum())

                ci = bootstrap_ci(recs, _bemi, n_boot=1000, random_state=0)
                out["ci_lower"] = ci.ci_lower
                out["ci_upper"] = ci.ci_upper
                out["ci_method"] = ci.method
        return MetricResult(out, name="BEMI", value_key="bemi")
