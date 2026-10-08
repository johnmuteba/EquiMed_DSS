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

from equimed_dss._validation import check_records


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
                Evidence of unknown origin should be left out (and its share
                reported separately), not passed as a region.
            burden_shares: region -> disease-burden share per region. Should sum
                to 1.0 (normalized internally if not). ``WHO_REGION_IHD_BURDEN``
                holds WHO Global Health Estimates 2023 IHD DALY shares. Every
                region with evidence must appear here (give 0 explicitly if a
                region truly carries no burden); a region missing from this
                mapping raises ``ValueError``, because mismatched region codes
                (for example "AFR" against "AFRO") would otherwise be scored as
                evidence outside every burden region.
            evidence_records: optional per-evidence region labels (one element
                per study or case). When supplied, BEMI gains a 95% percentile-
                bootstrap CI by resampling these records against the fixed burden
                distribution. The labels must use the same region codes.

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
        unmatched = sorted(
            r
            for r, v in evidence_counts.items()
            if r not in burden_shares and float(v) > 0
        )
        if unmatched:
            raise ValueError(
                f"evidence_counts has regions with no entry in burden_shares: "
                f"{unmatched}. Check that both mappings use the same region codes "
                "(WHO_REGION_CODES maps WHO GHO codes such as 'AFR' to the 'AFRO' "
                "keys of WHO_REGION_IHD_BURDEN); give a burden share of 0 "
                "explicitly if a region truly carries no burden."
            )

        regions = sorted(set(evidence_counts) | set(burden_shares))
        a = np.array([float(evidence_counts.get(r, 0.0)) for r in regions])
        b = np.array([float(burden_shares.get(r, 0.0)) for r in regions])
        if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
            raise ValueError("evidence_counts and burden_shares must be finite.")
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
            check_records(
                recs, regions, dict(zip(regions, a.tolist())), "evidence_records"
            )
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
