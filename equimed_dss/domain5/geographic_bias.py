"""Geographic Representation Bias Index (GRBI).

GRBI measures how far a corpus's geographic distribution departs from the global
disease-burden distribution, using the Kullback-Leibler divergence:
    GRBI = D_KL(P_C || P_burden) = sum_r P_C(r) * log( P_C(r) / P_burden(r) )
A simplified high-income-country overrepresentation ratio:
    HIC_ratio = P_C(HIC) / P_burden(HIC)
(values > 1 indicate HIC overrepresentation).

GRBI (a directed KL divergence) complements the geographic Burden-Evidence
Mismatch (BEMI, a symmetric total-variation distance).
"""
from typing import Any, Dict, Optional, Sequence

import numpy as np


class GeographicRepresentationBiasIndex:
    """Geographic Representation Bias Index (GRBI), KL of corpus vs burden."""

    def __init__(self):
        pass

    def calculate_grbi(
        self,
        corpus_counts: Dict[str, float],
        burden_shares: Dict[str, float],
        hic_regions: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """Compute GRBI between a corpus geography and a disease-burden reference.

        Args:
            corpus_counts: region -> evidence count or share in the corpus.
            burden_shares: region -> disease-burden share (normalized internally);
                must be positive for every region present in corpus_counts.
            hic_regions: optional set of regions counted as high-income, for the
                HIC overrepresentation ratio.

        Returns:
            Dict with grbi (KL divergence in nats), hic_ratio (or None),
            corpus_shares, and interpretation.
        """
        if not corpus_counts:
            raise ValueError("corpus_counts must be non-empty.")
        if not burden_shares:
            raise ValueError("burden_shares must be non-empty.")

        regions = sorted(set(corpus_counts) | set(burden_shares))
        c = np.array([float(corpus_counts.get(r, 0.0)) for r in regions])
        b = np.array([float(burden_shares.get(r, 0.0)) for r in regions])
        if c.sum() <= 0 or b.sum() <= 0:
            raise ValueError("corpus_counts and burden_shares must have positive totals.")
        pc = c / c.sum()
        pb = b / b.sum()

        # KL divergence; terms with pc=0 contribute 0; pb must be > 0 there.
        grbi = 0.0
        for pci, pbi, r in zip(pc, pb, regions):
            if pci > 0:
                if pbi <= 0:
                    raise ValueError(
                        f"burden_shares for region {r!r} is zero but corpus has mass; "
                        "KL divergence is undefined."
                    )
                grbi += pci * np.log(pci / pbi)
        grbi = float(grbi)

        hic_ratio = None
        if hic_regions:
            hic = set(hic_regions)
            pc_hic = float(sum(pc[i] for i, r in enumerate(regions) if r in hic))
            pb_hic = float(sum(pb[i] for i, r in enumerate(regions) if r in hic))
            hic_ratio = float(pc_hic / pb_hic) if pb_hic > 0 else None

        return {
            "grbi": grbi,
            "hic_ratio": hic_ratio,
            "corpus_shares": dict(zip(regions, pc.tolist())),
            "interpretation": (
                f"GRBI (KL divergence corpus || burden) = {grbi:.3f} nats"
                + (
                    f"; HIC overrepresentation ratio = {hic_ratio:.2f}"
                    f" ({'over' if hic_ratio and hic_ratio > 1 else 'under'}-represented)."
                    if hic_ratio is not None
                    else "."
                )
            ),
        }
