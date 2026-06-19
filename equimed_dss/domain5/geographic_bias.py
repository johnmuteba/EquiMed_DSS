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
        corpus_records: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """Compute GRBI between a corpus geography and a disease-burden reference.

        Args:
            corpus_counts: region -> evidence count or share in the corpus.
            burden_shares: region -> disease-burden share (normalized internally);
                must be positive for every region present in corpus_counts.
            hic_regions: optional set of regions counted as high-income, for the
                HIC overrepresentation ratio.
            corpus_records: optional sequence of per-evidence region labels (one
                element per record). When supplied, GRBI gains a 95% percentile-
                bootstrap CI by resampling evidence records. Without it a CI cannot
                be computed honestly from aggregate shares, and the result prints
                "95% CI unavailable (needs observation-level input)".

        Returns:
            MetricResult with grbi (KL divergence in nats), hic_ratio (or None),
            corpus_shares, and -- when ``corpus_records`` is given -- a 95% CI.
        """
        if not corpus_counts:
            raise ValueError("corpus_counts must be non-empty.")
        if not burden_shares:
            raise ValueError("burden_shares must be non-empty.")

        regions = sorted(set(corpus_counts) | set(burden_shares))
        b = np.array([float(burden_shares.get(r, 0.0)) for r in regions])
        if b.sum() <= 0:
            raise ValueError("burden_shares must have a positive total.")
        pb = b / b.sum()
        pb_map = dict(zip(regions, pb))

        def _kl(counts: Dict[str, float]) -> float:
            c = np.array([float(counts.get(r, 0.0)) for r in regions])
            if c.sum() <= 0:
                raise ValueError("corpus_counts must have a positive total.")
            pc = c / c.sum()
            kl = 0.0
            for pci, r in zip(pc, regions):
                if pci > 0:
                    pbi = pb_map[r]
                    if pbi <= 0:
                        raise ValueError(
                            f"burden_shares for region {r!r} is zero but corpus has "
                            "mass; KL divergence is undefined."
                        )
                    kl += pci * np.log(pci / pbi)
            return float(kl)

        c = np.array([float(corpus_counts.get(r, 0.0)) for r in regions])
        if c.sum() <= 0:
            raise ValueError("corpus_counts must have a positive total.")
        pc = c / c.sum()
        grbi = _kl(corpus_counts)

        hic_ratio = None
        if hic_regions:
            hic = set(hic_regions)
            pc_hic = float(sum(pc[i] for i, r in enumerate(regions) if r in hic))
            pb_hic = float(sum(pb[i] for i, r in enumerate(regions) if r in hic))
            hic_ratio = float(pc_hic / pb_hic) if pb_hic > 0 else None

        out = {
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

        from equimed_dss.inference import MetricResult, bootstrap_ci

        if corpus_records is not None:
            recs = [str(r) for r in corpus_records]
            if len(recs) >= 2:
                def _grbi(sample):
                    counts: Dict[str, float] = {}
                    for r in sample:
                        counts[r] = counts.get(r, 0.0) + 1.0
                    return _kl(counts)

                ci = bootstrap_ci(recs, _grbi, n_boot=1000, random_state=0)
                out["ci_lower"] = ci.ci_lower
                out["ci_upper"] = ci.ci_upper
                out["ci_method"] = ci.method
        return MetricResult(out, name="GRBI", value_key="grbi")
