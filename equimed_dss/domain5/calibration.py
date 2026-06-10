"""Intersectional Calibration Error (ICE).

Standard Expected Calibration Error (ECE) aggregates over all samples and can
mask severe miscalibration in specific intersectional subgroups. ICE computes a
group-specific ECE at the intersection of protected attributes and aggregates
with population weighting.

For an intersectional group i with confidence-binned samples S_{i,b}:
    ECE_i = sum_b (|S_{i,b}| / |S_i|) * |acc(S_{i,b}) - conf(S_{i,b})|
    ICE   = sum_i w_i * ECE_i,   w_i = |S_i| / sum_j |S_j|
    dICE  = max_{i,j} |ECE_i - ECE_j|
"""
from typing import Any, Dict, Sequence

import numpy as np


class IntersectionalCalibrationError:
    """Intersectional Calibration Error (ICE) and the maximum gap dICE."""

    def __init__(self):
        pass

    def calculate_ice(
        self,
        groups: Sequence[Any],
        confidences: Sequence[float],
        correct: Sequence[int],
        n_bins: int = 10,
    ) -> Dict[str, Any]:
        """Compute ICE across intersectional groups.

        Args:
            groups: per-sample intersectional group label (e.g. "Black|F").
            confidences: per-sample predicted confidence in [0, 1].
            correct: per-sample correctness indicator (1 correct, 0 incorrect).
            n_bins: number of equal-width confidence bins.

        Returns:
            Dict with ice, delta_ice, ece_by_group, n_groups, and interpretation.
        """
        g = np.asarray(groups)
        conf = np.asarray(confidences, dtype=float)
        corr = np.asarray(correct, dtype=float)
        if not (len(g) == len(conf) == len(corr)):
            raise ValueError("groups, confidences, correct must be the same length.")
        if len(g) == 0:
            raise ValueError("Inputs must be non-empty.")
        if np.any((conf < 0) | (conf > 1)):
            raise ValueError("confidences must lie in [0, 1].")

        edges = np.linspace(0.0, 1.0, n_bins + 1)
        ece_by_group: Dict[str, float] = {}
        sizes: Dict[str, int] = {}
        for grp in np.unique(g):
            mask = g == grp
            c = conf[mask]
            y = corr[mask]
            n = len(c)
            sizes[str(grp)] = n
            ece = 0.0
            for b in range(n_bins):
                lo, hi = edges[b], edges[b + 1]
                inbin = (c > lo) & (c <= hi) if b > 0 else (c >= lo) & (c <= hi)
                if inbin.sum() == 0:
                    continue
                acc = float(y[inbin].mean())
                avg_conf = float(c[inbin].mean())
                ece += (inbin.sum() / n) * abs(acc - avg_conf)
            ece_by_group[str(grp)] = float(ece)

        total = sum(sizes.values())
        ice = float(sum(sizes[k] / total * ece_by_group[k] for k in ece_by_group))
        vals = list(ece_by_group.values())
        delta_ice = float(max(vals) - min(vals)) if len(vals) > 1 else 0.0

        if delta_ice < 0.02:
            verdict = "negligible disparity"
        elif delta_ice < 0.05:
            verdict = "moderate disparity"
        elif delta_ice < 0.10:
            verdict = "substantial disparity"
        else:
            verdict = "severe disparity"

        return {
            "ice": ice,
            "delta_ice": delta_ice,
            "ece_by_group": ece_by_group,
            "n_groups": len(ece_by_group),
            "interpretation": (
                f"ICE = {ice:.3f}; maximum intersectional calibration gap "
                f"dICE = {delta_ice:.3f} ({verdict})."
            ),
        }
