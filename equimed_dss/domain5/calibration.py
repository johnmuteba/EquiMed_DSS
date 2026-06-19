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

        def _ice_from(gv, cv, yv) -> float:
            """Population-weighted ICE over intersectional groups."""
            ebg = {}
            szs = {}
            for grp in np.unique(gv):
                mask = gv == grp
                c = cv[mask]
                y = yv[mask]
                n = len(c)
                szs[str(grp)] = n
                ece = 0.0
                for b in range(n_bins):
                    lo, hi = edges[b], edges[b + 1]
                    inbin = (c > lo) & (c <= hi) if b > 0 else (c >= lo) & (c <= hi)
                    if inbin.sum() == 0:
                        continue
                    ece += (inbin.sum() / n) * abs(float(y[inbin].mean()) - float(c[inbin].mean()))
                ebg[str(grp)] = float(ece)
            tot = sum(szs.values())
            return float(sum(szs[k] / tot * ebg[k] for k in ebg)), ebg

        ice, ece_by_group = _ice_from(g, conf, corr)
        sizes = {str(grp): int((g == grp).sum()) for grp in np.unique(g)}
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

        out = {
            "ice": ice,
            "delta_ice": delta_ice,
            "ece_by_group": ece_by_group,
            "n_groups": len(ece_by_group),
            "interpretation": (
                f"ICE = {ice:.3f}; maximum intersectional calibration gap "
                f"dICE = {delta_ice:.3f} ({verdict})."
            ),
        }

        # Percentile bootstrap over samples (resample (group, conf, correct)
        # triples and recompute the population-weighted ICE).
        from equimed_dss.inference import MetricResult, bootstrap_ci

        if len(g) >= 2:
            idx = list(range(len(g)))
            ci = bootstrap_ci(
                idx,
                lambda s: _ice_from(g[list(s)], conf[list(s)], corr[list(s)])[0],
                n_boot=1000, random_state=0,
            )
            out["ci_lower"] = ci.ci_lower
            out["ci_upper"] = ci.ci_upper
            out["ci_method"] = ci.method
        return MetricResult(out, name="ICE", value_key="ice")
