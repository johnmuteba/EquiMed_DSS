"""Weighted Clinical Harm-Adjusted Fairness Gap (wHAFG).

Standard fairness metrics treat all errors equally. wHAFG weights each error by
its clinical severity (for example a severity score derived from HEART/TIMI/GRACE
for ACS, or Wells/PERC for PE) so that disparities causing the most clinical harm
dominate.

For demographic group g with severity weights omega(Y_i) and losses L(Yhat_i, Y_i):
    H(g)   = (1/n_g) sum_i omega(Y_i) * L(Yhat_i, Y_i)
    wHAFG(g, g') = |H(g) - H(g')|
    wHAFG_max    = max_{g, g'} wHAFG(g, g')

Distinct from domain2.HarmAdjustedFairnessGap (HAFG), which compares two groups by
aggregate false-negative/false-positive counts with fixed costs; wHAFG is the
per-sample, clinical-severity-weighted generalization over arbitrary groups.
"""
from typing import Any, Dict, Sequence

import numpy as np


class WeightedClinicalHarmAdjustedFairnessGap:
    """Weighted Clinical Harm-Adjusted Fairness Gap (wHAFG)."""

    def __init__(self):
        pass

    def calculate_whafg(
        self,
        groups: Sequence[Any],
        severity_weights: Sequence[float],
        losses: Sequence[float],
    ) -> Dict[str, Any]:
        """Compute wHAFG across demographic groups.

        Args:
            groups: per-sample demographic group label.
            severity_weights: per-sample clinical severity weight omega(Y_i) in [0, 1].
            losses: per-sample loss L(Yhat_i, Y_i) (e.g. 0/1 misclassification).

        Returns:
            Dict with harm_by_group, whafg_max, most_harmed_group, and interpretation.
        """
        g = np.asarray(groups)
        w = np.asarray(severity_weights, dtype=float)
        loss = np.asarray(losses, dtype=float)
        if not (len(g) == len(w) == len(loss)):
            raise ValueError("groups, severity_weights, losses must be the same length.")
        if len(g) == 0:
            raise ValueError("Inputs must be non-empty.")

        harm_by_group = {
            str(grp): float((w[g == grp] * loss[g == grp]).mean())
            for grp in np.unique(g)
        }
        items = sorted(harm_by_group.items(), key=lambda kv: kv[1])
        whafg_max = float(items[-1][1] - items[0][1]) if len(items) > 1 else 0.0
        most_harmed = items[-1][0]

        return {
            "harm_by_group": harm_by_group,
            "whafg_max": whafg_max,
            "most_harmed_group": most_harmed,
            "n_groups": len(harm_by_group),
            "interpretation": (
                f"Maximum severity-weighted harm gap wHAFG = {whafg_max:.3f}; "
                f"highest weighted harm in group '{most_harmed}'."
            ),
        }
