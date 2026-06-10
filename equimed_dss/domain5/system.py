"""Healthcare System Stratified Fairness (HSSF).

Healthcare-system type (single-payer, multi-payer, mixed) can confound
demographic fairness. HSSF measures demographic disparity within each system and
weights by system prevalence, separating the within-system demographic gap from
the between-system gap.

    Delta_s(g, g') = | E[Y | G=g, S=s] - E[Y | G=g', S=s] |
    HSSF           = sum_s P(S=s) * max_{g, g'} Delta_s(g, g')
    Delta_within   = sum_s P(s) * max_{g,g'} Delta_s(g, g')   (= HSSF)
    Delta_between  = Var_s( E[Y | S=s] )
"""
from typing import Any, Dict, Sequence

import numpy as np


class HealthcareSystemStratifiedFairness:
    """Healthcare System Stratified Fairness (HSSF)."""

    def __init__(self):
        pass

    def calculate_hssf(
        self,
        systems: Sequence[Any],
        groups: Sequence[Any],
        outcomes: Sequence[float],
    ) -> Dict[str, Any]:
        """Compute HSSF and the within/between-system disparity decomposition.

        Args:
            systems: per-sample healthcare-system label.
            groups: per-sample demographic group label.
            outcomes: per-sample outcome Y (e.g. 0/1 decision or rate).

        Returns:
            Dict with hssf, delta_within, delta_between, disparity_by_system,
            and interpretation.
        """
        s = np.asarray(systems)
        g = np.asarray(groups)
        y = np.asarray(outcomes, dtype=float)
        if not (len(s) == len(g) == len(y)):
            raise ValueError("systems, groups, outcomes must be the same length.")
        if len(s) == 0:
            raise ValueError("Inputs must be non-empty.")

        n = len(s)
        disparity_by_system: Dict[str, float] = {}
        hssf = 0.0
        system_means = []
        system_weights = []
        for sys in np.unique(s):
            mask = s == sys
            p_s = mask.sum() / n
            ys = y[mask]
            gs = g[mask]
            grp_means = [ys[gs == grp].mean() for grp in np.unique(gs) if (gs == grp).any()]
            max_gap = float(max(grp_means) - min(grp_means)) if len(grp_means) > 1 else 0.0
            disparity_by_system[str(sys)] = max_gap
            hssf += p_s * max_gap
            system_means.append(float(ys.mean()))
            system_weights.append(p_s)

        hssf = float(hssf)
        # Between-system variance of system mean outcomes (population-weighted).
        sm = np.array(system_means)
        sw = np.array(system_weights)
        grand = float((sm * sw).sum())
        delta_between = float((sw * (sm - grand) ** 2).sum())

        return {
            "hssf": hssf,
            "delta_within": hssf,
            "delta_between": delta_between,
            "disparity_by_system": disparity_by_system,
            "n_systems": len(disparity_by_system),
            "interpretation": (
                f"HSSF (population-weighted within-system demographic gap) = "
                f"{hssf:.3f}; between-system variance = {delta_between:.3f}. "
                "Compare the two to see whether observed disparity is demographic "
                "or driven by system differences."
            ),
        }
