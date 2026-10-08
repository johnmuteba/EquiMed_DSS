"""Healthcare System Stratified Fairness (HSSF).

Healthcare-system type (single-payer, multi-payer, mixed) can confound
demographic fairness. HSSF measures the demographic gap within each system and
averages it over systems, weighted by their size:

    Delta_s       = max_g E[Y | G=g, S=s] - min_g E[Y | G=g, S=s]
    HSSF          = sum_s w_s * Delta_s,   w_s proportional to the observations
                    compared in system s

Only systems with at least two demographic groups (each with at least
``min_group_n`` observations) have a within-system gap; the others are listed in
``systems_without_comparison`` and excluded. (Up to 1.9.5 they counted as a gap
of 0, so systems in which only one group was observed made disparity look
smaller.)

The spread of mean outcomes BETWEEN systems is reported separately, in outcome
units (``between_system_sd`` and ``between_system_range``). The within-system
gap and the between-system spread are separate descriptors on the same scale,
not additive components of a total disparity. ``delta_between`` (the weighted
variance of system means, in squared outcome units) is kept for compatibility
and must not be compared with HSSF directly.
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
        min_group_n: int = 1,
    ) -> Dict[str, Any]:
        """Compute HSSF and the between-system spread of mean outcomes.

        Args:
            systems: per-sample healthcare-system label.
            groups: per-sample demographic group label.
            outcomes: per-sample outcome Y (e.g. 0/1 decision or rate).
            min_group_n: a group enters a system's comparison only with at least
                this many observations in that system (default 1; a larger
                value, e.g. 30, avoids gaps driven by a handful of patients).

        Returns:
            Dict with hssf (= delta_within), disparity_by_system,
            systems_without_comparison, between_system_sd,
            between_system_range, delta_between (variance, for compatibility),
            n_systems, interpretation, and a 95% bootstrap CI for HSSF that
            resamples within each system-by-group cell.

        Raises:
            ValueError: for mismatched or empty inputs, non-finite outcomes, an
                invalid ``min_group_n``, or when no system has two comparable
                groups.
        """
        s = np.asarray(systems)
        g = np.asarray(groups)
        y = np.asarray(outcomes, dtype=float)
        if not (len(s) == len(g) == len(y)):
            raise ValueError("systems, groups, outcomes must be the same length.")
        if len(s) == 0:
            raise ValueError("Inputs must be non-empty.")
        if not np.all(np.isfinite(y)):
            raise ValueError("outcomes must be finite.")
        if int(min_group_n) != min_group_n or min_group_n < 1:
            raise ValueError("min_group_n must be an integer >= 1.")

        # Cells (system, group) that enter each system's comparison.
        cells: Dict[str, Dict[str, np.ndarray]] = {}
        without = []
        system_means, system_sizes = [], []
        for sys in np.unique(s):
            ms = s == sys
            system_means.append(float(y[ms].mean()))
            system_sizes.append(int(ms.sum()))
            kept = {
                str(grp): y[ms & (g == grp)]
                for grp in np.unique(g[ms])
                if int((ms & (g == grp)).sum()) >= min_group_n
            }
            if len(kept) >= 2:
                cells[str(sys)] = kept
            else:
                without.append(str(sys))
        if not cells:
            raise ValueError(
                "No system has two groups with at least min_group_n observations, "
                "so no within-system gap can be computed."
            )
        weights = {k: sum(len(v) for v in c.values()) for k, c in cells.items()}
        total_w = float(sum(weights.values()))

        def _gaps(cell_map):
            return {
                k: float(
                    max(v.mean() for v in c.values())
                    - min(v.mean() for v in c.values())
                )
                for k, c in cell_map.items()
            }

        def _hssf(cell_map) -> float:
            gaps = _gaps(cell_map)
            return float(sum(weights[k] / total_w * gaps[k] for k in gaps))

        disparity_by_system = _gaps(cells)
        hssf = _hssf(cells)

        sm = np.array(system_means)
        sw = np.array(system_sizes, dtype=float) / len(y)
        grand = float((sm * sw).sum())
        delta_between = float((sw * (sm - grand) ** 2).sum())

        out = {
            "hssf": hssf,
            "delta_within": hssf,
            "disparity_by_system": disparity_by_system,
            "systems_without_comparison": without,
            "between_system_sd": float(np.sqrt(delta_between)),
            "between_system_range": float(sm.max() - sm.min()),
            "delta_between": delta_between,
            "n_systems": int(len(sm)),
            "interpretation": (
                f"HSSF (size-weighted within-system demographic gap, over "
                f"{len(cells)} system(s) with two or more comparable groups) = "
                f"{hssf:.3f}. Between systems, mean outcomes differ by up to "
                f"{float(sm.max() - sm.min()):.3f} (weighted SD "
                f"{float(np.sqrt(delta_between)):.3f}). These are separate "
                "descriptors in outcome units, not parts of one decomposition."
                + (
                    f" Excluded (fewer than two comparable groups): {without}."
                    if without
                    else ""
                )
            ),
        }

        from equimed_dss.inference import MetricResult, stratified_bootstrap_ci

        strata = {(k, grp): v for k, c in cells.items() for grp, v in c.items()}

        def _hssf_boot(smp):
            cell_map: Dict[str, Dict[str, np.ndarray]] = {}
            for (sys_key, grp), v in smp.items():
                cell_map.setdefault(sys_key, {})[grp] = v
            return _hssf(cell_map)

        if sum(len(v) for v in strata.values()) >= 2:
            ci = stratified_bootstrap_ci(
                strata,
                _hssf_boot,
                n_boot=1000,
                random_state=0,
                label="system-group cell",
            )
            out["ci_lower"] = ci.ci_lower
            out["ci_upper"] = ci.ci_upper
            out["ci_method"] = ci.method
        return MetricResult(out, name="HSSF", value_key="hssf")
