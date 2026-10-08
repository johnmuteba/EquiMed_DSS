"""Intersectional Shapley Fairness Value (ISFV).

When bias appears at an intersection (for example Black women), ISFV uses
cooperative-game Shapley values to attribute the disparity fairly to each
protected attribute and their interaction.

With protected attributes A and a bias characteristic function
    v(S) = max_{a, a' in dom(S)} | E[Y | A_S = a] - E[Y | A_S = a'] |,  v(empty) = 0,
the Shapley value of attribute A_i is
    phi_i = sum_{S subset A\\{i}} [ |S|! (m-|S|-1)! / m! ] ( v(S U {i}) - v(S) ),
and the pairwise interaction is
    I(A_i, A_j) = v({i, j}) - v({i}) - v({j}) + v(empty).

Distinct from domain2.IntersectionalBiasScore (IBS), which uses subgroup-similarity
matrices and an ANOVA-style interaction; ISFV gives a game-theoretic attribution
of the disparity to each attribute and their interaction.

v(S) is a range (largest minus smallest cell mean), which grows with the number
of cells and is driven by small cells even when outcomes do not differ, so an
attribute with more categories tends to receive a larger share. Set ``min_cell``
to exclude small cells (for example 30) and compare the attribution with a
permutation of the outcomes before interpreting it.
"""

import warnings
from itertools import combinations
from math import factorial
from typing import Any, Dict, Sequence

import numpy as np


class IntersectionalShapleyFairnessValue:
    """Intersectional Shapley Fairness Value (ISFV)."""

    def __init__(self, min_cell: int = 1):
        # minimum samples per conditioning cell to count toward a disparity
        self.min_cell = min_cell

    def _v(self, codes, card, y, subset) -> float:
        """Range of cell means over the cross-classification of ``subset``."""
        if not subset:
            return 0.0
        key = np.zeros(len(y), dtype=np.int64)
        for a in subset:
            key = np.unique(key * card[a] + codes[a], return_inverse=True)[1]
        counts = np.bincount(key)
        sums = np.bincount(key, weights=y)
        mask = counts >= max(int(self.min_cell), 1)
        if mask.sum() < 2:
            return 0.0
        means = sums[mask] / counts[mask]
        return float(means.max() - means.min())

    def _shapley(self, names, codes, card, y):
        m = len(names)
        shapley = {}
        for a in names:
            others = [x for x in names if x != a]
            phi = 0.0
            for r in range(len(others) + 1):
                for S in combinations(others, r):
                    w = factorial(len(S)) * factorial(m - len(S) - 1) / factorial(m)
                    phi += w * (
                        self._v(codes, card, y, tuple(S) + (a,))
                        - self._v(codes, card, y, tuple(S))
                    )
            shapley[a] = float(phi)
        return shapley

    def calculate_isfv(
        self,
        attributes: Dict[str, Sequence[Any]],
        outcomes: Sequence[float],
    ) -> Dict[str, Any]:
        """Compute Shapley attribution of disparity to each protected attribute.

        Args:
            attributes: mapping attribute name -> per-sample values
                (e.g. {"race": [...], "gender": [...]}); no missing values.
            outcomes: per-sample outcome Y (finite).

        Returns:
            Dict with shapley_by_attribute, total_disparity, interactions
            (pairwise), interpretation, a 95% bootstrap CI for the total
            disparity (``ci_lower``/``ci_upper``) and for each attribute's share
            (``shapley_ci``), both resampling within the full cross-classified
            cells, and a permutation check of the total disparity against
            shuffled outcomes (``p_value_permutation``,
            ``total_disparity_null_mean``: the range expected from noise alone).
        """
        import pandas as pd

        names = list(attributes)
        if not names:
            raise ValueError("attributes must be non-empty.")
        y = np.asarray(outcomes, dtype=float)
        n = len(y)
        if n == 0 or any(len(attributes[a]) != n for a in names):
            raise ValueError("All attribute arrays and outcomes must share length > 0.")
        if not np.all(np.isfinite(y)):
            raise ValueError("outcomes must be finite.")
        codes, card = {}, {}
        for a in names:
            vals = pd.Series(list(attributes[a]))
            if vals.isna().any():
                raise ValueError(f"attribute {a!r} has missing values.")
            codes[a] = pd.factorize(vals)[0].astype(np.int64)
            card[a] = int(codes[a].max()) + 1

        shapley = self._shapley(names, codes, card, y)
        total = self._v(codes, card, y, tuple(names))  # v(A) - v(empty), v(empty)=0
        interactions = {}
        for ai, aj in combinations(names, 2):
            inter = (
                self._v(codes, card, y, (ai, aj))
                - self._v(codes, card, y, (ai,))
                - self._v(codes, card, y, (aj,))
            )
            interactions[f"{ai} x {aj}"] = float(inter)

        full = np.zeros(n, dtype=np.int64)
        for a in names:
            full = np.unique(full * card[a] + codes[a], return_inverse=True)[1]
        cell_sizes = np.bincount(full)
        if self.min_cell < 5 and cell_sizes.min() < 5:
            warnings.warn(
                f"{int((cell_sizes < 5).sum())} cross-classified cell(s) have fewer than "
                "5 observations; the range-based disparity is driven by small cells. "
                "Set min_cell (e.g. 30) and check p_value_permutation.",
                UserWarning,
                stacklevel=2,
            )

        top = max(shapley, key=shapley.get) if shapley else None
        out = {
            "shapley_by_attribute": shapley,
            "total_disparity": float(total),
            "interactions": interactions,
            "interpretation": (
                "Shapley attribution of the intersectional disparity (range of cell "
                "means): "
                + ", ".join(f"{k}={v:.3f}" for k, v in shapley.items())
                + (f"; largest single-attribute contribution: {top}." if top else ".")
                + " The pairwise interaction compares ranges and has no direction: a "
                "positive value is not by itself an intersectional penalty."
            ),
        }

        from equimed_dss.inference import MetricResult

        if n >= 2:
            rng = np.random.default_rng(0)
            strata = [np.flatnonzero(full == c) for c in range(len(cell_sizes))]
            boot_total, boot_phi = [], {a: [] for a in names}
            for _ in range(500):
                idx = np.concatenate(
                    [st[rng.integers(0, len(st), size=len(st))] for st in strata]
                )
                c_b = {a: codes[a][idx] for a in names}
                y_b = y[idx]
                boot_total.append(self._v(c_b, card, y_b, tuple(names)))
                for a, v in self._shapley(names, c_b, card, y_b).items():
                    boot_phi[a].append(v)
            lo, hi = np.percentile(boot_total, [2.5, 97.5])
            out["ci_lower"] = float(lo)
            out["ci_upper"] = float(hi)
            out["ci_method"] = "bootstrap (stratified by cell)"
            out["shapley_ci"] = {
                a: tuple(float(x) for x in np.percentile(v, [2.5, 97.5]))
                for a, v in boot_phi.items()
            }
            null = []
            for _ in range(200):
                null.append(self._v(codes, card, rng.permutation(y), tuple(names)))
            null = np.array(null)
            out["total_disparity_null_mean"] = float(null.mean())
            out["p_value_permutation"] = float(
                (1 + np.sum(null >= total - 1e-12)) / (1 + len(null))
            )
        return MetricResult(out, name="ISFV", value_key="total_disparity")
