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
"""
from itertools import combinations
from math import factorial
from typing import Any, Dict, Sequence

import numpy as np


class IntersectionalShapleyFairnessValue:
    """Intersectional Shapley Fairness Value (ISFV)."""

    def __init__(self, min_cell: int = 1):
        # minimum samples per conditioning cell to count toward a disparity
        self.min_cell = min_cell

    def _v(self, attr_values: Dict[str, np.ndarray], outcomes: np.ndarray,
           subset: tuple) -> float:
        if not subset:
            return 0.0
        keys = list(zip(*[attr_values[a] for a in subset]))
        means = {}
        counts = {}
        for k, y in zip(keys, outcomes):
            means.setdefault(k, []).append(y)
        cell_means = [float(np.mean(v)) for k, v in means.items() if len(v) >= self.min_cell]
        if len(cell_means) < 2:
            return 0.0
        return float(max(cell_means) - min(cell_means))

    def calculate_isfv(
        self,
        attributes: Dict[str, Sequence[Any]],
        outcomes: Sequence[float],
    ) -> Dict[str, Any]:
        """Compute Shapley attribution of disparity to each protected attribute.

        Args:
            attributes: mapping attribute name -> per-sample values
                (e.g. {"race": [...], "gender": [...]}).
            outcomes: per-sample outcome Y.

        Returns:
            Dict with shapley_by_attribute, total_disparity, interactions
            (pairwise), and interpretation.
        """
        names = list(attributes)
        if not names:
            raise ValueError("attributes must be non-empty.")
        av = {a: np.asarray(attributes[a]) for a in names}
        y = np.asarray(outcomes, dtype=float)
        n = len(y)
        if any(len(av[a]) != n for a in names) or n == 0:
            raise ValueError("All attribute arrays and outcomes must share length > 0.")
        m = len(names)

        shapley = {}
        for a in names:
            others = [x for x in names if x != a]
            phi = 0.0
            for r in range(len(others) + 1):
                for S in combinations(others, r):
                    w = factorial(len(S)) * factorial(m - len(S) - 1) / factorial(m)
                    phi += w * (self._v(av, y, tuple(S) + (a,)) - self._v(av, y, tuple(S)))
            shapley[a] = float(phi)

        total = self._v(av, y, tuple(names))  # v(A) - v(empty), with v(empty)=0
        interactions = {}
        for ai, aj in combinations(names, 2):
            inter = (
                self._v(av, y, (ai, aj))
                - self._v(av, y, (ai,))
                - self._v(av, y, (aj,))
            )
            interactions[f"{ai} x {aj}"] = float(inter)

        top = max(shapley, key=shapley.get) if shapley else None
        return {
            "shapley_by_attribute": shapley,
            "total_disparity": float(total),
            "interactions": interactions,
            "interpretation": (
                "Shapley attribution of the intersectional disparity: "
                + ", ".join(f"{k}={v:.3f}" for k, v in shapley.items())
                + (f"; largest single-attribute contribution: {top}." if top else ".")
                + " Positive interaction indicates a superadditive (intersectional) penalty."
            ),
        }
