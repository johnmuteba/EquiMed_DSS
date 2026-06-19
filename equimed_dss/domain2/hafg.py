from typing import Dict, Optional, Sequence, Union

import numpy as np


class HarmAdjustedFairnessGap:
    """
    Domain 2: Fairness, Equity, and Ethics Assessment
    Metric 5: Harm-Adjusted Fairness Gap (HAFG)

    Quantifies fairness weighted by potential clinical harm (cost of errors).
    """

    def __init__(self, cost_fn: float = 10.0, cost_fp: float = 3.0):
        """
        Initialize with costs for False Negatives and False Positives.

        Args:
            cost_fn: Cost of a false negative (default: 10).
            cost_fp: Cost of a false positive (default: 3).
        """
        self.cost_fn = cost_fn
        self.cost_fp = cost_fp

    def _harm_per_case(self, label: str) -> float:
        """Per-case harm contribution for an error label ('fn'/'fp', else 0)."""
        if label == "fn":
            return self.cost_fn
        if label == "fp":
            return self.cost_fp
        return 0.0

    def calculate_hafg(
        self,
        group1_errors: Dict[str, int],
        group2_errors: Dict[str, int],
        group1_cases: Optional[Sequence[str]] = None,
        group2_cases: Optional[Sequence[str]] = None,
    ) -> Dict[str, float]:
        """
        Calculate HAFG between two groups (e.g., Marginalized vs Privileged).

        Args:
            group1_errors: Dict with 'fn' (count) and 'fp' (count) for group 1.
            group2_errors: Dict with 'fn' (count) and 'fp' (count) for group 2.
            group1_cases / group2_cases: optional per-case error labels for each
                group (each element one of 'fn', 'fp', 'tp', 'tn'). When BOTH are
                supplied, HAFG gains a 95% percentile-bootstrap CI by resampling
                cases within each group. Without them a CI cannot be computed
                honestly from aggregate counts, and the result prints
                "95% CI unavailable (needs observation-level input)".

        Returns:
            MetricResult with harm for each group, the normalized gap (``hafg``),
            and -- when per-case labels are provided -- its 95% CI.
        """
        harm1 = (
            group1_errors.get("fn", 0) * self.cost_fn
            + group1_errors.get("fp", 0) * self.cost_fp
        )
        harm2 = (
            group2_errors.get("fn", 0) * self.cost_fn
            + group2_errors.get("fp", 0) * self.cost_fp
        )

        gap = abs(harm1 - harm2)
        # HAFG is normalized by the larger harm so it lies in [0, 1] and is
        # comparable across datasets: HAFG = |H1 - H2| / max(H1, H2).
        denom = max(harm1, harm2)
        hafg = float(gap / denom) if denom > 0 else 0.0

        if hafg < 0.1:
            verdict = "Minimal harm disparity"
        elif hafg < 0.2:
            verdict = "Moderate harm disparity"
        else:
            verdict = "Significant harm disparity"

        out = {
            "harm_group1": float(harm1),
            "harm_group2": float(harm2),
            "hafg": hafg,
            "absolute_harm_gap": float(gap),
            "ratio": float(harm1 / harm2) if harm2 > 0 else float("inf"),
            "interpretation": {
                "range": "[0, 1]",
                "ideal": "Lower is better (close to 0)",
                "verdict": verdict,
            },
        }

        from equimed_dss.inference import MetricResult, bootstrap_ci

        if group1_cases is not None and group2_cases is not None:
            # Tag each case with its group so a single resample preserves group
            # sizes only in expectation (standard two-sample bootstrap of HAFG).
            records = (
                [{"group": 1, "label": str(c)} for c in group1_cases]
                + [{"group": 2, "label": str(c)} for c in group2_cases]
            )
            if not records:
                raise ValueError("group1_cases/group2_cases contain no cases.")

            def _hafg(sample):
                h1 = sum(self._harm_per_case(r["label"]) for r in sample if r["group"] == 1)
                h2 = sum(self._harm_per_case(r["label"]) for r in sample if r["group"] == 2)
                d = max(h1, h2)
                return abs(h1 - h2) / d if d > 0 else 0.0

            ci = bootstrap_ci(records, _hafg, n_boot=1000, random_state=0)
            out["ci_lower"] = ci.ci_lower
            out["ci_upper"] = ci.ci_upper
            out["ci_method"] = ci.method

        return MetricResult(out, name="HAFG", value_key="hafg")
