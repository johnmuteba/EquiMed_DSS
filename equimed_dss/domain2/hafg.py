import warnings
from typing import Dict, Optional, Sequence, Union

import numpy as np


class HarmAdjustedFairnessGap:
    """
    Domain 2: Fairness, Equity, and Ethics Assessment
    Metric 5: Harm-Adjusted Fairness Gap (HAFG)

    Quantifies fairness weighted by potential clinical harm (cost of errors).

    HAFG compares the TOTAL harm of two groups (error counts x costs). When the
    groups differ in size, the gap reflects group size as well as error rates:
    pass error counts per 1,000 patients, or use
    ``domain5.WeightedClinicalHarmAdjustedFairnessGap``, which averages harm per
    patient.
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
                cases within each group (group sizes fixed). Without them a CI
                cannot be computed honestly from aggregate counts, and the result
                prints "95% CI unavailable (needs observation-level input)". A
                warning is raised if the case lists disagree with the error
                counts, or if the groups differ in size by more than 10%.

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

        from equimed_dss.inference import MetricResult

        if group1_cases is not None and group2_cases is not None:
            labels1 = [str(c) for c in group1_cases]
            labels2 = [str(c) for c in group2_cases]
            if not labels1 and not labels2:
                raise ValueError("group1_cases/group2_cases contain no cases.")
            for name, labels, errors in (
                ("group1", labels1, group1_errors),
                ("group2", labels2, group2_errors),
            ):
                tally = {k: labels.count(k) for k in ("fn", "fp")}
                if tally != {k: int(errors.get(k, 0)) for k in ("fn", "fp")}:
                    warnings.warn(
                        f"{name}_cases has {tally} but {name}_errors has "
                        f"{dict(errors)}; the interval describes the case lists, "
                        "not the reported HAFG.",
                        UserWarning,
                        stacklevel=2,
                    )
            n1, n2 = len(labels1), len(labels2)
            if min(n1, n2) > 0 and max(n1, n2) / min(n1, n2) > 1.1:
                warnings.warn(
                    f"The groups differ in size ({n1} vs {n2} cases). HAFG compares "
                    "total harm, so the gap reflects group size as well as error "
                    "rates; use counts per 1,000 patients or "
                    "WeightedClinicalHarmAdjustedFairnessGap (harm per patient).",
                    UserWarning,
                    stacklevel=2,
                )

            # Two-sample bootstrap: resample cases within each group so the
            # group sizes stay fixed. (Up to 1.9.5 the two groups were pooled, so
            # resampled group sizes varied and the interval was too wide.)
            c1 = np.array([self._harm_per_case(c) for c in labels1], dtype=float)
            c2 = np.array([self._harm_per_case(c) for c in labels2], dtype=float)
            rng = np.random.default_rng(0)
            boots = []
            for _ in range(1000):
                h1 = c1[rng.integers(0, n1, size=n1)].sum() if n1 else 0.0
                h2 = c2[rng.integers(0, n2, size=n2)].sum() if n2 else 0.0
                d = max(h1, h2)
                boots.append(abs(h1 - h2) / d if d > 0 else 0.0)
            lo, hi = np.percentile(boots, [2.5, 97.5])
            out["ci_lower"] = float(lo)
            out["ci_upper"] = float(hi)
            out["ci_method"] = "bootstrap (stratified by group)"

        return MetricResult(out, name="HAFG", value_key="hafg")
