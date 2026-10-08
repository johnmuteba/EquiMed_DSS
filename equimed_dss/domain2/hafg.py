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
        group1_n: Optional[int] = None,
        group2_n: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Calculate HAFG between two groups (e.g., Marginalized vs Privileged).

        Args:
            group1_errors: Dict with 'fn' (count) and 'fp' (count) for group 1.
            group2_errors: Dict with 'fn' (count) and 'fp' (count) for group 2.
            group1_cases / group2_cases: optional per-case error labels for each
                group (each element one of 'fn', 'fp', 'tp', 'tn'). When BOTH are
                supplied, HAFG gains a 95% percentile-bootstrap CI by resampling
                cases within each group (group sizes fixed); without them a CI
                cannot be computed honestly from aggregate counts, and the result
                prints "95% CI unavailable (needs observation-level input)". The
                case lists must agree with the error counts (``ValueError``
                otherwise), and they give the group sizes.
            group1_n / group2_n: optional group sizes (number of patients) when
                case lists are not given.

        Returns:
            MetricResult with harm for each group, the normalized gap (``hafg``,
            total harm), and, when the group sizes are known, the harm per patient
            in each group and ``hafg_per_patient``, the gap that compares groups
            of different sizes fairly. A warning recommends it when the sizes
            differ by more than 10%.
        """
        for name, errors in (("group1", group1_errors), ("group2", group2_errors)):
            for k in ("fn", "fp"):
                v = errors.get(k, 0)
                if not np.isfinite(float(v)) or float(v) < 0:
                    raise ValueError(
                        f"{name}_errors['{k}'] must be a non-negative count."
                    )
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
            verdict = "Large harm disparity"

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
                "note": "Compares total harm; see hafg_per_patient when the "
                "groups differ in size.",
            },
        }

        from equimed_dss.inference import MetricResult

        labels1 = labels2 = None
        if group1_cases is not None and group2_cases is not None:
            labels1 = [str(c) for c in group1_cases]
            labels2 = [str(c) for c in group2_cases]
            if not labels1 or not labels2:
                raise ValueError(
                    "group1_cases and group2_cases must both be non-empty."
                )
            for name, labels, errors in (
                ("group1", labels1, group1_errors),
                ("group2", labels2, group2_errors),
            ):
                tally = {k: labels.count(k) for k in ("fn", "fp")}
                if tally != {k: int(errors.get(k, 0)) for k in ("fn", "fp")}:
                    raise ValueError(
                        f"{name}_cases has {tally} but {name}_errors has "
                        f"{dict(errors)}; the case lists must describe the same "
                        "errors as the counts (up to 1.9.5 this only warned, and the "
                        "interval then described a different estimate)."
                    )
            for name, labels, n_given in (
                ("group1", labels1, group1_n),
                ("group2", labels2, group2_n),
            ):
                if n_given is not None and int(n_given) != len(labels):
                    raise ValueError(
                        f"{name}_n disagrees with the length of {name}_cases."
                    )
            group1_n, group2_n = len(labels1), len(labels2)

        if group1_n is not None and group2_n is not None:
            n1, n2 = int(group1_n), int(group2_n)
            if n1 <= 0 or n2 <= 0:
                raise ValueError("group sizes must be positive.")
            for name, errors, n_g in (
                ("group1", group1_errors, n1),
                ("group2", group2_errors, n2),
            ):
                if errors.get("fn", 0) + errors.get("fp", 0) > n_g:
                    raise ValueError(f"{name} has more errors than patients.")
            pp1, pp2 = harm1 / n1, harm2 / n2
            d_pp = max(pp1, pp2)
            out["harm_per_patient_group1"] = float(pp1)
            out["harm_per_patient_group2"] = float(pp2)
            out["hafg_per_patient"] = float(abs(pp1 - pp2) / d_pp) if d_pp > 0 else 0.0
            out["group1_n"], out["group2_n"] = n1, n2
            if max(n1, n2) / min(n1, n2) > 1.1:
                warnings.warn(
                    f"The groups differ in size ({n1} vs {n2}). HAFG compares total "
                    "harm, so it reflects group size as well as error rates; report "
                    "hafg_per_patient (harm per patient) for a fair comparison.",
                    UserWarning,
                    stacklevel=2,
                )

        if labels1 is not None:
            # Two-sample bootstrap: resample cases within each group so the
            # group sizes stay fixed. (Up to 1.9.5 the two groups were pooled, so
            # resampled group sizes varied and the interval was too wide.)
            n1, n2 = len(labels1), len(labels2)
            c1 = np.array([self._harm_per_case(c) for c in labels1], dtype=float)
            c2 = np.array([self._harm_per_case(c) for c in labels2], dtype=float)
            rng = np.random.default_rng(0)
            boots = []
            for _ in range(1000):
                h1 = c1[rng.integers(0, n1, size=n1)].sum()
                h2 = c2[rng.integers(0, n2, size=n2)].sum()
                d = max(h1, h2)
                boots.append(abs(h1 - h2) / d if d > 0 else 0.0)
            lo, hi = np.percentile(boots, [2.5, 97.5])
            out["ci_lower"] = float(lo)
            out["ci_upper"] = float(hi)
            out["ci_method"] = "bootstrap (stratified by group)"

        return MetricResult(out, name="HAFG", value_key="hafg")
