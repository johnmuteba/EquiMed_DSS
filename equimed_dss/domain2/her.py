from typing import Dict, List, Optional, Sequence, Union

import numpy as np


class HierarchicalEquityRatio:
    """
    Domain 2: Fairness, Equity, and Ethics Assessment
    Metric 4: Hierarchical Equity Ratio (HER)

    Calculates HER (ratio of metric for group vs reference) and Bias-Gini Index
    to assess multi-level inequities.
    """

    def __init__(self):
        pass

    def calculate_her(
        self,
        group_scores: Dict[str, float],
        reference_group: str = "White",
        group_observations: Optional[Dict[str, Sequence[float]]] = None,
    ) -> Dict[str, float]:
        """
        Calculate Hierarchical Equity Ratio for each group relative to a reference group.

        Args:
            group_scores: Dictionary mapping group names to their performance scores.
            reference_group: Name of the reference group (default: 'White').
            group_observations: optional mapping group -> per-observation scores. When
                supplied, the reported scalar (the max-min HER gap across groups) gains
                a 95% percentile-bootstrap confidence interval; each group's score is
                recomputed as the mean of its resampled observations. Without it, the
                gap is reported without a CI (a CI cannot be computed honestly from a
                single aggregate score per group).

        Returns:
            MetricResult mapping each group name to its HER (``{"score", ...}``) and
            also carrying the scalar ``her_gap`` (and a 95% CI when
            ``group_observations`` is given). Printing shows the HER gap with its CI.
        """
        if reference_group not in group_scores:
            raise ValueError(f"Reference group '{reference_group}' not found in scores")

        reference_score = group_scores[reference_group]
        ratios = {}
        for group, score in group_scores.items():
            ratios[group] = 0.0 if reference_score == 0 else score / reference_score

        her_scores = {}
        for group, val in ratios.items():
            verdict = "Equitable" if 0.8 <= val <= 1.25 else "Disparity Detected"
            her_scores[group] = {
                "score": float(val),
                "interpretation": {
                    "range": "[0, inf)",
                    "ideal": "Close to 1 (0.8 - 1.25)",
                    "verdict": verdict,
                },
            }

        # The printed scalar is the spread of HER across groups (max - min); a
        # value of 0 means every group matches the reference equally. It is kept
        # OFF the returned mapping (carried as the MetricResult point/ci instead)
        # so the dict holds only per-group entries and stays cleanly iterable:
        # ``for g, r in result.items(): r["score"]`` works unchanged.
        her_gap = float(max(ratios.values()) - min(ratios.values())) if ratios else 0.0

        from equimed_dss.inference import MetricResult, bootstrap_ci

        ci_tuple = None
        if group_observations is not None:
            missing = set(group_scores) - set(group_observations)
            if missing:
                raise ValueError(
                    f"group_observations missing groups: {sorted(missing)}"
                )
            # Flat record list with a group tag, so a single percentile bootstrap
            # resamples observations and recomputes the HER gap each time.
            records = [
                {"group": g, "value": float(v)}
                for g, obs in group_observations.items()
                for v in obs
            ]
            if not records:
                raise ValueError("group_observations contains no observations.")

            def _gap(sample):
                means: Dict[str, list] = {}
                for r in sample:
                    means.setdefault(r["group"], []).append(r["value"])
                if reference_group not in means:
                    return her_gap
                rs = float(np.mean(means[reference_group]))
                rr = [
                    (0.0 if rs == 0 else float(np.mean(v)) / rs)
                    for v in means.values()
                ]
                return max(rr) - min(rr)

            ci = bootstrap_ci(records, _gap, n_boot=1000, random_state=0)
            ci_tuple = (ci.ci_lower, ci.ci_upper, ci.method)

        return MetricResult(her_scores, name="HER (gap)", point=her_gap, ci=ci_tuple)

    def calculate_bias_gini(self, scores: List[float]) -> Dict[str, float]:
        """
        Calculate Bias-Gini Index to measure dispersion of fairness metrics.

        Args:
            scores: List of performance scores across groups.

        Returns:
            MetricResult with ``bias_gini`` plus a 95% percentile-bootstrap CI over
            the group scores. Printing shows the Bias-Gini Index with its CI; the
            point estimate is still available as ``result["bias_gini"]``.
        """
        from equimed_dss.inference import MetricResult, bootstrap_ci

        def _gini(vals) -> float:
            vals = list(vals)
            if not vals:
                return 0.0
            n = len(vals)
            mean_score = np.mean(vals)
            if mean_score == 0:
                return 0.0
            gini_sum = sum(abs(vals[i] - vals[j]) for i in range(n) for j in range(n))
            return float(gini_sum / (2 * n * n * mean_score))

        gini = _gini(scores)
        out = {"bias_gini": gini, "n_groups": len(scores)}

        # A CI needs at least two group scores to resample meaningfully.
        if len(scores) >= 2:
            ci = bootstrap_ci(list(scores), _gini, n_boot=1000, random_state=0)
            out["ci_lower"] = ci.ci_lower
            out["ci_upper"] = ci.ci_upper
            out["ci_method"] = ci.method

        return MetricResult(out, name="Bias-Gini", value_key="bias_gini")
