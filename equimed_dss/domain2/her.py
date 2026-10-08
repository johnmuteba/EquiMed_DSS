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

        HER_g = s_g / s_ref. The ratio is meaningful only for non-negative scores on
        a ratio scale (rates, proportions, accuracies). Each group is labelled
        "within" or "outside" the 0.8-1.25 band of the four-fifths convention;
        the label describes the ratio, not whether a difference is fair or
        clinically important, and the same band applies whichever direction of
        the score is better.

        Args:
            group_scores: Dictionary mapping group names to their performance scores.
            reference_group: Name of the reference group (default: 'White').
            group_observations: optional mapping group -> per-observation scores. When
                supplied, the reported scalar (the max-min HER gap across groups) gains
                a 95% percentile-bootstrap confidence interval that resamples
                observations within each group (group sizes fixed); each group's
                score is recomputed as the mean of its resampled observations.
                Without it, the gap is reported without a CI (a CI cannot be
                computed honestly from a single aggregate score per group).

        Returns:
            MetricResult mapping each group name to its HER (``{"score", ...}``) and
            also carrying the scalar ``her_gap`` (and a 95% CI when
            ``group_observations`` is given). Printing shows the HER gap with its CI.

        Raises:
            ValueError: if the reference group is missing or its score is not
                positive (the ratio is then undefined; up to 1.9.5 a zero reference
                silently gave every group HER = 0), or if a score is negative or
                not finite.
        """
        if reference_group not in group_scores:
            raise ValueError(f"Reference group '{reference_group}' not found in scores")
        vals = np.asarray([float(v) for v in group_scores.values()], dtype=float)
        if not np.all(np.isfinite(vals)) or np.any(vals < 0):
            raise ValueError("group_scores must be finite and non-negative.")

        reference_score = float(group_scores[reference_group])
        if reference_score <= 0:
            raise ValueError(
                f"The reference group '{reference_group}' has a score of 0, so the "
                "equity ratio is undefined; choose another reference group or report "
                "absolute differences instead."
            )
        ratios = {}
        for group, score in group_scores.items():
            ratios[group] = float(score) / reference_score

        her_scores = {}
        for group, val in ratios.items():
            verdict = (
                "Within the 0.8-1.25 band"
                if 0.8 <= val <= 1.25
                else "Outside the 0.8-1.25 band"
            )
            her_scores[group] = {
                "score": float(val),
                "interpretation": {
                    "range": "[0, inf)",
                    "ideal": "Close to 1 (0.8 - 1.25, four-fifths convention)",
                    "verdict": verdict,
                },
            }

        # The printed scalar is the spread of HER across groups (max - min); a
        # value of 0 means every group matches the reference equally. It is kept
        # OFF the returned mapping (carried as the MetricResult point/ci instead)
        # so the dict holds only per-group entries and stays cleanly iterable:
        # ``for g, r in result.items(): r["score"]`` works unchanged.
        her_gap = float(max(ratios.values()) - min(ratios.values())) if ratios else 0.0

        from equimed_dss.inference import MetricResult, stratified_bootstrap_ci

        ci_tuple = None
        if group_observations is not None:
            missing = set(group_scores) - set(group_observations)
            if missing:
                raise ValueError(
                    f"group_observations missing groups: {sorted(missing)}"
                )
            strata = {
                g: np.asarray(group_observations[g], dtype=float) for g in group_scores
            }
            if any(len(v) == 0 for v in strata.values()):
                raise ValueError(
                    "every group in group_observations needs observations."
                )

            def _gap(sample):
                rs = float(np.mean(sample[reference_group]))
                if rs <= 0:
                    return float("nan")  # undefined ratio: replicate dropped
                rr = [float(np.mean(v)) / rs for v in sample.values()]
                return max(rr) - min(rr)

            # Resample within each group, so every replicate keeps every group,
            # the reference included. (Up to 1.9.5 the groups were pooled, and a
            # replicate without the reference returned the point estimate.)
            ci = stratified_bootstrap_ci(strata, _gap, n_boot=1000, random_state=0)
            ci_tuple = (ci.ci_lower, ci.ci_upper, ci.method)

        return MetricResult(her_scores, name="HER (gap)", point=her_gap, ci=ci_tuple)

    def calculate_bias_gini(
        self,
        scores: Optional[List[float]] = None,
        group_observations: Optional[Dict[str, Sequence[float]]] = None,
    ) -> Dict[str, float]:
        """
        Calculate Bias-Gini Index to measure dispersion of fairness metrics.

        Args:
            scores: List of performance scores across groups (non-negative). May be
                omitted when ``group_observations`` is given; the scores are then
                the group means.
            group_observations: optional mapping group -> per-observation scores.
                When supplied, the index gains a 95% percentile-bootstrap CI that
                resamples observations within each group. If ``scores`` is also
                given, it must equal the group means (in the mapping's order).

        Returns:
            MetricResult with ``bias_gini``. Without observation-level input the
            result prints "95% CI unavailable": the group scores are fixed values,
            not a sample of groups, so resampling them (as versions up to 1.9.5
            did) does not give a sampling interval.
        """
        from equimed_dss.inference import MetricResult, stratified_bootstrap_ci

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

        strata = None
        if group_observations is not None:
            strata = {
                g: np.asarray(v, dtype=float) for g, v in group_observations.items()
            }
            if not strata or any(len(v) == 0 for v in strata.values()):
                raise ValueError(
                    "every group in group_observations needs observations."
                )
            means = [float(np.mean(v)) for v in strata.values()]
            if scores is None:
                scores = means
            elif len(scores) != len(means) or not np.allclose(scores, means):
                raise ValueError(
                    "scores must equal the group means of group_observations, in the "
                    "same order."
                )
        if scores is None:
            raise ValueError("give scores, group_observations, or both.")
        arr = np.asarray(scores, dtype=float)
        if not np.all(np.isfinite(arr)) or np.any(arr < 0):
            raise ValueError("scores must be finite and non-negative.")

        gini = _gini(scores)
        out = {"bias_gini": gini, "n_groups": len(scores)}

        if strata is not None and len(strata) >= 2:
            ci = stratified_bootstrap_ci(
                strata,
                lambda smp: _gini([float(np.mean(v)) for v in smp.values()]),
                n_boot=1000,
                random_state=0,
            )
            out["ci_lower"] = ci.ci_lower
            out["ci_upper"] = ci.ci_upper
            out["ci_method"] = ci.method

        return MetricResult(out, name="Bias-Gini", value_key="bias_gini")
