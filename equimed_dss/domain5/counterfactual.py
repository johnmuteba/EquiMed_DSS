"""Counterfactual and robustness parity metrics: CPS and SRPI.

These take precomputed response-similarity scores (for example cosine similarity
of response embeddings or BERTScore); generating the responses is the caller's
responsibility.
"""

from typing import Any, Dict, Sequence, Union

import numpy as np


class CounterfactualParityScore:
    """Counterfactual Parity Score (CPS).

    For a query x and protected-attribute swap a -> a', CS = sim(f(x), f(x_{A<-a'})).
    CPS(a, a') = mean_i CS_i; counterfactual unfairness CFU = 1 - min_{a,a'} CPS(a, a').
    Distinct from DecisionFlipRate (binary decision change) and SemanticParityGap
    (embedding-centroid distance): CPS is the continuous semantic similarity of the
    full response under a demographic swap.
    """

    def __init__(self):
        pass

    def calculate_cps(
        self,
        similarities: Union[Sequence[float], Dict[str, Sequence[float]]],
    ) -> Dict[str, Any]:
        """Compute CPS and CFU from counterfactual response similarities.

        Args:
            similarities: either a flat sequence of per-case similarities (single
                attribute-value pair), or a mapping {pair_label: [similarities]}
                for multiple swapped pairs. Similarities must be on a 0-1 scale
                (1 = identical response), so that CFU = 1 - CPS lies in [0, 1].
                Rescale a cosine similarity c in [-1, 1] as (1 + c) / 2 first.

        Returns:
            Dict with cps (overall mean), cps_by_pair, cfu, n, interpretation, and
            95% bootstrap CIs for CPS (``ci_lower``/``ci_upper``) and for CFU
            (``cfu_ci_lower``/``cfu_ci_upper``), resampling within each pair.

        Raises:
            ValueError: for empty input, or similarities that are missing or
                outside [0, 1].
        """
        if isinstance(similarities, dict):
            if not similarities:
                raise ValueError("similarities mapping must be non-empty.")
            strata = {}
            for pair, sims in similarities.items():
                s = np.asarray(sims, dtype=float)
                if s.size == 0:
                    raise ValueError(f"Pair {pair!r} has no similarities.")
                strata[str(pair)] = s
            single = False
        else:
            s = np.asarray(similarities, dtype=float)
            if s.size == 0:
                raise ValueError("similarities must be non-empty.")
            strata = {"overall": s}
            single = True
        allsim = np.concatenate(list(strata.values()))
        if not np.all(np.isfinite(allsim)) or np.any((allsim < 0) | (allsim > 1)):
            raise ValueError(
                "similarities must be finite and lie in [0, 1]; rescale a cosine "
                "similarity c as (1 + c) / 2."
            )

        def _cps(smp):
            return float(np.mean(np.concatenate(list(smp.values()))))

        def _cfu(smp):
            return float(1.0 - min(float(v.mean()) for v in smp.values()))

        cps_by_pair = {k: float(v.mean()) for k, v in strata.items()}
        cps = _cps(strata)
        cfu = _cfu(strata) if not single else float(1.0 - cps)
        n = int(allsim.size)

        out = {
            "cps": cps,
            "cps_by_pair": cps_by_pair,
            "cfu": cfu,
            "n": n,
            "interpretation": (
                f"CPS = {cps:.3f} (mean response similarity under demographic swap; "
                f"1 = identical responses); counterfactual unfairness "
                f"CFU = {cfu:.3f} (1 minus the lowest pair mean)."
            ),
        }

        # Percentile bootstrap that resamples cases within each swapped pair.
        from equimed_dss.inference import MetricResult, stratified_bootstrap_ci

        if n >= 2:
            ci = stratified_bootstrap_ci(
                strata, _cps, n_boot=1000, random_state=0, label="pair"
            )
            out["ci_lower"] = ci.ci_lower
            out["ci_upper"] = ci.ci_upper
            out["ci_method"] = ci.method
            ci_cfu = stratified_bootstrap_ci(
                strata, _cfu, n_boot=1000, random_state=0, label="pair"
            )
            out["cfu_ci_lower"] = ci_cfu.ci_lower
            out["cfu_ci_upper"] = ci_cfu.ci_upper
        return MetricResult(out, name="CPS", value_key="cps")


class SemanticRobustnessParityIndex:
    """Semantic Robustness Parity Index (SRPI).

    Per-query robustness R(x) = mean pairwise similarity of responses to
    semantically-equivalent paraphrases; R(g) = mean_x R(x) within group g;
    SRPI = min_g R(g) / max_g R(g), in [0, 1] (1 = equal robustness across groups).
    SRPI describes parity only: equal but low robustness also gives 1, so read it
    together with the robustness magnitudes it reports. Distinct from
    EmbeddingConsistencyScore / ObservedPerturbationAgreement, which measure
    robustness magnitude rather than its parity across groups.
    """

    def __init__(self):
        pass

    def calculate_srpi(
        self, robustness_by_group: Dict[str, Sequence[float]]
    ) -> Dict[str, Any]:
        """Compute SRPI from per-query robustness scores grouped by demographic.

        Args:
            robustness_by_group: mapping group -> sequence of per-query robustness
                scores R(x) in [0, 1].

        Returns:
            Dict with robustness_by_group (means), min_robustness,
            max_robustness, srpi, least_robust_group, interpretation and a 95%
            bootstrap CI that resamples queries within each group. SRPI is NaN
            (undefined) when every group has zero robustness (up to 1.9.5 it was
            0, which read as maximal disparity).

        Raises:
            ValueError: for fewer than 2 groups, an empty group, or scores that
                are missing or outside [0, 1].
        """
        if len(robustness_by_group) < 2:
            raise ValueError("Need at least 2 groups.")
        strata = {}
        for grp, scores in robustness_by_group.items():
            s = np.asarray(scores, dtype=float)
            if s.size == 0:
                raise ValueError(f"Group {grp!r} has no robustness scores.")
            if not np.all(np.isfinite(s)) or np.any((s < 0) | (s > 1)):
                raise ValueError(
                    f"Group {grp!r}: robustness scores must lie in [0, 1]."
                )
            strata[str(grp)] = s
        rg = {k: float(v.mean()) for k, v in strata.items()}
        mx = max(rg.values())
        srpi = float(min(rg.values()) / mx) if mx > 0 else float("nan")
        least = min(rg, key=rg.get)

        def _srpi(smp) -> float:
            mvals = [float(v.mean()) for v in smp.values()]
            m = max(mvals)
            return (min(mvals) / m) if m > 0 else float("nan")

        out = {
            "robustness_by_group": rg,
            "min_robustness": float(min(rg.values())),
            "max_robustness": float(mx),
            "srpi": srpi,
            "least_robust_group": least,
            "interpretation": (
                f"SRPI = {srpi:.3f} (1 = equal paraphrase robustness across groups); "
                f"robustness ranges from {min(rg.values()):.3f} to {mx:.3f}; lowest "
                f"in group '{least}'."
                if mx > 0
                else "SRPI undefined: every group has zero robustness."
            ),
        }

        from equimed_dss.inference import MetricResult, stratified_bootstrap_ci

        if mx > 0:
            ci = stratified_bootstrap_ci(strata, _srpi, n_boot=1000, random_state=0)
            out["ci_lower"] = ci.ci_lower
            out["ci_upper"] = ci.ci_upper
            out["ci_method"] = ci.method
        return MetricResult(out, name="SRPI", value_key="srpi")
