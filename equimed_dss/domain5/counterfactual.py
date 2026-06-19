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
                for multiple swapped pairs.

        Returns:
            Dict with cps (overall mean), cps_by_pair, cfu, n, and interpretation.
        """
        if isinstance(similarities, dict):
            if not similarities:
                raise ValueError("similarities mapping must be non-empty.")
            cps_by_pair = {}
            allsim = []
            for pair, sims in similarities.items():
                s = np.asarray(sims, dtype=float)
                if s.size == 0:
                    raise ValueError(f"Pair {pair!r} has no similarities.")
                cps_by_pair[str(pair)] = float(s.mean())
                allsim.extend(s.tolist())
            cps = float(np.mean(allsim))
            cfu = float(1.0 - min(cps_by_pair.values()))
            n = len(allsim)
        else:
            s = np.asarray(similarities, dtype=float)
            if s.size == 0:
                raise ValueError("similarities must be non-empty.")
            cps = float(s.mean())
            cps_by_pair = {"overall": cps}
            cfu = float(1.0 - cps)
            n = int(s.size)
            allsim = s.tolist()

        out = {
            "cps": cps,
            "cps_by_pair": cps_by_pair,
            "cfu": cfu,
            "n": n,
            "interpretation": (
                f"CPS = {cps:.3f} (mean response similarity under demographic swap; "
                f"1 = perfect counterfactual parity); counterfactual unfairness "
                f"CFU = {cfu:.3f}."
            ),
        }

        # CPS is a mean over per-case similarities: a percentile bootstrap over the
        # pooled similarities gives its 95% CI.
        from equimed_dss.inference import MetricResult, bootstrap_ci

        if len(allsim) >= 2:
            ci = bootstrap_ci(list(allsim), lambda x: float(np.mean(x)),
                              n_boot=1000, random_state=0)
            out["ci_lower"] = ci.ci_lower
            out["ci_upper"] = ci.ci_upper
            out["ci_method"] = ci.method
        return MetricResult(out, name="CPS", value_key="cps")


class SemanticRobustnessParityIndex:
    """Semantic Robustness Parity Index (SRPI).

    Per-query robustness R(x) = mean pairwise similarity of responses to
    semantically-equivalent paraphrases; R(g) = mean_x R(x) within group g;
    SRPI = min_g R(g) / max_g R(g), in [0, 1] (1 = equal robustness across groups).
    Distinct from EmbeddingConsistencyScore / RobustnessCertificationScore, which
    measure robustness magnitude rather than its parity across groups.
    """

    def __init__(self):
        pass

    def calculate_srpi(self, robustness_by_group: Dict[str, Sequence[float]]) -> Dict[str, Any]:
        """Compute SRPI from per-query robustness scores grouped by demographic.

        Args:
            robustness_by_group: mapping group -> sequence of per-query robustness
                scores R(x) in [0, 1].

        Returns:
            Dict with robustness_by_group (means), srpi, least_robust_group, and
            interpretation.
        """
        if len(robustness_by_group) < 2:
            raise ValueError("Need at least 2 groups.")
        rg = {}
        records = []
        for grp, scores in robustness_by_group.items():
            s = np.asarray(scores, dtype=float)
            if s.size == 0:
                raise ValueError(f"Group {grp!r} has no robustness scores.")
            rg[str(grp)] = float(s.mean())
            records.extend({"group": str(grp), "value": float(v)} for v in s)
        mx = max(rg.values())
        srpi = float(min(rg.values()) / mx) if mx > 0 else 0.0
        least = min(rg, key=rg.get)

        def _srpi(sample) -> float:
            means: Dict[str, list] = {}
            for r in sample:
                means.setdefault(r["group"], []).append(r["value"])
            mvals = [float(np.mean(v)) for v in means.values()]
            m = max(mvals)
            return (min(mvals) / m) if m > 0 else 0.0

        out = {
            "robustness_by_group": rg,
            "srpi": srpi,
            "least_robust_group": least,
            "interpretation": (
                f"SRPI = {srpi:.3f} (1 = equal paraphrase robustness across groups); "
                f"lowest robustness in group '{least}'."
            ),
        }

        # Percentile bootstrap over pooled per-query robustness scores (tagged by
        # group) for the min/max robustness ratio.
        from equimed_dss.inference import MetricResult, bootstrap_ci

        if len(records) >= 2:
            ci = bootstrap_ci(records, _srpi, n_boot=1000, random_state=0)
            out["ci_lower"] = ci.ci_lower
            out["ci_upper"] = ci.ci_upper
            out["ci_method"] = ci.method
        return MetricResult(out, name="SRPI", value_key="srpi")
