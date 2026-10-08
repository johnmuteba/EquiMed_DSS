"""Text-based fairness metrics: LDDI, REG, CIDR, DCI, UQG.

These operate on LLM response text (or precomputed derived quantities) grouped by
demographic group. Formulas follow the technical supplement.
"""

import re
from typing import Any, Dict, Optional, Sequence

import numpy as np


def _tokens(text: str) -> list:
    return re.findall(r"[A-Za-z']+", (text or "").lower())


def _sentences(text: str) -> list:
    # Split on terminal punctuation followed by whitespace or the end, so that
    # decimals ("troponin 0.04 ng/mL") do not end a sentence.
    parts = re.split(r"[.!?]+(?=\s|$)", text or "")
    return [p for p in parts if p.strip()]


class LexicalDiversityDisparityIndex:
    """Lexical Diversity Disparity Index (LDDI) via Root Type-Token Ratio.

    RTTR(g) = |V(union R_g)| / sqrt( sum_i |R_i^g| );
    LDDI = max_g RTTR(g) - min_g RTTR(g); LDDI_norm = LDDI / RTTR_overall.
    """

    def __init__(self):
        pass

    def calculate_lddi(
        self, responses_by_group: Dict[str, Sequence[str]]
    ) -> Dict[str, Any]:
        """Compute LDDI from response texts grouped by demographic group.

        Args:
            responses_by_group: mapping group -> list of response texts (at least
                two groups). Tokens are runs of ASCII letters and apostrophes.

        Returns:
            MetricResult with rttr_by_group, lddi, lddi_norm and a 95%
            percentile-bootstrap CI. RTTR pools each group's tokens, so it still
            depends on the amount of text: compare groups with similar numbers
            and lengths of responses.
        """
        if len(responses_by_group) < 2:
            raise ValueError("Need at least 2 groups.")
        rttr = {}
        all_tokens = []
        for grp, texts in responses_by_group.items():
            toks = [t for r in texts for t in _tokens(r)]
            if not toks:
                raise ValueError(f"Group {grp!r} has no tokens.")
            rttr[str(grp)] = float(len(set(toks)) / np.sqrt(len(toks)))
            all_tokens.extend(toks)
        rttr_overall = float(len(set(all_tokens)) / np.sqrt(len(all_tokens)))
        vals = list(rttr.values())
        lddi = float(max(vals) - min(vals))
        out = {
            "rttr_by_group": rttr,
            "lddi": lddi,
            "lddi_norm": float(lddi / rttr_overall) if rttr_overall else 0.0,
            "interpretation": (
                f"LDDI = {lddi:.3f} (max-min Root Type-Token Ratio across groups); "
                "larger values mean response vocabulary richness varies more by group."
            ),
        }

        # Bootstrap that resamples responses within each group (group sizes
        # fixed; up to 1.9.5 responses were pooled across groups).
        from equimed_dss.inference import MetricResult, stratified_bootstrap_ci

        strata = {
            str(grp): [_tokens(r) for r in texts]
            for grp, texts in responses_by_group.items()
        }

        def _lddi(smp):
            rv = []
            for resp in smp.values():
                toks = [t for r in resp for t in r]
                if not toks:
                    return float("nan")  # a resample without tokens: dropped
                rv.append(len(set(toks)) / np.sqrt(len(toks)))
            return float(max(rv) - min(rv))

        ci = stratified_bootstrap_ci(strata, _lddi, n_boot=1000, random_state=0)
        out["ci_lower"] = ci.ci_lower
        out["ci_upper"] = ci.ci_upper
        out["ci_method"] = ci.method
        return MetricResult(out, name="LDDI", value_key="lddi")


class RecommendationEntropyGap:
    """Recommendation Entropy Gap (REG).

    H(T|g) = -sum_t P(t|g) log2 P(t|g); REG = max_{g,g'} |H(T|g) - H(T|g')|;
    REG_KL = max_g D_KL( P(t|g) || P(t) ).
    """

    def __init__(self):
        pass

    def calculate_reg(
        self, recommendations_by_group: Dict[str, Sequence[Any]]
    ) -> Dict[str, Any]:
        """Compute REG from recommendation labels grouped by demographic group.

        Args:
            recommendations_by_group: mapping group -> list of recommendation
                labels (at least two groups).

        Returns:
            MetricResult with entropy_by_group (bits), reg, reg_kl and a 95%
            percentile-bootstrap CI. Plug-in entropy is biased downward in small
            samples, so groups with few recommendations look less diverse.
        """
        if len(recommendations_by_group) < 2:
            raise ValueError("Need at least 2 groups.")
        labels = sorted({t for recs in recommendations_by_group.values() for t in recs})
        if not labels:
            raise ValueError("No recommendation labels found.")

        def dist(recs):
            n = len(recs)
            return (
                np.array([sum(1 for x in recs if x == t) / n for t in labels])
                if n
                else None
            )

        entropy_by_group = {}
        dists = {}
        for grp, recs in recommendations_by_group.items():
            p = dist(recs)
            if p is None:
                raise ValueError(f"Group {grp!r} has no recommendations.")
            dists[str(grp)] = p
            nz = p[p > 0]
            entropy_by_group[str(grp)] = float(-(nz * np.log2(nz)).sum())

        ev = list(entropy_by_group.values())
        reg = float(max(ev) - min(ev))

        # marginal P(t) and REG_KL
        allrecs = [x for recs in recommendations_by_group.values() for x in recs]
        pt = dist(allrecs)
        reg_kl = 0.0
        for grp, p in dists.items():
            kl = float(
                sum(
                    pi * np.log2(pi / pti)
                    for pi, pti in zip(p, pt)
                    if pi > 0 and pti > 0
                )
            )
            reg_kl = max(reg_kl, kl)

        out = {
            "entropy_by_group": entropy_by_group,
            "reg": reg,
            "reg_kl": float(reg_kl),
            "interpretation": (
                f"REG = {reg:.3f} bits (max-min recommendation entropy across groups); "
                f"REG_KL = {reg_kl:.3f} bits. Larger values indicate more divergent "
                "recommendation distributions by group."
            ),
        }

        # Bootstrap that resamples recommendations within each group.
        from equimed_dss.inference import MetricResult, stratified_bootstrap_ci

        def _entropy(recs):
            n = len(recs)
            p = np.array([recs.count(t) / n for t in set(recs)])
            nz = p[p > 0]
            return float(-(nz * np.log2(nz)).sum())

        strata = {
            str(grp): list(recs) for grp, recs in recommendations_by_group.items()
        }
        ci = stratified_bootstrap_ci(
            strata,
            lambda smp: float(
                max(_entropy(v) for v in smp.values())
                - min(_entropy(v) for v in smp.values())
            ),
            n_boot=1000,
            random_state=0,
        )
        out["ci_lower"] = ci.ci_lower
        out["ci_upper"] = ci.ci_upper
        out["ci_method"] = ci.method
        return MetricResult(out, name="REG", value_key="reg")


class ClinicalInformationDensityRatio:
    """Clinical Information Density Ratio (CIDR).

    CID(r) = (|concepts(r)| / |tokens(r)|) * 100; CID(g) = mean CID over group;
    CIDR(g) = CID(g) / max_g' CID(g'); CIDR_min = min_g CIDR(g).
    Takes precomputed (n_concepts, n_tokens) per response (UMLS extraction is external).
    """

    def __init__(self):
        pass

    def calculate_cidr(
        self, concept_counts_by_group: Dict[str, Sequence[tuple]]
    ) -> Dict[str, Any]:
        """Compute CIDR from (n_concepts, n_tokens) pairs grouped by group.

        Args:
            concept_counts_by_group: mapping group -> list of (n_concepts,
                n_tokens) pairs, one per response (at least two groups).
                Responses with zero tokens are skipped.

        Returns:
            MetricResult with cid_by_group, cidr_by_group, cidr_min and a 95%
            percentile-bootstrap CI.
        """
        if len(concept_counts_by_group) < 2:
            raise ValueError("Need at least 2 groups.")
        cid = {}
        for grp, pairs in concept_counts_by_group.items():
            vals = [(nc / nt) * 100 for nc, nt in pairs if nt > 0]
            if not vals:
                raise ValueError(
                    f"Group {grp!r} has no valid (concepts, tokens) pairs."
                )
            cid[str(grp)] = float(np.mean(vals))
        mx = max(cid.values())
        cidr = (
            {k: float(v / mx) for k, v in cid.items()}
            if mx > 0
            else {k: 0.0 for k in cid}
        )
        cidr_min = float(min(cidr.values()))
        worst = min(cidr, key=cidr.get)
        out = {
            "cid_by_group": cid,
            "cidr_by_group": cidr,
            "cidr_min": cidr_min,
            "interpretation": (
                f"CIDR_min = {cidr_min:.3f} (group '{worst}' has the lowest clinical "
                "concept density relative to the richest group; 1.0 = parity)."
            ),
        }

        # Bootstrap that resamples responses (with tokens) within each group.
        from equimed_dss.inference import MetricResult, stratified_bootstrap_ci

        strata = {
            str(grp): np.array([(nc / nt) * 100 for nc, nt in pairs if nt > 0])
            for grp, pairs in concept_counts_by_group.items()
        }

        def _cidr_min(smp):
            cids = [float(v.mean()) for v in smp.values()]
            m = max(cids)
            return float(min(cids) / m) if m > 0 else float("nan")

        ci = stratified_bootstrap_ci(strata, _cidr_min, n_boot=1000, random_state=0)
        out["ci_lower"] = ci.ci_lower
        out["ci_upper"] = ci.ci_upper
        out["ci_method"] = ci.method
        return MetricResult(out, name="CIDR", value_key="cidr_min")


class DiagnosticCompletenessIndex:
    """Diagnostic Completeness Index (DCI).

    DCI(r) = |D(r) ∩ D*(r)| / |D*(r)|; DCI(g) = mean over group; dDCI = max_g - min_g.
    D*(r) is the reference differential list for the case behind response r:
    one shared list, or a case-specific list per response. Optional
    severity-weighted wDCI with per-differential weights.
    """

    def __init__(self):
        pass

    def calculate_dci(
        self,
        reference_differentials: Optional[Sequence[str]],
        mentioned_by_group: Dict[str, Sequence[Sequence[str]]],
        weights: Optional[Dict[str, float]] = None,
        references_by_group: Optional[Dict[str, Sequence[Sequence[str]]]] = None,
    ) -> Dict[str, Any]:
        """Compute DCI coverage of a reference differential list by group.

        Args:
            reference_differentials: the guideline differential list D* shared by
                all responses; pass None when ``references_by_group`` is given.
            mentioned_by_group: mapping group -> list of responses, each a list
                of the differentials the response mentions (at least two groups,
                none empty).
            weights: optional differential -> severity weight for wDCI.
            references_by_group: optional case-specific references, with the
                same structure as ``mentioned_by_group`` (one reference list per
                response). Use it when the cases differ, so that each response is
                scored against the differentials appropriate to its own case.

        Returns:
            MetricResult with dci_by_group, delta_dci (max - min), optional
            wdci_by_group, and a 95% bootstrap CI for delta_dci that resamples
            responses within each group.

        Raises:
            ValueError: for fewer than 2 groups, an empty group, an empty
                reference list, or references that do not match the responses.
        """
        if len(mentioned_by_group) < 2:
            raise ValueError("Need at least 2 groups.")
        if references_by_group is None:
            if not reference_differentials:
                raise ValueError("reference_differentials must be non-empty.")
            shared = set(reference_differentials)
            references_by_group = {
                grp: [shared] * len(resp) for grp, resp in mentioned_by_group.items()
            }
        elif set(references_by_group) != set(mentioned_by_group):
            raise ValueError("references_by_group must have the same groups.")

        dci, wdci, strata = {}, {}, {}
        for grp, responses in mentioned_by_group.items():
            refs = references_by_group[grp]
            if len(responses) == 0:
                raise ValueError(f"Group {grp!r} has no responses.")
            if len(refs) != len(responses):
                raise ValueError(
                    f"Group {grp!r}: one reference list is needed per response."
                )
            scores, wscores = [], []
            for m, ref in zip(responses, refs):
                ref = set(ref)
                if not ref:
                    raise ValueError(f"Group {grp!r} has an empty reference list.")
                hit = set(m) & ref
                scores.append(len(hit) / len(ref))
                if weights:
                    wtot = sum(weights.get(d, 0.0) for d in ref)
                    if wtot > 0:
                        wscores.append(sum(weights.get(d, 0.0) for d in hit) / wtot)
            dci[str(grp)] = float(np.mean(scores))
            strata[str(grp)] = np.array(scores, dtype=float)
            if weights and wscores:
                wdci[str(grp)] = float(np.mean(wscores))
        vals = list(dci.values())
        ddci = float(max(vals) - min(vals))
        out = {
            "dci_by_group": dci,
            "delta_dci": ddci,
            "interpretation": (
                f"dDCI = {ddci:.3f} (max-min guideline-differential coverage across "
                "groups); larger means more unequal diagnostic thoroughness."
            ),
        }
        if weights:
            out["wdci_by_group"] = wdci

        # Bootstrap that resamples responses within each group.
        from equimed_dss.inference import MetricResult, stratified_bootstrap_ci

        ci = stratified_bootstrap_ci(
            strata,
            lambda smp: float(
                max(v.mean() for v in smp.values())
                - min(v.mean() for v in smp.values())
            ),
            n_boot=1000,
            random_state=0,
        )
        out["ci_lower"] = ci.ci_lower
        out["ci_upper"] = ci.ci_upper
        out["ci_method"] = ci.method
        return MetricResult(out, name="DCI", value_key="delta_dci")


class UncertaintyQuantificationGap:
    """Uncertainty Quantification Gap (UQG), a hedging-language density gap.

    UD(r) = |hedging terms in r| / |sentences(r)|; UD(g) = mean; UQG = max_g - min_g.

    UQG counts hedging words and phrases ("may", "likely", "rule out", ...). It
    measures how often hedging language is used, not whether a model's stated
    uncertainty is calibrated, and the default lexicon has not been validated
    against annotated clinical language; the name is kept for compatibility.
    Interpret differences as differences in wording.
    """

    DEFAULT_HEDGES = [
        "may",
        "might",
        "could",
        "possible",
        "possibly",
        "consider",
        "suspect",
        "likely",
        "unlikely",
        "uncertain",
        "cannot rule out",
        "rule out",
        "differential includes",
        "suggestive of",
        "concerning for",
    ]

    def __init__(self, hedging_terms: Optional[Sequence[str]] = None):
        self.hedges = [h.lower() for h in (hedging_terms or self.DEFAULT_HEDGES)]
        # One alternation, longest term first, matched left to right without
        # overlap: "cannot rule out" counts once, not also as "rule out" (up to
        # 1.9.5 each term was counted separately, so overlapping terms counted
        # twice).
        terms = sorted(set(self.hedges), key=len, reverse=True)
        self._pattern = re.compile(
            r"\b(?:" + "|".join(re.escape(h) for h in terms) + r")\b"
        )

    def _ud(self, text: str) -> float:
        sents = _sentences(text)
        if not sents:
            return 0.0
        tl = (text or "").lower()
        hits = len(self._pattern.findall(tl))
        return hits / len(sents)

    def calculate_uqg(
        self, responses_by_group: Dict[str, Sequence[str]]
    ) -> Dict[str, Any]:
        """Compute UQG, the max - min hedging density across groups.

        Args:
            responses_by_group: mapping group -> list of response texts (at least
                two groups). Hedging density is the number of hedging terms per
                sentence; overlapping terms are counted once.

        Returns:
            MetricResult with ud_by_group, uqg and a 95% percentile-bootstrap CI.
        """
        if len(responses_by_group) < 2:
            raise ValueError("Need at least 2 groups.")
        ud, strata = {}, {}
        for grp, texts in responses_by_group.items():
            vals = [self._ud(r) for r in texts]
            if not vals:
                raise ValueError(f"Group {grp!r} has no responses.")
            ud[str(grp)] = float(np.mean(vals))
            strata[str(grp)] = np.array(vals, dtype=float)
        vals = list(ud.values())
        uqg = float(max(vals) - min(vals))
        out = {
            "ud_by_group": ud,
            "uqg": uqg,
            "interpretation": (
                f"UQG = {uqg:.3f} (max-min hedging-term density across groups); "
                "a lexical count of hedging words, not a measure of calibrated "
                "uncertainty."
            ),
        }

        # Bootstrap that resamples responses within each group.
        from equimed_dss.inference import MetricResult, stratified_bootstrap_ci

        ci = stratified_bootstrap_ci(
            strata,
            lambda smp: float(
                max(v.mean() for v in smp.values())
                - min(v.mean() for v in smp.values())
            ),
            n_boot=1000,
            random_state=0,
        )
        out["ci_lower"] = ci.ci_lower
        out["ci_upper"] = ci.ci_upper
        out["ci_method"] = ci.method
        return MetricResult(out, name="UQG", value_key="uqg")
