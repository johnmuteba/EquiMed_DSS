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
    parts = re.split(r"[.!?]+", text or "")
    return [p for p in parts if p.strip()]


class LexicalDiversityDisparityIndex:
    """Lexical Diversity Disparity Index (LDDI) via Root Type-Token Ratio.

    RTTR(g) = |V(union R_g)| / sqrt( sum_i |R_i^g| );
    LDDI = max_g RTTR(g) - min_g RTTR(g); LDDI_norm = LDDI / RTTR_overall.
    """

    def __init__(self):
        pass

    def calculate_lddi(self, responses_by_group: Dict[str, Sequence[str]]) -> Dict[str, Any]:
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
        return {
            "rttr_by_group": rttr,
            "lddi": lddi,
            "lddi_norm": float(lddi / rttr_overall) if rttr_overall else 0.0,
            "interpretation": (
                f"LDDI = {lddi:.3f} (max-min Root Type-Token Ratio across groups); "
                "larger values mean response vocabulary richness varies more by group."
            ),
        }


class RecommendationEntropyGap:
    """Recommendation Entropy Gap (REG).

    H(T|g) = -sum_t P(t|g) log2 P(t|g); REG = max_{g,g'} |H(T|g) - H(T|g')|;
    REG_KL = max_g D_KL( P(t|g) || P(t) ).
    """

    def __init__(self):
        pass

    def calculate_reg(self, recommendations_by_group: Dict[str, Sequence[Any]]) -> Dict[str, Any]:
        if len(recommendations_by_group) < 2:
            raise ValueError("Need at least 2 groups.")
        labels = sorted({t for recs in recommendations_by_group.values() for t in recs})
        if not labels:
            raise ValueError("No recommendation labels found.")

        def dist(recs):
            n = len(recs)
            return np.array([sum(1 for x in recs if x == t) / n for t in labels]) if n else None

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
            kl = float(sum(pi * np.log2(pi / pti) for pi, pti in zip(p, pt) if pi > 0 and pti > 0))
            reg_kl = max(reg_kl, kl)

        return {
            "entropy_by_group": entropy_by_group,
            "reg": reg,
            "reg_kl": float(reg_kl),
            "interpretation": (
                f"REG = {reg:.3f} bits (max-min recommendation entropy across groups); "
                f"REG_KL = {reg_kl:.3f} bits. Larger values indicate more divergent "
                "recommendation distributions by group."
            ),
        }


class ClinicalInformationDensityRatio:
    """Clinical Information Density Ratio (CIDR).

    CID(r) = (|concepts(r)| / |tokens(r)|) * 100; CID(g) = mean CID over group;
    CIDR(g) = CID(g) / max_g' CID(g'); CIDR_min = min_g CIDR(g).
    Takes precomputed (n_concepts, n_tokens) per response (UMLS extraction is external).
    """

    def __init__(self):
        pass

    def calculate_cidr(self, concept_counts_by_group: Dict[str, Sequence[tuple]]) -> Dict[str, Any]:
        if len(concept_counts_by_group) < 2:
            raise ValueError("Need at least 2 groups.")
        cid = {}
        for grp, pairs in concept_counts_by_group.items():
            vals = [(nc / nt) * 100 for nc, nt in pairs if nt > 0]
            if not vals:
                raise ValueError(f"Group {grp!r} has no valid (concepts, tokens) pairs.")
            cid[str(grp)] = float(np.mean(vals))
        mx = max(cid.values())
        cidr = {k: float(v / mx) for k, v in cid.items()} if mx > 0 else {k: 0.0 for k in cid}
        cidr_min = float(min(cidr.values()))
        worst = min(cidr, key=cidr.get)
        return {
            "cid_by_group": cid,
            "cidr_by_group": cidr,
            "cidr_min": cidr_min,
            "interpretation": (
                f"CIDR_min = {cidr_min:.3f} (group '{worst}' has the lowest clinical "
                "concept density relative to the richest group; 1.0 = parity)."
            ),
        }


class DiagnosticCompletenessIndex:
    """Diagnostic Completeness Index (DCI).

    DCI(r) = |D(r) ∩ D*| / |D*|; DCI(g) = mean over group; dDCI = max_g - min_g.
    Optional severity-weighted wDCI with per-differential weights.
    """

    def __init__(self):
        pass

    def calculate_dci(
        self,
        reference_differentials: Sequence[str],
        mentioned_by_group: Dict[str, Sequence[Sequence[str]]],
        weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        Dstar = set(reference_differentials)
        if not Dstar:
            raise ValueError("reference_differentials must be non-empty.")
        if len(mentioned_by_group) < 2:
            raise ValueError("Need at least 2 groups.")
        dci = {}
        wdci = {}
        wtot = sum(weights.get(d, 0.0) for d in Dstar) if weights else None
        for grp, responses in mentioned_by_group.items():
            scores = [len(set(m) & Dstar) / len(Dstar) for m in responses]
            dci[str(grp)] = float(np.mean(scores)) if scores else 0.0
            if weights and wtot:
                wscores = [sum(weights.get(d, 0.0) for d in (set(m) & Dstar)) / wtot
                           for m in responses]
                wdci[str(grp)] = float(np.mean(wscores)) if wscores else 0.0
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
        return out


class UncertaintyQuantificationGap:
    """Uncertainty Quantification Gap (UQG).

    UD(r) = |hedging terms in r| / |sentences(r)|; UD(g) = mean; UQG = max_g - min_g.
    """

    DEFAULT_HEDGES = [
        "may", "might", "could", "possible", "possibly", "consider", "suspect",
        "likely", "unlikely", "uncertain", "cannot rule out", "rule out",
        "differential includes", "suggestive of", "concerning for",
    ]

    def __init__(self, hedging_terms: Optional[Sequence[str]] = None):
        self.hedges = [h.lower() for h in (hedging_terms or self.DEFAULT_HEDGES)]

    def _ud(self, text: str) -> float:
        sents = _sentences(text)
        if not sents:
            return 0.0
        tl = (text or "").lower()
        hits = sum(len(re.findall(r"\b" + re.escape(h) + r"\b", tl)) for h in self.hedges)
        return hits / len(sents)

    def calculate_uqg(self, responses_by_group: Dict[str, Sequence[str]]) -> Dict[str, Any]:
        if len(responses_by_group) < 2:
            raise ValueError("Need at least 2 groups.")
        ud = {}
        for grp, texts in responses_by_group.items():
            vals = [self._ud(r) for r in texts]
            ud[str(grp)] = float(np.mean(vals)) if vals else 0.0
        vals = list(ud.values())
        uqg = float(max(vals) - min(vals))
        return {
            "ud_by_group": ud,
            "uqg": uqg,
            "interpretation": (
                f"UQG = {uqg:.3f} (max-min hedging density across groups); large "
                "values mean the model expresses uncertainty unequally, a possible "
                "overconfidence-by-group risk."
            ),
        }
