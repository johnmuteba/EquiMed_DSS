"""Task A contract test: every EquiMed-DSS metric, when called, prints its value
alongside a 95% CI (or the explicit "unavailable" string for aggregate-only
inputs), and remains a plain dict for key access / JSON.

This guards the v1.9.0 guarantee that no metric returns a bare dict without a
printable confidence interval.
"""
import json

import numpy as np
import pytest

from equimed_dss.domain1 import (
    DecisionFlipRate,
    EmbeddingConsistencyScore,
    InterRaterReliability,
)
from equimed_dss.domain2 import (
    EthicalRiskIndex,
    HarmAdjustedFairnessGap,
    HierarchicalEquityRatio,
    IntersectionalBiasScore,
)
from equimed_dss.domain3 import (
    AuditTraceabilityScore,
    GovernanceComplianceIndex,
    TemporalFairnessDrift,
)
from equimed_dss.domain4 import (
    ClinicalHallucinationRate,
    GeographicRepresentationIndex,
    InstructionalVulnerabilityIndex,
    SemanticParityGap,
)
from equimed_dss.domain5 import (
    ClinicalInformationDensityRatio,
    CounterfactualParityScore,
    DiagnosticCompletenessIndex,
    GeographicRepresentationBiasIndex,
    HealthcareSystemStratifiedFairness,
    IntersectionalCalibrationError,
    IntersectionalShapleyFairnessValue,
    LexicalDiversityDisparityIndex,
    RecommendationEntropyGap,
    SemanticRobustnessParityIndex,
    UncertaintyQuantificationGap,
    WeightedClinicalHarmAdjustedFairnessGap,
)
from equimed_dss.geographic import BurdenEvidenceMismatch, GeographicConcentration
from equimed_dss.appendix import (
    BiasConcentrationIndex,
    BootstrapConfidenceIntervals,
    JensenShannonDivergence,
    MutualInformationContent,
    NetworkModularity,
    RobustnessCertificationScore,
    StatisticalPowerAnalysis,
    TransparencyScore,
    WassersteinDistance,
)


def _rng():
    return np.random.RandomState(0)


# Each entry: (label, callable returning the metric result).
def _all_metric_results():
    rng = _rng()
    n = 60
    groups2 = ["m"] * (n // 2) + ["p"] * (n // 2)
    conf = rng.uniform(0, 1, n).tolist()
    correct = rng.randint(0, 2, n).tolist()
    yield "DFR", DecisionFlipRate().calculate_dfr([0, 1, 0, 1, 0], [0, 0, 0, 1, 1])
    yield "ECS", EmbeddingConsistencyScore().calculate_ecs(rng.rand(8, 5), rng.rand(8, 5))
    yield "ICC", InterRaterReliability().calculate_icc_2_1(rng.rand(15, 3))
    yield "HER", HierarchicalEquityRatio().calculate_her(
        {"White": 0.85, "Black": 0.75}, group_observations={
            "White": rng.rand(10).tolist(), "Black": rng.rand(10).tolist()}
    )
    yield "Bias-Gini", HierarchicalEquityRatio().calculate_bias_gini([0.85, 0.75, 0.80, 0.82])
    yield "HAFG", HarmAdjustedFairnessGap().calculate_hafg(
        {"fn": 5, "fp": 10}, {"fn": 2, "fp": 5},
        group1_cases=["fn", "fp", "tn", "tn", "fp"], group2_cases=["fn", "tn", "tn", "fp", "tn"],
    )
    yield "ERI", EthicalRiskIndex().calculate_eri([{"severity": 2.5}, {"severity": 5.0}], 100)
    yield "IBS", IntersectionalBiasScore().calculate_subgroup_similarity(
        {"A": rng.rand(6), "B": rng.rand(6), "C": rng.rand(6)}
    )
    yield "TFD", TemporalFairnessDrift().calculate_drift([0.1, 0.12, 0.11, 0.5, 0.13])
    yield "ATS", AuditTraceabilityScore().calculate_ats(95, 100)
    yield "GCI", GovernanceComplianceIndex().calculate_gci({"p1": True, "p2": True, "p3": False})
    yield "SPG", SemanticParityGap().calculate_spg(rng.rand(8, 5), rng.rand(10, 5))
    yield "CHR", ClinicalHallucinationRate().calculate_chr([0.2, 0.4, 0.8, 0.9])
    yield "IVI", InstructionalVulnerabilityIndex().calculate_ivi([0, 0, 0, 1], [1, 1, 0, 1])
    yield "GRI", GeographicRepresentationIndex().calculate_gri(
        ["US", "GB", "BR", "CN", "TZ", "US"], ["US", "GB", "DE"]
    )
    yield "ICE", IntersectionalCalibrationError().calculate_ice(groups2, conf, correct, n_bins=5)
    yield "wHAFG", WeightedClinicalHarmAdjustedFairnessGap().calculate_whafg(
        groups2, rng.rand(n).tolist(), rng.randint(0, 2, n).tolist()
    )
    yield "CPS", CounterfactualParityScore().calculate_cps([1.0, 0.8, 0.9, 0.85])
    yield "SRPI", SemanticRobustnessParityIndex().calculate_srpi(
        {"A": [0.9, 0.9, 0.8], "B": [0.6, 0.6, 0.7]}
    )
    yield "LDDI", LexicalDiversityDisparityIndex().calculate_lddi(
        {"A": ["pain pain", "ache ache"], "B": ["chest pain radiating arm", "dyspnea nausea"]}
    )
    yield "REG", RecommendationEntropyGap().calculate_reg(
        {"A": ["acs", "acs", "acs", "acs"], "B": ["acs", "acs", "nc", "nc"]}
    )
    yield "CIDR", ClinicalInformationDensityRatio().calculate_cidr(
        {"A": [(5, 100), (5, 100)], "B": [(10, 100), (8, 100)]}
    )
    yield "DCI", DiagnosticCompletenessIndex().calculate_dci(
        ["ACS", "PE", "Dissection", "GERD"],
        {"A": [["ACS", "PE", "Dissection"], ["ACS", "GERD"]], "B": [["ACS"], ["PE"]]},
    )
    yield "UQG", UncertaintyQuantificationGap().calculate_uqg(
        {"A": ["This is ACS. Treat now.", "Clear MI."],
         "B": ["This may be ACS. Consider PE.", "Possibly unstable."]}
    )
    burden = {"AFRO": 0.25, "AMRO": 0.25, "EURO": 0.25, "SEARO": 0.25}
    yield "GRBI", GeographicRepresentationBiasIndex().calculate_grbi(
        {"AMRO": 3, "EURO": 1}, burden, corpus_records=["AMRO", "AMRO", "AMRO", "EURO"]
    )
    yield "HSSF", HealthcareSystemStratifiedFairness().calculate_hssf(
        ["US"] * 30 + ["CA"] * 30, groups2, rng.rand(n).tolist()
    )
    race = rng.choice(["W", "B"], 120)
    gender = rng.choice(["F", "M"], 120)
    yield "ISFV", IntersectionalShapleyFairnessValue(min_cell=5).calculate_isfv(
        {"race": race, "gender": gender}, (race == "B").astype(float)
    )
    yield "BEMI", BurdenEvidenceMismatch().calculate_bemi(
        {"EURO": 2, "AMRO": 2}, burden, evidence_records=["EURO", "EURO", "AMRO", "AMRO", "EURO"]
    )
    yield "GCC", GeographicConcentration().calculate_gcc(
        {"A": 1.0, "B": 3.0, "C": 2.0}, region_records=["A", "B", "B", "B", "C", "C"]
    )
    yield "BCI", BootstrapConfidenceIntervals(n_bootstrap=100, random_state=42).calculate_bci(
        np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    )
    yield "SampleSize", StatisticalPowerAnalysis().calculate_sample_size(effect_size=0.5)
    yield "BiasConcentration", BiasConcentrationIndex().calculate_bci([0.25, 0.25, 0.25, 0.25])
    yield "MIC", MutualInformationContent().calculate_mic(
        np.array([0, 0, 1, 1, 2, 2, 0, 1, 2, 0]), np.array([0, 0, 1, 1, 1, 1, 0, 1, 1, 0])
    )
    yield "WD", WassersteinDistance().calculate_wd(
        np.array([1.0, 2.0, 3.0, 4.0]), np.array([5.0, 6.0, 7.0, 8.0])
    )
    yield "NM", NetworkModularity().calculate_modularity(
        np.array([[0, 1, 1, 0, 0], [1, 0, 1, 0, 0], [1, 1, 0, 0, 0],
                  [0, 0, 0, 0, 1], [0, 0, 0, 1, 0]])
    )
    yield "TS", TransparencyScore().calculate_ts(
        [{"explanation_quality": 0.8, "feature_importance": 0.7, "interpretability": 0.9},
         {"explanation_quality": 0.7, "feature_importance": 0.8, "interpretability": 0.8}]
    )
    yield "RCS", RobustnessCertificationScore().calculate_rcs(
        np.array([0, 1, 1, 0, 1]), [np.array([0, 1, 1, 0, 1]), np.array([0, 1, 0, 0, 1])]
    )


# Metrics whose inputs are aggregate-only here and therefore legitimately print
# "CI unavailable" rather than a numeric interval.
_UNAVAILABLE_OK = {"JSD", "SampleSize"}


def test_every_metric_prints_a_confidence_interval():
    results = list(_all_metric_results())
    # The suite advertises 37 metrics; this fixture exercises the full set.
    assert len(results) >= 37, f"only {len(results)} metrics exercised"
    for label, res in results:
        s = str(res)
        assert "95% CI" in s, f"{label} does not print a 95% CI: {s}"
        if label not in _UNAVAILABLE_OK:
            assert "[" in s and ";" in s, f"{label} prints no numeric CI: {s}"


def test_metric_results_are_still_dicts():
    for label, res in _all_metric_results():
        assert isinstance(res, dict), f"{label} is not dict-compatible"
        # JSON-serialisable after dropping non-JSON helper values (e.g. DataFrames).
        json.dumps({k: v for k, v in res.items()
                    if isinstance(v, (int, float, str, list, dict, type(None)))})
