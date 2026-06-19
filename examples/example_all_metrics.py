"""Worked, runnable examples for every EquiMed-DSS metric (used to validate the
vignette). Each block prints its result. Run: python examples/example_all_metrics.py
"""
import numpy as np
import pandas as pd

rng = np.random.RandomState(42)


def show(name, fn):
    try:
        print(f"\n### {name}")
        fn()
    except Exception as e:  # surface signature problems during validation
        print(f"  ERROR: {type(e).__name__}: {e}")


# ---------------------------------------------------------------- Domain 1
def d1():
    from equimed_dss.domain1 import (DecisionFlipRate, EmbeddingConsistencyScore,
                                      InterRaterReliability)
    orig = [1, 0, 1, 1, 0, 1, 0, 0]
    cf = [1, 0, 1, 0, 0, 1, 1, 0]  # 2 of 8 flip under a counterfactual swap
    print("DFR:", DecisionFlipRate().calculate_dfr(orig, cf))

    o = rng.normal(size=(20, 16)); p = o + rng.normal(scale=0.05, size=(20, 16))
    print("ECS:", EmbeddingConsistencyScore().calculate_ecs(o, p))

    judges = np.array([[4, 4, 5], [3, 3, 4], [5, 5, 5], [2, 3, 2], [4, 5, 4]])
    print("ICC(2,1):", InterRaterReliability().calculate_icc_2_1(judges))


# ---------------------------------------------------------------- Domain 2
def d2():
    from equimed_dss.domain2 import (HierarchicalEquityRatio, HarmAdjustedFairnessGap,
                                     EthicalRiskIndex, IntersectionalBiasScore)
    scores = {"White": 0.85, "Black": 0.78, "Hispanic": 0.80, "Asian": 0.87}
    her = HierarchicalEquityRatio()
    # HER carries per-group scores plus the scalar her_gap; printing shows the gap
    # with its CI (pass group_observations for a numeric CI rather than "unavailable").
    print("HER:", her.calculate_her(scores))
    print("Bias-Gini:", her.calculate_bias_gini(list(scores.values())))

    print("HAFG:", HarmAdjustedFairnessGap().calculate_hafg(
        {"fn": 5, "fp": 10}, {"fn": 2, "fp": 5},
        group1_cases=["fn"] * 5 + ["fp"] * 10 + ["tn"] * 85,
        group2_cases=["fn"] * 2 + ["fp"] * 5 + ["tn"] * 93))

    print("ERI:", EthicalRiskIndex().calculate_eri(
        [{"severity": 0.8}, {"severity": 0.3}, {"severity": 0.9}], n_total_outputs=100))

    print("IBS:", IntersectionalBiasScore().calculate_subgroup_similarity(
        {"White_M": np.array([0.85, 0.9]), "Black_F": np.array([0.7, 0.6]),
         "Asian_M": np.array([0.88, 0.85])}))


# ---------------------------------------------------------------- Domain 3
def d3():
    from equimed_dss.domain3 import (AuditTraceabilityScore, GovernanceComplianceIndex,
                                     TemporalFairnessDrift)
    print("ATS:", AuditTraceabilityScore().calculate_ats(n_traceable=92, n_total=100))
    print("GCI:", GovernanceComplianceIndex().calculate_gci(
        {"audit_logging": True, "bias_testing": True, "human_oversight": False}))
    print("TFD:", TemporalFairnessDrift().calculate_drift([0.80, 0.82, 0.79, 0.85, 0.91, 0.95]))


# ---------------------------------------------------------------- Domain 4
def d4():
    from equimed_dss.domain4 import (SemanticParityGap, ClinicalHallucinationRate,
        InstructionalVulnerabilityIndex, GeographicRepresentationIndex)
    ep = rng.normal(size=(10, 8)); em = rng.normal(loc=0.3, size=(10, 8))
    print("SPG:", SemanticParityGap().calculate_spg(ep, em))
    print("CHR:", ClinicalHallucinationRate().calculate_chr([0.2, 0.4, 0.8, 0.9], tau=0.5))
    print("IVI:", InstructionalVulnerabilityIndex().calculate_ivi(
        ["acs", "non_cardiac", "acs"], ["acs", "acs", "acs"]))
    print("GRI:", GeographicRepresentationIndex().calculate_gri(
        ["US", "GB", "BR", "CN", "TZ"], ["US", "GB", "DE"]))


# ---------------------------------------------------------------- Domain 5
def d5():
    from equimed_dss.domain5 import (IntersectionalCalibrationError,
        WeightedClinicalHarmAdjustedFairnessGap, LexicalDiversityDisparityIndex,
        RecommendationEntropyGap, CounterfactualParityScore,
        ClinicalInformationDensityRatio, DiagnosticCompletenessIndex,
        UncertaintyQuantificationGap, GeographicRepresentationBiasIndex,
        HealthcareSystemStratifiedFairness, IntersectionalShapleyFairnessValue,
        SemanticRobustnessParityIndex)
    g = ["A"] * 4 + ["B"] * 4
    print("ICE:", IntersectionalCalibrationError().calculate_ice(
        g, [0.9] * 8, [1, 1, 1, 0, 0, 0, 0, 0], n_bins=5))
    print("wHAFG:", WeightedClinicalHarmAdjustedFairnessGap().calculate_whafg(
        ["m", "m", "p", "p"], [1, 1, 1, 1], [1, 1, 0, 0]))
    print("LDDI:", LexicalDiversityDisparityIndex().calculate_lddi(
        {"A": ["pain pain pain"], "B": ["chest pain radiating to the left arm"]}))
    print("REG:", RecommendationEntropyGap().calculate_reg(
        {"A": ["acs", "acs"], "B": ["acs", "non_cardiac"]}))
    print("CPS:", CounterfactualParityScore().calculate_cps([1.0, 0.8, 0.9]))
    print("CIDR:", ClinicalInformationDensityRatio().calculate_cidr(
        {"A": [(5, 100), (4, 100)], "B": [(10, 100), (9, 100)]}))
    print("DCI:", DiagnosticCompletenessIndex().calculate_dci(
        ["ACS", "PE", "GERD"], {"A": [["ACS", "PE", "GERD"], ["ACS", "PE"]], "B": [["ACS"], ["PE"]]}))
    print("UQG:", UncertaintyQuantificationGap().calculate_uqg(
        {"A": ["This is ACS.", "Clear MI."], "B": ["This may be ACS. Consider PE.", "Possibly unstable."]}))
    print("GRBI:", GeographicRepresentationBiasIndex().calculate_grbi(
        {"AMRO": 100, "EURO": 10, "AFRO": 1}, {"AMRO": 0.2, "EURO": 0.4, "AFRO": 0.4},
        corpus_records=["AMRO"] * 100 + ["EURO"] * 10 + ["AFRO"]))
    print("HSSF:", HealthcareSystemStratifiedFairness().calculate_hssf(
        ["US", "US", "UK", "UK"], ["m", "p", "m", "p"], [0.0, 0.4, 0.1, 0.2]))
    race = rng.choice(["W", "B"], 200); gender = rng.choice(["F", "M"], 200)
    print("ISFV:", IntersectionalShapleyFairnessValue(min_cell=5).calculate_isfv(
        {"race": race, "gender": gender}, (race == "B").astype(float)))
    print("SRPI:", SemanticRobustnessParityIndex().calculate_srpi(
        {"A": [0.9, 0.9], "B": [0.6, 0.6]}))


# ---------------------------------------------------------------- Statistics
def stats():
    from equimed_dss.statistics import (HierarchicalLinearModeling, MediationAnalysis,
                                        NetworkStatistics, ReliabilityAnalysis)
    rows = []
    for grp in range(8):
        ge = rng.normal(0, 2)
        for _ in range(25):
            x = rng.normal(0, 1)
            rows.append({"group": grp, "x": x, "y": ge + 0.5 * x + rng.normal(0, 1)})
    df = pd.DataFrame(rows)
    hlm = HierarchicalLinearModeling().fit_model(
        df, outcome_var="y", level1_predictors=["x"], level2_var="group")
    print("HLM ICC:", round(hlm["icc"], 3), "| AIC:", round(hlm["aic"], 1))

    n = 300; X = rng.normal(0, 1, n); M = 0.5 * X + rng.normal(0, 1, n)
    Y = 0.3 * X + 0.4 * M + rng.normal(0, 1, n)
    med = MediationAnalysis().analyze_mediation(
        pd.DataFrame({"X": X, "M": M, "Y": Y}),
        treatment_var="X", mediator_var="M", outcome_var="Y")
    print("Mediation prop_mediated:", round(med["proportion_mediated"], 3))

    adj = np.array([[0, .6, .2, 0], [.6, 0, 0, .5], [.2, 0, 0, 0], [0, .5, 0, 0]])
    net = NetworkStatistics().analyze_network(adj, node_labels=["DFR", "ECS", "HER", "IBS"])
    print("Network density:", round(net["density"], 3))

    ratings = np.array([[4, 4, 5], [3, 3, 4], [5, 5, 5], [2, 3, 2], [4, 5, 4]])
    rel = ReliabilityAnalysis().cronbachs_alpha(ratings)
    print("Cronbach alpha:", round(rel["alpha"], 3))


# ---------------------------------------------------------------- Appendix
def appendix():
    from equimed_dss.appendix import (BiasConcentrationIndex, BootstrapConfidenceIntervals,
        JensenShannonDivergence, MutualInformationContent, WassersteinDistance,
        NetworkModularity, RobustnessCertificationScore, TransparencyScore,
        StatisticalPowerAnalysis)
    print("BiasConcentration:", BiasConcentrationIndex().calculate_bci([0.1, 0.4, 0.3, 0.2]))
    print("BCI:", BootstrapConfidenceIntervals(n_bootstrap=500, random_state=42).calculate_bci(
        rng.normal(0.7, 0.1, 100)))
    p = np.array([0.9, 0.85, 0.78, 0.92]); q = np.array([0.75, 0.70, 0.68, 0.72])
    print("JSD:", JensenShannonDivergence().calculate_jsd(p, q))
    print("WD:", WassersteinDistance().calculate_wd(p, q))
    print("MIC:", MutualInformationContent().calculate_mic(
        rng.randint(0, 2, 200), rng.randint(0, 2, 200)))
    adj = np.array([[0, .8, .1, 0], [.8, 0, 0, .7], [.1, 0, 0, .6], [0, .7, .6, 0]])
    print("NM:", NetworkModularity().calculate_modularity(adj))
    print("RCS:", RobustnessCertificationScore().calculate_rcs(
        rng.normal(0.8, 0.05, 50), [rng.normal(0.8, 0.05, 50) for _ in range(5)], epsilon=0.1))
    print("TS:", TransparencyScore().calculate_ts([
        {"explanation_quality": 0.9, "feature_importance": 0.8, "interpretability": 0.85},
        {"explanation_quality": 0.7, "feature_importance": 0.6, "interpretability": 0.65}]))
    print("SampleSize:", StatisticalPowerAnalysis().calculate_sample_size(
        effect_size=0.2, alpha=0.05, power=0.8))


for name, fn in [("Domain 1", d1), ("Domain 2", d2), ("Domain 3", d3),
                 ("Domain 4", d4), ("Domain 5", d5), ("Statistics", stats),
                 ("Appendix", appendix)]:
    show(name, fn)
