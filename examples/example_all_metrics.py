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
    print("DFR:", DecisionFlipRate().calculate_dfr(orig, cf)["flip_rate"])

    o = rng.normal(size=(20, 16)); p = o + rng.normal(scale=0.05, size=(20, 16))
    print("ECS:", round(EmbeddingConsistencyScore().calculate_ecs(o, p)["mean_ecs"], 4))

    judges = np.array([[4, 4, 5], [3, 3, 4], [5, 5, 5], [2, 3, 2], [4, 5, 4]])
    print("ICC(2,1):", round(InterRaterReliability().calculate_icc_2_1(judges)["score"], 3))


# ---------------------------------------------------------------- Domain 2
def d2():
    from equimed_dss.domain2 import (HierarchicalEquityRatio, HarmAdjustedFairnessGap,
                                     EthicalRiskIndex, IntersectionalBiasScore)
    scores = {"White": 0.85, "Black": 0.78, "Hispanic": 0.80, "Asian": 0.87}
    her = HierarchicalEquityRatio()
    print("HER:", {k: round(v["score"], 3) for k, v in her.calculate_her(scores).items()})
    print("Bias-Gini:", round(her.calculate_bias_gini(list(scores.values())), 4))

    hafg = HarmAdjustedFairnessGap().calculate_hafg({"fn": 5, "fp": 10}, {"fn": 2, "fp": 5})
    print("HAFG:", hafg["hafg"])

    eri = EthicalRiskIndex().calculate_eri(
        [{"severity": 0.8}, {"severity": 0.3}, {"severity": 0.9}], n_total_outputs=100)
    print("ERI:", round(eri["eri"], 4), "| SVR:", round(eri["svr"], 2))

    ibs = IntersectionalBiasScore().calculate_subgroup_similarity(
        {"White_M": np.array([0.85, 0.9]), "Black_F": np.array([0.7, 0.6]),
         "Asian_M": np.array([0.88, 0.85])})
    print("IBS outlier:", ibs["outlier_subgroup"])


# ---------------------------------------------------------------- Domain 3
def d3():
    from equimed_dss.domain3 import (AuditTraceabilityScore, GovernanceComplianceIndex,
                                     TemporalFairnessDrift)
    print("ATS:", AuditTraceabilityScore().calculate_ats(n_traceable=92, n_total=100)["ats_score"])
    gci = GovernanceComplianceIndex().calculate_gci(
        {"audit_logging": True, "bias_testing": True, "human_oversight": False})
    print("GCI:", gci["gci"])
    tfd = TemporalFairnessDrift().calculate_drift([0.80, 0.82, 0.79, 0.85, 0.91, 0.95])
    print("TFD drift_detected:", tfd["drift_detected"])


# ---------------------------------------------------------------- Domain 4
def d4():
    from equimed_dss.domain4 import (SemanticParityGap, ClinicalHallucinationRate,
        InstructionalVulnerabilityIndex, GeographicRepresentationIndex)
    ep = rng.normal(size=(10, 8)); em = rng.normal(loc=0.3, size=(10, 8))
    print("SPG:", round(SemanticParityGap().calculate_spg(ep, em)["spg_euclidean"], 4))
    print("CHR:", ClinicalHallucinationRate().calculate_chr([0.2, 0.4, 0.8, 0.9], tau=0.5)["chr"])
    print("IVI:", InstructionalVulnerabilityIndex().calculate_ivi(
        ["acs", "non_cardiac", "acs"], ["acs", "acs", "acs"])["ivi_flip_rate"])
    gri = GeographicRepresentationIndex().calculate_gri(
        ["US", "GB", "BR", "CN", "TZ"], ["US", "GB", "DE"])
    print("GRI:", gri["gri"])


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
    print("ICE:", round(IntersectionalCalibrationError().calculate_ice(
        g, [0.9] * 8, [1, 1, 1, 0, 0, 0, 0, 0], n_bins=5)["delta_ice"], 3))
    print("wHAFG:", WeightedClinicalHarmAdjustedFairnessGap().calculate_whafg(
        ["m", "m", "p", "p"], [1, 1, 1, 1], [1, 1, 0, 0])["whafg_max"])
    print("LDDI:", round(LexicalDiversityDisparityIndex().calculate_lddi(
        {"A": ["pain pain pain"], "B": ["chest pain radiating to the left arm"]})["lddi"], 3))
    print("REG:", RecommendationEntropyGap().calculate_reg(
        {"A": ["acs", "acs"], "B": ["acs", "non_cardiac"]})["reg"])
    print("CPS:", CounterfactualParityScore().calculate_cps([1.0, 0.8, 0.9])["cps"])
    print("CIDR_min:", ClinicalInformationDensityRatio().calculate_cidr(
        {"A": [(5, 100)], "B": [(10, 100)]})["cidr_min"])
    print("dDCI:", DiagnosticCompletenessIndex().calculate_dci(
        ["ACS", "PE", "GERD"], {"A": [["ACS", "PE", "GERD"]], "B": [["ACS"]]})["delta_dci"])
    print("UQG:", round(UncertaintyQuantificationGap().calculate_uqg(
        {"A": ["This is ACS."], "B": ["This may be ACS. Consider PE."]})["uqg"], 3))
    print("GRBI:", round(GeographicRepresentationBiasIndex().calculate_grbi(
        {"AMRO": 100, "EURO": 10, "AFRO": 1}, {"AMRO": 0.2, "EURO": 0.4, "AFRO": 0.4})["grbi"], 3))
    print("HSSF:", HealthcareSystemStratifiedFairness().calculate_hssf(
        ["US", "US", "UK", "UK"], ["m", "p", "m", "p"], [0.0, 0.4, 0.1, 0.2])["hssf"])
    race = rng.choice(["W", "B"], 200); gender = rng.choice(["F", "M"], 200)
    isfv = IntersectionalShapleyFairnessValue(min_cell=5).calculate_isfv(
        {"race": race, "gender": gender}, (race == "B").astype(float))
    print("ISFV shapley:", {k: round(v, 3) for k, v in isfv["shapley_by_attribute"].items()})
    print("SRPI:", round(SemanticRobustnessParityIndex().calculate_srpi(
        {"A": [0.9, 0.9], "B": [0.6, 0.6]})["srpi"], 3))


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
    print("BCI:", round(BiasConcentrationIndex().calculate_bci([0.1, 0.4, 0.3, 0.2])["bci"], 3))
    boot = BootstrapConfidenceIntervals(n_bootstrap=500, random_state=42).calculate_bci(
        rng.normal(0.7, 0.1, 100))
    print("BootstrapCI:", [round(boot["ci_lower"], 3), round(boot["ci_upper"], 3)])
    p = np.array([0.9, 0.85, 0.78, 0.92]); q = np.array([0.75, 0.70, 0.68, 0.72])
    print("JSD:", round(JensenShannonDivergence().calculate_jsd(p, q)["jsd"], 4))
    print("WD:", round(WassersteinDistance().calculate_wd(p, q)["wasserstein_distance"], 4))
    mic = MutualInformationContent().calculate_mic(
        rng.randint(0, 2, 200), rng.randint(0, 2, 200))
    print("MIC:", round(mic["mic"], 4))
    adj = np.array([[0, .8, .1, 0], [.8, 0, 0, .7], [.1, 0, 0, .6], [0, .7, .6, 0]])
    print("Modularity:", round(NetworkModularity().calculate_modularity(adj)["modularity"], 3))
    rcs = RobustnessCertificationScore().calculate_rcs(
        rng.normal(0.8, 0.05, 50), [rng.normal(0.8, 0.05, 50) for _ in range(5)], epsilon=0.1)
    print("RCS:", round(rcs["rcs"], 3))
    ts = TransparencyScore().calculate_ts([
        {"explanation_quality": 0.9, "feature_importance": 0.8, "interpretability": 0.85},
        {"explanation_quality": 0.7, "feature_importance": 0.6, "interpretability": 0.65}])
    print("Transparency (TS):", round(ts["ts"], 3))
    spa = StatisticalPowerAnalysis().calculate_sample_size(effect_size=0.2, alpha=0.05, power=0.8)
    print("SPA required n:", spa.get("required_sample_size", spa))


for name, fn in [("Domain 1", d1), ("Domain 2", d2), ("Domain 3", d3),
                 ("Domain 4", d4), ("Domain 5", d5), ("Statistics", stats),
                 ("Appendix", appendix)]:
    show(name, fn)
