"""Regression tests for the changes made in 1.10.0 after an external methods review.

Expected values are computed independently of the library where possible.
"""
import math
import warnings

import numpy as np
import pandas as pd
import pytest

from equimed_dss.appendix import (
    AdvancedNetworkMetrics,
    BiasConcentrationIndex,
    JensenShannonDivergence,
    MutualInformationContent,
    NetworkModularity,
    ObservedPerturbationAgreement,
    RobustnessCertificationScore,
    StatisticalPowerAnalysis,
    TransparencyScore,
    WassersteinDistance,
)
from equimed_dss.appendix.reliability import AdvancedReliabilityMetrics
from equimed_dss.domain2 import (
    EthicalRiskIndex,
    HarmAdjustedFairnessGap,
    HierarchicalEquityRatio,
    IntersectionalBiasScore,
)
from equimed_dss.domain3 import AuditTraceabilityScore, GovernanceComplianceIndex
from equimed_dss.domain3.tfd import TemporalFairnessDrift
from equimed_dss.domain4 import GeographicRepresentationIndex, SemanticParityGap
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
from equimed_dss.geographic import GeographicConcentration
from equimed_dss.inference import (
    block_bootstrap_ci,
    bootstrap_ci,
    permutation_test,
    proportion_ci,
    stratified_bootstrap_ci,
)
from equimed_dss.statistics import HierarchicalLinearModeling, MediationAnalysis
from equimed_dss.statistics.reliability_stats import ReliabilityAnalysis

STRATIFIED = "bootstrap (stratified by group)"


# --- inference helpers ---------------------------------------------------------

def test_stratified_bootstrap_keeps_every_group_in_every_replicate():
    seen = []

    def stat(sample):
        seen.append(sorted(len(v) for v in sample.values()))
        return float(np.mean(sample["a"]) - np.mean(sample["b"]))

    stratified_bootstrap_ci(
        {"a": np.arange(5.0), "b": np.arange(7.0)}, stat, n_boot=50, random_state=0
    )
    assert all(sizes == [5, 7] for sizes in seen)


def test_stratified_bootstrap_flags_small_groups_and_counts_failures():
    with pytest.warns(UserWarning, match="fewer than 5"):
        stratified_bootstrap_ci({"a": [1.0], "b": [1.0, 2.0, 3.0, 4.0, 5.0]},
                                lambda s: float(np.mean(s["b"])), n_boot=20)
    r = stratified_bootstrap_ci(
        {"a": np.array([0.0, 1.0] * 5)},
        lambda s: 1.0 / float(np.mean(s["a"])) if np.mean(s["a"]) > 0 else float("nan"),
        n_boot=200,
    )
    assert r.n_failed is not None and r.n_failed >= 0


def test_resampling_inputs_are_validated():
    with pytest.raises(ValueError):
        bootstrap_ci([1.0, 2.0], np.mean, n_boot=1)
    with pytest.raises(ValueError):
        bootstrap_ci([1.0, 2.0], np.mean, conf=1.5)
    with pytest.raises(ValueError):
        proportion_ci(3, 10, null_value=0.0)
    with pytest.raises(ValueError):
        permutation_test([1.0], [2.0], n_perm=0)


def test_block_bootstrap_uses_blocks():
    r = block_bootstrap_ci(np.arange(27.0), np.mean, n_boot=100)
    assert r.method == "moving-block bootstrap (block 3)"   # round(27 ** (1/3))


# --- HER, Bias-Gini, HAFG, wHAFG -----------------------------------------------

def test_her_interval_is_stratified_and_scores_are_validated():
    obs = {"White": [0.8, 0.9, 0.85, 0.85, 0.9], "Black": [0.7, 0.8, 0.75, 0.75, 0.7]}
    res = HierarchicalEquityRatio().calculate_her({"White": 0.86, "Black": 0.74},
                                                  group_observations=obs)
    assert "stratified" in str(res)
    assert res["Black"]["interpretation"]["verdict"] == "Within the 0.8-1.25 band"
    with pytest.raises(ValueError):
        HierarchicalEquityRatio().calculate_her({"White": 0.8, "Black": -0.1})


def test_bias_gini_interval_needs_observations():
    her = HierarchicalEquityRatio()
    assert "unavailable" in str(her.calculate_bias_gini([0.85, 0.75, 0.80]))
    obs = {"a": [0.8, 0.9, 0.85, 0.8, 0.9], "b": [0.7, 0.8, 0.75, 0.7, 0.8]}
    assert "stratified" in str(her.calculate_bias_gini(group_observations=obs))
    with pytest.raises(ValueError):
        her.calculate_bias_gini([0.1, 0.2], group_observations=obs)


def test_hafg_rejects_inconsistent_cases_and_reports_per_patient_gap():
    hafg = HarmAdjustedFairnessGap()           # cost_fn = 10, cost_fp = 3
    with pytest.raises(ValueError):
        hafg.calculate_hafg({"fn": 5, "fp": 0}, {"fn": 1, "fp": 0},
                            group1_cases=["fn", "tn"], group2_cases=["fn", "tn"])
    with pytest.warns(UserWarning, match="differ in size"):
        res = hafg.calculate_hafg({"fn": 4, "fp": 0}, {"fn": 2, "fp": 0},
                                  group1_n=100, group2_n=50)
    # Total harm 40 vs 20 (HAFG 0.5) but 0.4 vs 0.4 per patient (gap 0)
    assert res["hafg"] == pytest.approx(0.5)
    assert res["hafg_per_patient"] == pytest.approx(0.0)


def test_whafg_interval_is_stratified_and_inputs_validated():
    groups = ["a"] * 6 + ["b"] * 6
    w = [1.0] * 12
    loss = [1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0]
    res = WeightedClinicalHarmAdjustedFairnessGap().calculate_whafg(groups, w, loss)
    assert res["whafg_max"] == pytest.approx(4 / 6 - 2 / 6)
    assert res["ci_method"] == STRATIFIED
    with pytest.raises(ValueError):
        WeightedClinicalHarmAdjustedFairnessGap().calculate_whafg(["a", "b"], [-1.0, 1.0], [1, 1])


# --- ICE, CPS, SRPI, HSSF --------------------------------------------------------

def test_ice_outcome_and_bins_are_validated():
    ice = IntersectionalCalibrationError()
    with pytest.raises(ValueError):
        ice.calculate_ice(["a", "b"], [0.2, 0.4], [0, 2])
    with pytest.raises(ValueError):
        ice.calculate_ice(["a", "b"], [0.2, float("nan")], [0, 1])
    with pytest.raises(ValueError):
        ice.calculate_ice(["a", "b"], [0.2, 0.4], [0, 1], n_bins=0)
    res = ice.calculate_ice(["a"] * 6 + ["b"] * 6, np.linspace(0.05, 0.95, 12),
                            [0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1], n_bins=5)
    assert res["ci_method"] == STRATIFIED


def test_cps_requires_a_0_1_scale_and_reports_a_cfu_interval():
    cps = CounterfactualParityScore()
    with pytest.raises(ValueError):
        cps.calculate_cps([0.9, -0.2, 0.8])
    res = cps.calculate_cps({"p1": [0.9, 0.8, 0.85, 0.9, 0.95], "p2": [0.7, 0.75, 0.8, 0.7, 0.75]})
    assert res["cfu"] == pytest.approx(1 - 0.74)
    assert 0 <= res["cfu_ci_lower"] <= res["cfu_ci_upper"] <= 1


def test_srpi_separates_parity_from_magnitude():
    res = SemanticRobustnessParityIndex().calculate_srpi({"a": [0.0, 0.0], "b": [0.0, 0.0]})
    assert math.isnan(res["srpi"])
    res = SemanticRobustnessParityIndex().calculate_srpi(
        {"a": [0.2, 0.2, 0.2, 0.2, 0.2], "b": [0.2, 0.2, 0.2, 0.2, 0.2]})
    assert res["srpi"] == 1.0 and res["max_robustness"] == pytest.approx(0.2)
    with pytest.raises(ValueError):
        SemanticRobustnessParityIndex().calculate_srpi({"a": [1.5], "b": [0.5]})


def test_hssf_excludes_single_group_systems_and_reports_same_unit_spread():
    systems = ["S1"] * 4 + ["S2"] * 4 + ["S3"] * 2
    groups = ["F", "M", "F", "M"] + ["F", "M", "F", "M"] + ["F", "F"]
    y = [1, 0, 1, 0, 1, 1, 0, 1, 0, 0]
    res = HealthcareSystemStratifiedFairness().calculate_hssf(systems, groups, y)
    # S1: F mean 1, M mean 0 -> gap 1; S2: F 0.5, M 1 -> gap 0.5; S3 has one group
    assert res["systems_without_comparison"] == ["S3"]
    assert res["hssf"] == pytest.approx((4 * 1.0 + 4 * 0.5) / 8)
    means = np.array([0.5, 0.75, 0.0])           # S1, S2, S3 mean outcomes
    assert res["between_system_range"] == pytest.approx(0.75)
    w = np.array([4, 4, 2]) / 10
    sd = math.sqrt(float((w * (means - (w * means).sum()) ** 2).sum()))
    assert res["between_system_sd"] == pytest.approx(sd)
    with pytest.raises(ValueError):
        HealthcareSystemStratifiedFairness().calculate_hssf(["A", "B"], ["F", "M"], [1, 0])


# --- text metrics ------------------------------------------------------------------

def test_text_metric_intervals_are_stratified():
    groups = {"A": ["chest pain radiating", "pain pain"], "B": ["no pain", "breath noted"]}
    assert LexicalDiversityDisparityIndex().calculate_lddi(groups)["ci_method"] == STRATIFIED
    recs = {"A": ["ecg", "trop", "ecg"], "B": ["ecg", "ecg", "ct"]}
    assert RecommendationEntropyGap().calculate_reg(recs)["ci_method"] == STRATIFIED
    cid = {"A": [(5, 50), (6, 40)], "B": [(2, 50), (3, 60)]}
    assert ClinicalInformationDensityRatio().calculate_cidr(cid)["ci_method"] == STRATIFIED
    uqg = UncertaintyQuantificationGap().calculate_uqg({"A": ["May be GERD."], "B": ["ACS."]})
    assert uqg["ci_method"] == STRATIFIED
    assert "not a measure of calibrated uncertainty" in uqg["interpretation"]


def test_dci_case_specific_references_and_empty_groups():
    dci = DiagnosticCompletenessIndex()
    res = dci.calculate_dci(
        None,
        {"A": [["acs", "pe"], ["acs"]], "B": [["acs"], ["pe"]]},
        references_by_group={"A": [["acs", "pe"], ["acs", "gerd"]], "B": [["acs"], ["pe", "pna"]]},
    )
    # A: (2/2 + 1/2)/2 = 0.75; B: (1/1 + 1/2)/2 = 0.75
    assert res["dci_by_group"] == {"A": 0.75, "B": 0.75}
    with pytest.raises(ValueError):
        dci.calculate_dci(["acs"], {"A": [["acs"]], "B": []})


# --- ISFV, IBS, TFD ----------------------------------------------------------------

def test_isfv_reports_attribution_intervals_and_a_permutation_check():
    rng = np.random.default_rng(0)
    n = 400
    race = rng.choice(["W", "B"], n)
    sex = rng.choice(["F", "M"], n)
    y = (rng.random(n) < 0.2 + 0.3 * (race == "B")).astype(float)
    res = IntersectionalShapleyFairnessValue(min_cell=30).calculate_isfv(
        {"race": race, "sex": sex}, y)
    assert set(res["shapley_ci"]) == {"race", "sex"}
    assert res["p_value_permutation"] < 0.05
    assert "penalty" not in res["interpretation"].replace("not by itself an intersectional penalty", "")
    with pytest.warns(UserWarning, match="fewer than 5"):
        IntersectionalShapleyFairnessValue().calculate_isfv(
            {"a": ["x", "y", "y", "y"]}, [1.0, 0.0, 1.0, 0.0])


def test_ibs_warns_that_formula_is_not_parsed():
    df = pd.DataFrame({"race": ["W", "B"] * 10, "gender": ["F", "M"] * 10,
                       "ses": ["lo", "hi"] * 10, "score": np.linspace(0, 1, 20)})
    with pytest.warns(UserWarning, match="does not parse"):
        IntersectionalBiasScore().interaction_analysis(df, formula="score ~ C(race)")


def test_tfd_baseline_monitoring_detects_a_shift_the_retrospective_chart_misses():
    s = [0.10, 0.11, 0.09, 0.10, 0.105, 0.095, 0.10, 0.10, 0.20, 0.21, 0.19, 0.20]
    tfd = TemporalFairnessDrift()
    assert tfd.calculate_drift(s)["drift_detected"] is False
    pro = tfd.calculate_drift(s, baseline_n=8)
    assert pro["out_of_control_indices"] == [8, 9, 10, 11]
    assert pro["mean_pdi"] == pytest.approx(np.mean(s[:8]))
    with pytest.raises(ValueError):
        tfd.calculate_drift([])


# --- geographic, governance --------------------------------------------------------

def test_gri_reports_the_volume_view():
    res = GeographicRepresentationIndex().calculate_gri(["US"] * 98 + ["KE", "NG"], ["US"])
    assert res["gri"] == pytest.approx(2 / 3)
    assert res["non_western_mention_share"] == pytest.approx(0.02)


def test_gcc_alias_and_finite_checks():
    res = GeographicConcentration().calculate_gcc({"A": 1.0, "B": 3.0})
    assert res["gini_normalized"] == res["gini_corrected"]
    with pytest.raises(ValueError):
        GeographicConcentration().calculate_gcc({"A": float("nan"), "B": 1.0})
    with pytest.raises(ValueError):
        GeographicRepresentationBiasIndex().calculate_grbi({"A": -1, "B": 2}, {"A": 0.5, "B": 0.5})


def test_eri_ats_gci_reject_undefined_inputs_and_use_neutral_labels():
    with pytest.raises(ValueError):
        EthicalRiskIndex().calculate_eri([{"severity": 1.0}], n_total_outputs=0)
    with pytest.raises(ValueError):
        EthicalRiskIndex().calculate_eri([{}], n_total_outputs=5)
    with pytest.raises(ValueError):
        EthicalRiskIndex().calculate_eri([{"severity": -1.0}], n_total_outputs=5)
    with pytest.raises(ValueError):
        AuditTraceabilityScore().calculate_ats(0, 0)
    assert AuditTraceabilityScore().calculate_ats(19, 20)["interpretation"]["verdict"] == (
        "Meets the 0.95 target")
    with pytest.raises(ValueError):
        GovernanceComplianceIndex().calculate_gci({})
    gci = GovernanceComplianceIndex().calculate_gci({"p1": True, "p2": False})
    assert gci["interpretation"]["verdict"] == "50% of listed checks met"


# --- SPG ------------------------------------------------------------------------------

def test_spg_paired_inference_respects_case_pairs():
    rng = np.random.default_rng(1)
    base = rng.normal(size=(40, 6))
    p = base + rng.normal(scale=0.05, size=(40, 6))
    m = base + 0.05 + rng.normal(scale=0.05, size=(40, 6))
    spg = SemanticParityGap()
    assert spg.calculate_spg(p, m)["p_value_permutation"] > 0.5
    paired = spg.calculate_spg(p, m, paired=True)
    assert paired["p_value_permutation"] < 0.01 and paired["paired"] is True
    with pytest.raises(ValueError):
        spg.calculate_spg(p, m[:30], paired=True)


# --- appendix -------------------------------------------------------------------------

def test_mic_requires_discrete_outcomes_and_reports_a_permutation_null():
    mic = MutualInformationContent()
    with pytest.raises(ValueError):
        mic.calculate_mic(np.array([0, 1, 0, 1]), np.array([0.13, 0.52, 0.77, 0.21]))
    rng = np.random.default_rng(0)
    res = mic.calculate_mic(rng.integers(0, 3, 200), rng.integers(0, 2, 200))
    assert {"mic_null_mean", "p_value_permutation"} <= set(res)
    assert "Intervention" not in str(res["interpretation"])


def test_bias_concentration_has_no_interval():
    res = BiasConcentrationIndex().calculate_bci([0.25, 0.25, 0.25, 0.25])
    assert "ci_lower" not in res
    assert "not how large" in res["interpretation"]["note"]


def test_jsd_validates_and_wd_has_no_units_free_verdict():
    jsd = JensenShannonDivergence()
    for p, q in (([0.5, 0.5], [1.0]), ([0.5, -0.5], [0.5, 0.5]), ([], [])):
        with pytest.raises(ValueError):
            jsd.calculate_jsd(p, q)
    wd = WassersteinDistance().calculate_wd([0.1, 0.2, 0.3], [0.4, 0.5, 0.6])
    assert "Equitable" not in str(wd["interpretation"])


def test_modularity_interval_resamples_observations():
    rng = np.random.default_rng(0)
    z = rng.normal(size=(60, 2))
    obs = np.column_stack([z[:, 0], z[:, 0] + 0.3 * rng.normal(size=60),
                           z[:, 1], z[:, 1] + 0.3 * rng.normal(size=60)])
    corr = np.corrcoef(obs, rowvar=False)
    assert "ci_lower" not in NetworkModularity().calculate_modularity(corr)
    res = NetworkModularity().calculate_modularity(corr, observations=obs)
    assert res["ci_method"] == "bootstrap (observations)"


def test_transparency_score_validates_ratings():
    ts = TransparencyScore()
    with pytest.raises(ValueError):
        ts.calculate_ts([])
    with pytest.raises(ValueError):
        ts.calculate_ts([{"explanation_quality": 0.8, "feature_importance": 0.7}])
    with pytest.raises(ValueError):
        ts.calculate_ts([{"explanation_quality": 1.8, "feature_importance": 0.7,
                          "interpretability": 0.9}])


def test_perturbation_agreement_names():
    res = ObservedPerturbationAgreement().calculate_agreement([1, 0, 1, 1], [[1, 0, 1, 0]])
    assert res["agreement"] == res["rcs"] == pytest.approx(0.75)
    assert "certified" not in res["interpretation"]["note"].replace("not a certified", "")
    with pytest.warns(DeprecationWarning):
        RobustnessCertificationScore()
    with pytest.raises(ValueError):
        ObservedPerturbationAgreement().calculate_agreement([1, 0], [])


def test_aggregator_names_are_distinct_and_old_names_deprecated():
    agg = AdvancedNetworkMetrics()
    assert agg.calculate_explained_fraction(3, 4) == 0.75
    r = agg.calculate_stability_pass_rate([0.9, 0.7, 0.85, 0.95])
    assert r["pass_rate"] == 0.75 and r["meets_target"] is False
    with pytest.warns(DeprecationWarning):
        assert agg.calculate_transparency_score(3, 4) == 0.75
    with pytest.warns(DeprecationWarning):
        assert agg.calculate_rcs([0.9, 0.7])["rcs_score"] == 0.5


def test_total_sample_size_is_twice_the_group_size():
    for d in (0.2, 0.33, 0.5, 0.71, 1.1):
        r = StatisticalPowerAnalysis().calculate_sample_size(d)
        assert r["total_n"] == 2 * r["n_per_group"]
        a = AdvancedReliabilityMetrics().calculate_power_analysis(d)
        assert a["total_n"] == 2 * a["required_n_per_group"]


# --- statistics -----------------------------------------------------------------------

def test_mediation_is_associational_and_handles_categories():
    rng = np.random.default_rng(3)
    n = 300
    grp = rng.choice(["Black", "White"], n)
    x = (grp == "White").astype(float)
    m = 0.8 * x + rng.normal(size=n)
    y = 0.6 * m + rng.normal(size=n)
    site = rng.choice(["s1", "s2", "s3"], n)
    med = MediationAnalysis(n_bootstrap=100, random_state=0)
    r = med.analyze_mediation(pd.DataFrame({"g": grp, "m": m, "y": y, "site": site}),
                              "g", "m", "y", covariates=["site"])
    assert r["treatment_coding"] == {"reference (0)": "Black", "exposed (1)": "White"}
    assert r["estimand"].startswith("associational")
    assert "intervention" not in r["interpretation"]["clinical_implication"].lower().replace(
        "which intervention would work", "")
    with pytest.raises(ValueError):
        med.analyze_mediation(pd.DataFrame({"g": rng.choice(list("abc"), n), "m": m, "y": y}),
                              "g", "m", "y")


def test_mediation_proportion_is_nan_when_total_effect_is_zero():
    # x alternates 0, 1 and y cycles 1, 2, 2, 1: the slope of y on x is exactly 0.
    x = np.array([0.0, 1.0] * 50)
    y = np.array([1.0, 2.0, 2.0, 1.0] * 25)
    m = np.random.default_rng(0).normal(size=100)
    r = MediationAnalysis(n_bootstrap=50, random_state=0).analyze_mediation(
        pd.DataFrame({"x": x, "m": m, "y": y}), "x", "m", "y")
    assert r["total_effect"] == pytest.approx(0.0, abs=1e-12)
    assert math.isnan(r["proportion_mediated"])
    assert "undefined" in r["interpretation"]["proportion_description"]


def test_hlm_and_bland_altman_interpretations_are_descriptive():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"h": np.repeat([f"h{i}" for i in range(8)], 20),
                       "x": rng.normal(size=160)})
    df["y"] = 0.5 * df["x"] + np.repeat(rng.normal(size=8), 20) + rng.normal(size=160)
    res = HierarchicalLinearModeling().fit_model(df, "y", ["x"], "h")
    assert "intervene" in res["interpretation"]["clinical_implication"]
    ba = ReliabilityAnalysis().bland_altman_analysis([1.0, 2.0, 3.0, 4.0], [1.1, 2.0, 2.9, 4.2])
    assert "limits of agreement" in ba["interpretation"]["agreement"]
