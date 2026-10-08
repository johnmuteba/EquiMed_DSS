"""Regression tests for the corrections made in 1.10.0 (metric-by-metric review).

Each expected value is computed independently of the library (textbook formula
or hand calculation), so a test fails if the defect returns.
"""
import math
import warnings
from statistics import NormalDist

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from equimed_dss import WHO_REGION_CODES, WHO_REGION_IHD_BURDEN
from equimed_dss.appendix import (
    AdvancedNetworkAnalysis,
    AdvancedNetworkMetrics,
    AdvancedReliabilityMetrics,
    BiasConcentrationIndex,
    MutualInformationContent,
    NetworkModularity,
    RobustnessCertificationScore,
    StatisticalPowerAnalysis,
    WassersteinDistance,
)
from equimed_dss.domain1 import (
    DecisionFlipRate,
    EmbeddingConsistencyScore,
    InterRaterReliability,
)
from equimed_dss.domain2 import (
    EthicalRiskIndex,
    HarmAdjustedFairnessGap,
    HierarchicalEquityRatio,
)
from equimed_dss.domain3.ats import AuditTraceabilityScore
from equimed_dss.domain3.tfd import TemporalFairnessDrift
from equimed_dss.domain4 import (
    ClinicalHallucinationRate,
    InstructionalVulnerabilityIndex,
    SemanticParityGap,
)
from equimed_dss.domain5 import (
    GeographicRepresentationBiasIndex,
    IntersectionalCalibrationError,
    UncertaintyQuantificationGap,
)
from equimed_dss.geographic import BurdenEvidenceMismatch, GeographicConcentration
from equimed_dss.inference import permutation_test, proportion_ci, wilson_ci
from equimed_dss.statistics import (
    HierarchicalLinearModeling,
    MediationAnalysis,
    NetworkStatistics,
)
from equimed_dss.utils.data_loader import generate_synthetic_judge_data

Z = NormalDist().inv_cdf


# --- appendix: power, concentration index, bootstrap -------------------------

def test_power_analysis_uses_two_sample_formula():
    # n per group = 2 (z_{0.975} + z_{0.80})^2 / d^2 = 62.8 -> 63 for d = 0.5
    expected = math.ceil(2 * (Z(0.975) + Z(0.80)) ** 2 / 0.5**2)
    res = AdvancedReliabilityMetrics().calculate_power_analysis(0.5)
    assert res["required_n_per_group"] == expected == 63
    t_test = StatisticalPowerAnalysis().calculate_sample_size(0.5)["n_per_group"]
    assert abs(res["required_n_per_group"] - t_test) <= 1


def test_concentration_index_equal_and_population_weights():
    h = np.array([0.3, 0.2, 0.1])
    arm = AdvancedReliabilityMetrics()
    # Equal shares: ranks 1/6, 1/2, 5/6; C = 2 cov_pop(h, R) / mean(h) = -2/9
    assert arm.calculate_bias_concentration(None, h) == pytest.approx(-2 / 9)
    # Grouped data: shares 0.1, 0.3, 0.6 give ranks 0.05, 0.25, 0.70 and
    # C = 2 * (-0.0165) / 0.15 = -0.22 (hand calculation)
    w = np.array([0.1, 0.3, 0.6])
    assert arm.calculate_bias_concentration(w, h) == pytest.approx(-0.22)
    with pytest.raises(ValueError):
        arm.calculate_bias_concentration(np.array([0.5, 0.5]), h)


def test_appendix_bootstrap_ci_is_reproducible_with_seed():
    arm = AdvancedReliabilityMetrics()
    data = [0.1, 0.4, 0.35, 0.8, 0.5, 0.9, 0.2]
    a = arm.calculate_bootstrap_ci(data, random_state=7)
    b = arm.calculate_bootstrap_ci(data, random_state=7)
    assert a == b


# --- appendix: RCS, MIC, BCI, WD, modularity ----------------------------------

def test_rcs_compares_lists_element_wise():
    lists = RobustnessCertificationScore().calculate_rcs([1, 0, 1, 1], [[1, 0, 1, 0]])
    assert lists["rcs"] == pytest.approx(0.75)
    with pytest.raises(ValueError):
        RobustnessCertificationScore().calculate_rcs([1, 0, 1], [[1, 0]])


def test_mic_accepts_string_categories():
    demo_s = np.array(["a", "b", "a", "b", "c", "c"])
    demo_i = np.array([0, 1, 0, 1, 2, 2])
    out = np.array([1, 0, 1, 0, 1, 0])
    s = MutualInformationContent().calculate_mic(demo_s, out)
    i = MutualInformationContent().calculate_mic(demo_i, out)
    assert s["mic"] == pytest.approx(i["mic"])
    assert s["normalized_mic"] == pytest.approx(i["normalized_mic"])


def test_bias_concentration_verdict_uses_normalized_index():
    res = BiasConcentrationIndex().calculate_bci([0.5, 0.5])
    assert res["bci"] == pytest.approx(0.5)            # maximum for 2 groups
    assert res["bci_normalized"] == pytest.approx(1.0)
    assert res["interpretation"]["verdict"] == "Acceptable (distributed)"


def test_wasserstein_histograms_with_support():
    wd = WassersteinDistance()
    # As samples, [0.2, 0.8] and [0.8, 0.2] are identical
    assert wd.calculate_wd([0.2, 0.8], [0.8, 0.2])["wasserstein_distance"] == 0.0
    # As histograms on {0, 1}, 0.6 of the mass moves a distance of 1
    hist = wd.calculate_wd([0.2, 0.8], [0.8, 0.2], support=[0.0, 1.0])
    assert hist["wasserstein_distance"] == pytest.approx(0.6)


def _block_corr():
    return np.array([[1.0, 0.8, 0.1, 0.1],
                     [0.8, 1.0, 0.1, 0.1],
                     [0.1, 0.1, 1.0, 0.8],
                     [0.1, 0.1, 0.8, 1.0]])


def test_modularity_ignores_diagonal_and_uses_weights():
    # Two blocks: m = 2.0, each node strength 1.0; Q = 2 * (0.8/2 - (2/4)^2) = 0.30
    nm = NetworkModularity().calculate_modularity(_block_corr())
    assert nm["modularity"] == pytest.approx(0.30)
    assert nm["n_communities"] == 2
    assert AdvancedNetworkMetrics().calculate_modularity(_block_corr()) == pytest.approx(0.30)


def test_network_statistics_has_no_self_loops():
    res = NetworkStatistics().analyze_network(_block_corr())
    assert max(res["degree_centrality"].values()) <= 1.0
    thr = NetworkStatistics().analyze_network(_block_corr(), threshold=0.5)
    assert thr["n_edges"] == 2


# --- appendix: network analysis ------------------------------------------------

def test_temporal_dynamics_treats_undirected_edges_as_unordered():
    g1, g2 = nx.Graph(), nx.Graph()
    g1.add_edge("A", "B")
    g2.add_edge("B", "A")
    res = AdvancedNetworkAnalysis().temporal_fairness_dynamics([g1, g2])
    assert res["mean_stability"] == 1.0


def test_metric_correlation_network_keeps_isolated_metrics():
    rng = np.random.RandomState(0)
    x = rng.randn(60)
    df = pd.DataFrame({"m1": x, "m2": x + 0.1 * rng.randn(60), "m3": rng.randn(60)})
    res = AdvancedNetworkAnalysis().metric_correlation_network(df, threshold=0.5)
    assert res["num_nodes"] == 3
    assert res["centrality"]["m3"] == 0.0


def test_concept_cooccurrence_matches_whole_words():
    ana = AdvancedNetworkAnalysis()
    res = ana.concept_cooccurrence_network(
        ["family history reviewed, aspirin given", "history of MI, aspirin given"],
        ["MI", "aspirin"],
    )
    assert int(res["cooccurrence_matrix"].loc["MI", "aspirin"]) == 1


# --- domain1 -----------------------------------------------------------------

def test_icc_verdict_bands_are_consistent():
    irr = InterRaterReliability()
    assert irr.interpret_score(0.75) == "Excellent"
    assert irr.interpret_score(0.60) == "Good"
    assert irr.interpret_score(0.40) == "Fair"
    res = irr.calculate_icc_2_1(generate_synthetic_judge_data(30, 3, random_state=1))
    assert res["interpretation"]["verdict"] == irr.interpret_score(res["score"])
    with pytest.raises(ValueError):
        irr.calculate_icc_2_1(np.ones((5, 1)))


def test_ecs_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        EmbeddingConsistencyScore().calculate_ecs(np.ones((4, 3)), np.ones((5, 3)))


def test_dfr_rejects_missing_and_empty():
    with pytest.raises(ValueError):
        DecisionFlipRate().calculate_dfr([1, float("nan")], [1, float("nan")])
    with pytest.raises(ValueError):
        DecisionFlipRate().calculate_dfr(["a", None], ["a", None])
    with pytest.raises(ValueError):
        DecisionFlipRate().calculate_dfr([], [])


# --- domain2 -----------------------------------------------------------------

def test_her_zero_reference_raises():
    with pytest.raises(ValueError):
        HierarchicalEquityRatio().calculate_her({"White": 0.0, "Black": 0.2})


def test_eri_rejects_more_violations_than_outputs():
    with pytest.raises(ValueError):
        EthicalRiskIndex().calculate_eri([{"severity": 1.0}] * 3, n_total_outputs=2)


def test_hafg_bootstrap_is_stratified_and_warns_on_unequal_sizes():
    g1 = ["fn"] * 2 + ["tn"] * 8
    g2 = ["fn"] * 1 + ["tn"] * 9
    with warnings.catch_warnings():
        warnings.simplefilter("error")          # consistent and equal: no warning
        res = HarmAdjustedFairnessGap().calculate_hafg(
            {"fn": 2, "fp": 0}, {"fn": 1, "fp": 0}, group1_cases=g1, group2_cases=g2
        )
    assert res["ci_method"] == "bootstrap (stratified by group)"
    with pytest.warns(UserWarning, match="differ in size"):
        HarmAdjustedFairnessGap().calculate_hafg(
            {"fn": 2, "fp": 0}, {"fn": 1, "fp": 0},
            group1_cases=g1, group2_cases=["fn"] + ["tn"] * 29,
        )


# --- domain3 -----------------------------------------------------------------

def test_ats_interval_is_wilson():
    k, n, z = 18, 20, 1.96
    p = k / n
    den = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    res = AuditTraceabilityScore().calculate_ats(k, n)
    assert res["ci_lower"] == pytest.approx(centre - half, abs=1e-4)
    assert res["ci_upper"] == pytest.approx(centre + half, abs=1e-4)
    with pytest.raises(ValueError):
        AuditTraceabilityScore().calculate_ats(21, 20)


def test_tfd_moving_range_detects_a_level_shift():
    series = [0.10] * 10 + [0.20] * 10
    tfd = TemporalFairnessDrift()
    assert tfd.calculate_drift(series)["drift_detected"] is False
    mr = tfd.calculate_drift(series, sigma_method="moving_range")
    assert mr["drift_detected"] is True
    # sigma = mean moving range / 1.128 = (0.1 / 19) / 1.128
    assert mr["std_pdi"] == pytest.approx(0.1 / 19 / 1.128)


# --- domain4 -----------------------------------------------------------------

def test_chr_rejects_missing_and_out_of_range_scores():
    with pytest.raises(ValueError):
        ClinicalHallucinationRate().calculate_chr([0.9, float("nan"), 0.1])
    with pytest.raises(ValueError):
        ClinicalHallucinationRate().calculate_chr([0.9, 1.4])


def test_ivi_rejects_missing_outputs():
    with pytest.raises(ValueError):
        InstructionalVulnerabilityIndex().calculate_ivi([1, None], [1, 0])


def test_spg_permutation_p_value():
    rng = np.random.default_rng(3)
    same = SemanticParityGap().calculate_spg(rng.normal(size=(30, 4)), rng.normal(size=(30, 4)))
    shifted = SemanticParityGap().calculate_spg(
        rng.normal(size=(30, 4)), rng.normal(loc=1.0, size=(30, 4))
    )
    assert same["p_value_permutation"] > 0.05
    assert shifted["p_value_permutation"] < 0.01


# --- domain5 -----------------------------------------------------------------

def test_uqg_counts_overlapping_hedges_once_and_keeps_decimals():
    u = UncertaintyQuantificationGap()
    assert u._ud("Cannot rule out ACS.") == 1.0
    assert u._ud("Troponin 0.04 ng/mL may rise.") == 1.0


def test_ice_reports_group_sizes():
    res = IntersectionalCalibrationError().calculate_ice(
        ["a", "a", "b"], [0.2, 0.4, 0.6], [0, 1, 1], n_bins=5
    )
    assert res["n_by_group"] == {"a": 2, "b": 1}


def test_grbi_records_must_use_known_regions():
    with pytest.raises(ValueError):
        GeographicRepresentationBiasIndex().calculate_grbi(
            {"A": 1, "B": 1}, {"A": 0.5, "B": 0.5}, corpus_records=["A", "C"]
        )


# --- geographic ----------------------------------------------------------------

def test_bemi_rejects_mismatched_region_codes():
    gho = {"AFR": 10, "AMR": 50, "EMR": 5, "EUR": 20, "SEAR": 5, "WPR": 10}
    with pytest.raises(ValueError, match="no entry in burden_shares"):
        BurdenEvidenceMismatch().calculate_bemi(gho, WHO_REGION_IHD_BURDEN)
    mapped = {WHO_REGION_CODES[k]: v for k, v in gho.items()}
    res = BurdenEvidenceMismatch().calculate_bemi(mapped, WHO_REGION_IHD_BURDEN)
    expected = 0.5 * sum(
        abs(mapped[r] / 100 - WHO_REGION_IHD_BURDEN[r]) for r in WHO_REGION_IHD_BURDEN
    )
    assert res["bemi"] == pytest.approx(expected)


def test_bemi_and_gcc_records_must_use_known_regions():
    burden = {"EURO": 0.5, "AMRO": 0.5}
    with pytest.raises(ValueError):
        BurdenEvidenceMismatch().calculate_bemi(
            {"EURO": 1, "AMRO": 1}, burden, evidence_records=["EURO", "AFRO"]
        )
    with pytest.raises(ValueError):
        GeographicConcentration().calculate_gcc({"A": 1, "B": 1}, region_records=["A", "Z"])


# --- inference -----------------------------------------------------------------

def test_wilson_bounds_stay_in_unit_interval():
    assert wilson_ci(0, 10).ci_lower == 0.0
    assert wilson_ci(10, 10).ci_upper == 1.0


def test_unknown_alternative_raises():
    with pytest.raises(ValueError):
        permutation_test([1.0, 2.0], [3.0, 4.0], alternative="two_sided")
    with pytest.raises(ValueError):
        proportion_ci(3, 10, null_value=0.2, alternative="bigger")


# --- statistics ----------------------------------------------------------------

def _nested(seed=0, n_groups=12, n_per=25):
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_groups):
        u = rng.normal(0, 1.0)
        z = rng.normal()
        for _ in range(n_per):
            x = rng.normal()
            rows.append({"hosp": f"h{g}", "x": x, "z": z,
                         "y": 0.5 * x + 0.8 * z + u + rng.normal(0, 1.0)})
    return pd.DataFrame(rows)


def test_hlm_uses_level2_predictors():
    res = HierarchicalLinearModeling().fit_model(_nested(), "y", ["x"], "hosp", ["z"])
    assert res["method"] == "mixedlm"
    assert "z" in [c["term"] for c in res["coefficients"]]


def test_hlm_fallback_warns_and_reports_variance_components(monkeypatch):
    from statsmodels.regression import mixed_linear_model

    def boom(*args, **kwargs):
        raise RuntimeError("forced failure")

    monkeypatch.setattr(mixed_linear_model.MixedLM, "from_formula", boom)
    df = _nested()
    with pytest.warns(UserWarning, match="ANOVA"):
        res = HierarchicalLinearModeling().fit_model(df, "y", ["x"], "hosp")
    assert res["method"] == "anova"
    # Balanced groups: n0 = 25, sigma_u^2 = (MSB - MSW) / n0
    assert res["n0"] == pytest.approx(25.0)
    assert res["variance_between_groups"] == pytest.approx(
        max(0.0, (res["ms_between"] - res["ms_within"]) / 25.0)
    )
    assert res["icc"] == pytest.approx(
        res["variance_between_groups"]
        / (res["variance_between_groups"] + res["variance_within_groups"])
    )


def test_mediation_types_use_direct_effect_ci():
    rng = np.random.default_rng(5)
    n = 400
    x = rng.normal(size=n)
    m = 0.8 * x + rng.normal(size=n)
    y_full = 0.7 * m + rng.normal(size=n)                 # no direct effect
    y_part = 0.7 * m + 0.6 * x + rng.normal(size=n)       # direct effect too
    med = MediationAnalysis(n_bootstrap=300, random_state=1)
    full = med.analyze_mediation(pd.DataFrame({"x": x, "m": m, "y": y_full}), "x", "m", "y")
    part = med.analyze_mediation(pd.DataFrame({"x": x, "m": m, "y": y_part}), "x", "m", "y")
    assert full["interpretation"]["mediation_type"] == "Complete mediation"
    assert full["direct_ci_lower"] < 0 < full["direct_ci_upper"]
    assert part["interpretation"]["mediation_type"] == "Partial mediation (complementary)"


# --- utils ---------------------------------------------------------------------

def test_synthetic_generators_are_reproducible():
    a = generate_synthetic_judge_data(5, 2, random_state=11)
    b = generate_synthetic_judge_data(5, 2, random_state=11)
    assert np.array_equal(a, b)
