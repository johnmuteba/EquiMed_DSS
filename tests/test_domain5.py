"""Tests for the 12 domain5 metrics (technical-supplement fairness metrics)."""
import numpy as np
import pytest

from equimed_dss.domain5 import (
    IntersectionalCalibrationError,
    WeightedClinicalHarmAdjustedFairnessGap,
    LexicalDiversityDisparityIndex,
    RecommendationEntropyGap,
    CounterfactualParityScore,
    ClinicalInformationDensityRatio,
    DiagnosticCompletenessIndex,
    UncertaintyQuantificationGap,
    GeographicRepresentationBiasIndex,
    HealthcareSystemStratifiedFairness,
    IntersectionalShapleyFairnessValue,
    SemanticRobustnessParityIndex,
)


def test_ice_detects_group_miscalibration():
    # group A perfectly calibrated (conf=acc), group B overconfident
    groups = ["A"] * 4 + ["B"] * 4
    conf = [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]
    correct = [1, 1, 1, 0, 0, 0, 0, 0]  # A: 75% acc vs 90 conf; B: 0% acc vs 90
    res = IntersectionalCalibrationError().calculate_ice(groups, conf, correct, n_bins=5)
    assert res["delta_ice"] > 0
    assert res["ece_by_group"]["B"] > res["ece_by_group"]["A"]


def test_whafg_gap():
    groups = ["m", "m", "p", "p"]
    sev = [1.0, 1.0, 1.0, 1.0]
    loss = [1.0, 1.0, 0.0, 0.0]  # H(m)=1, H(p)=0
    res = WeightedClinicalHarmAdjustedFairnessGap().calculate_whafg(groups, sev, loss)
    assert res["whafg_max"] == pytest.approx(1.0, abs=1e-9)
    assert res["most_harmed_group"] == "m"


def test_lddi_identical_vs_diverse():
    res = LexicalDiversityDisparityIndex().calculate_lddi(
        {"A": ["pain pain pain pain"], "B": ["chest pain radiating to left arm with dyspnea"]}
    )
    assert res["lddi"] > 0
    assert res["rttr_by_group"]["B"] > res["rttr_by_group"]["A"]


def test_reg_uniform_vs_pointmass():
    # group A point mass (entropy 0), group B uniform over 2 (entropy 1 bit)
    res = RecommendationEntropyGap().calculate_reg(
        {"A": ["acs", "acs", "acs", "acs"], "B": ["acs", "acs", "non_cardiac", "non_cardiac"]}
    )
    assert res["entropy_by_group"]["A"] == pytest.approx(0.0, abs=1e-9)
    assert res["entropy_by_group"]["B"] == pytest.approx(1.0, abs=1e-9)
    assert res["reg"] == pytest.approx(1.0, abs=1e-9)


def test_cidr_min():
    res = ClinicalInformationDensityRatio().calculate_cidr(
        {"A": [(5, 100), (5, 100)], "B": [(10, 100)]}  # CID A=5, B=10 -> CIDR_min=0.5
    )
    assert res["cidr_min"] == pytest.approx(0.5, abs=1e-9)


def test_dci_coverage_gap():
    Dstar = ["ACS", "PE", "Dissection", "GERD"]
    res = DiagnosticCompletenessIndex().calculate_dci(
        Dstar,
        {"A": [["ACS", "PE", "Dissection", "GERD"]], "B": [["ACS"]]},  # 1.0 vs 0.25
    )
    assert res["dci_by_group"]["A"] == pytest.approx(1.0, abs=1e-9)
    assert res["dci_by_group"]["B"] == pytest.approx(0.25, abs=1e-9)
    assert res["delta_dci"] == pytest.approx(0.75, abs=1e-9)


def test_uqg_hedging_gap():
    res = UncertaintyQuantificationGap().calculate_uqg(
        {"A": ["This is ACS. Start treatment."],  # no hedging
         "B": ["This may be ACS. Consider possible PE."]}  # hedging present
    )
    assert res["uqg"] > 0
    assert res["ud_by_group"]["B"] > res["ud_by_group"]["A"]


def test_grbi_zero_when_matched_and_positive_otherwise():
    burden = {"AFRO": 0.25, "AMRO": 0.25, "EURO": 0.25, "SEARO": 0.25}
    same = GeographicRepresentationBiasIndex().calculate_grbi(burden, burden)
    assert same["grbi"] == pytest.approx(0.0, abs=1e-9)
    skewed = GeographicRepresentationBiasIndex().calculate_grbi(
        {"AMRO": 100, "EURO": 0, "AFRO": 0, "SEARO": 0}, burden, hic_regions=["AMRO", "EURO"]
    )
    assert skewed["grbi"] > 0
    assert skewed["hic_ratio"] > 1


def test_hssf_within_system_gap():
    # one system, two groups differ by 0.4 in outcome
    systems = ["US"] * 4
    groups = ["m", "m", "p", "p"]
    y = [0.0, 0.0, 0.4, 0.4]
    res = HealthcareSystemStratifiedFairness().calculate_hssf(systems, groups, y)
    assert res["hssf"] == pytest.approx(0.4, abs=1e-9)


def test_isfv_efficiency_and_interaction():
    # race drives a disparity; gender does not
    rng = np.random.RandomState(0)
    n = 400
    race = rng.choice(["W", "B"], n)
    gender = rng.choice(["F", "M"], n)
    y = (race == "B").astype(float)  # outcome depends only on race
    res = IntersectionalShapleyFairnessValue(min_cell=5).calculate_isfv(
        {"race": race, "gender": gender}, y
    )
    # Shapley values sum to total disparity (efficiency); race >> gender
    assert sum(res["shapley_by_attribute"].values()) == pytest.approx(
        res["total_disparity"], abs=1e-6
    )
    assert res["shapley_by_attribute"]["race"] > res["shapley_by_attribute"]["gender"]


def test_cps_and_cfu():
    res = CounterfactualParityScore().calculate_cps([1.0, 0.8, 0.9])
    assert res["cps"] == pytest.approx(0.9, abs=1e-9)
    assert res["cfu"] == pytest.approx(0.1, abs=1e-9)


def test_srpi_ratio():
    res = SemanticRobustnessParityIndex().calculate_srpi(
        {"A": [0.9, 0.9], "B": [0.6, 0.6]}  # min/max = 0.6/0.9
    )
    assert res["srpi"] == pytest.approx(0.6 / 0.9, abs=1e-9)
    assert res["least_robust_group"] == "B"
