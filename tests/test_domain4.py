"""Tests for domain4 metrics (SPG, CHR, IVI, GRI)."""
import numpy as np
import pytest

from equimed_dss.domain4 import (
    SemanticParityGap,
    ClinicalHallucinationRate,
    InstructionalVulnerabilityIndex,
    GeographicRepresentationIndex,
)


class TestSemanticParityGap:
    def test_identical_clusters_give_zero(self):
        emb = np.tile(np.array([[1.0, 0.0, 0.0]]), (5, 1))
        res = SemanticParityGap().calculate_spg(emb, emb)
        assert res["spg_euclidean"] == pytest.approx(0.0, abs=1e-9)
        assert res["spg_cosine"] == pytest.approx(0.0, abs=1e-9)

    def test_known_distance(self):
        p = np.array([[0.0, 0.0]])
        m = np.array([[3.0, 4.0]])
        res = SemanticParityGap().calculate_spg(p, m)
        assert res["spg_euclidean"] == pytest.approx(5.0, abs=1e-9)

    def test_orthogonal_cosine_is_one(self):
        res = SemanticParityGap().calculate_spg([[1.0, 0.0]], [[0.0, 1.0]])
        assert res["spg_cosine"] == pytest.approx(1.0, abs=1e-9)

    def test_dim_mismatch_raises(self):
        with pytest.raises(ValueError):
            SemanticParityGap().calculate_spg([[1.0, 0.0]], [[1.0, 0.0, 0.0]])


class TestClinicalHallucinationRate:
    def test_basic_rate(self):
        # scores below tau=0.5: 0.2, 0.4 -> 2 of 4 unsupported
        res = ClinicalHallucinationRate().calculate_chr([0.2, 0.4, 0.8, 0.9], tau=0.5)
        assert res["chr"] == pytest.approx(0.5, abs=1e-9)
        assert res["n_unsupported"] == 2

    def test_weighted(self):
        # only the high-weight claim is unsupported
        res = ClinicalHallucinationRate().calculate_chr(
            [0.1, 0.9, 0.9], tau=0.5, weights=[10.0, 1.0, 1.0]
        )
        assert res["chr"] == pytest.approx(1 / 3, abs=1e-9)
        assert res["chr_weighted"] == pytest.approx(10 / 12, abs=1e-9)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            ClinicalHallucinationRate().calculate_chr([])

    def test_weight_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            ClinicalHallucinationRate().calculate_chr([0.1, 0.2], weights=[1.0])


class TestInstructionalVulnerabilityIndex:
    def test_flip_rate(self):
        neutral = ["acs", "non_cardiac", "acs", "other"]
        biased = ["acs", "acs", "acs", "non_cardiac"]
        res = InstructionalVulnerabilityIndex().calculate_ivi(neutral, biased)
        assert res["ivi_flip_rate"] == pytest.approx(0.5, abs=1e-9)
        assert res["ivi_effect"] is None  # non-numeric labels

    def test_numeric_effect(self):
        res = InstructionalVulnerabilityIndex().calculate_ivi([0, 0, 0], [1, 1, 0])
        assert res["ivi_flip_rate"] == pytest.approx(2 / 3, abs=1e-9)
        assert res["ivi_effect"] == pytest.approx(2 / 3, abs=1e-9)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            InstructionalVulnerabilityIndex().calculate_ivi([1, 2], [1])


class TestGeographicRepresentationIndex:
    def test_known_gri(self):
        locs = ["US", "GB", "BR", "CN", "TZ", "US"]  # unique: 5
        west = ["US", "GB", "DE"]  # DE not present -> intersect = {US, GB}
        res = GeographicRepresentationIndex().calculate_gri(locs, west)
        assert res["n_locations"] == 5
        assert res["n_western"] == 2
        assert res["gri"] == pytest.approx(3 / 5, abs=1e-9)

    def test_all_western_gives_zero(self):
        res = GeographicRepresentationIndex().calculate_gri(["US", "GB"], ["US", "GB"])
        assert res["gri"] == pytest.approx(0.0, abs=1e-9)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            GeographicRepresentationIndex().calculate_gri([], ["US"])

    def test_geographic_bias_correlation(self):
        gri = [0.1, 0.3, 0.5, 0.7, 0.9]
        err = [0.5, 0.4, 0.3, 0.2, 0.1]  # perfectly negatively correlated
        res = GeographicRepresentationIndex().calculate_geographic_bias(gri, err)
        assert res["gb"] == pytest.approx(-1.0, abs=1e-6)
