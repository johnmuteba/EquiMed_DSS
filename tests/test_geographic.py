"""Tests for geographic-equity metrics (BEMI, GCC)."""
import numpy as np
import pandas as pd
import pytest

from equimed_dss.geographic import (
    BurdenEvidenceMismatch,
    GeographicConcentration,
    WHO_REGION_IHD_BURDEN,
)


class TestBurdenEvidenceMismatch:
    def test_identical_distributions_give_zero(self):
        bemi = BurdenEvidenceMismatch()
        ev = {"AFRO": 0.25, "EURO": 0.25, "AMRO": 0.25, "WPRO": 0.25}
        res = bemi.calculate_bemi(evidence_counts=ev, burden_shares=ev)
        assert res["bemi"] == pytest.approx(0.0, abs=1e-9)

    def test_disjoint_support_gives_one(self):
        bemi = BurdenEvidenceMismatch()
        res = bemi.calculate_bemi(
            evidence_counts={"AFRO": 1.0, "EURO": 0.0},
            burden_shares={"AFRO": 0.0, "EURO": 1.0},
        )
        assert res["bemi"] == pytest.approx(1.0, abs=1e-9)

    def test_known_mismatch_value(self):
        bemi = BurdenEvidenceMismatch()
        res = bemi.calculate_bemi(
            evidence_counts={"AFRO": 0.0, "SEARO": 0.0, "EURO": 0.5, "AMRO": 0.5},
            burden_shares={"AFRO": 0.18, "SEARO": 0.18, "EURO": 0.32, "AMRO": 0.32},
        )
        assert res["bemi"] == pytest.approx(0.36, abs=1e-9)
        assert isinstance(res["per_region"], pd.DataFrame)
        assert isinstance(res["evidence_shares"], dict)
        assert isinstance(res["interpretation"], str)

    def test_empty_evidence_raises(self):
        bemi = BurdenEvidenceMismatch()
        with pytest.raises(ValueError):
            bemi.calculate_bemi(evidence_counts={}, burden_shares={"AFRO": 1.0})

    def test_zero_total_raises(self):
        bemi = BurdenEvidenceMismatch()
        with pytest.raises(ValueError):
            bemi.calculate_bemi(
                evidence_counts={"AFRO": 0.0, "EURO": 0.0},
                burden_shares={"AFRO": 1.0, "EURO": 0.0},
            )


class TestGeographicConcentration:
    def test_uniform_low_gini_high_entropy(self):
        gcc = GeographicConcentration()
        res = gcc.calculate_gcc({f"R{i}": 1.0 for i in range(6)})
        assert res["gini_corrected"] == pytest.approx(0.0, abs=1e-9)
        assert res["entropy_normalized"] == pytest.approx(1.0, abs=1e-9)
        assert res["concentration"] == pytest.approx(0.0, abs=1e-9)

    def test_single_region_gini_one_entropy_zero(self):
        gcc = GeographicConcentration()
        ev = {f"R{i}": (1.0 if i == 0 else 0.0) for i in range(6)}
        res = gcc.calculate_gcc(ev)
        assert res["gini_corrected"] == pytest.approx(1.0, abs=1e-9)
        assert res["entropy_normalized"] == pytest.approx(0.0, abs=1e-9)

    def test_per_region_sorted_descending(self):
        gcc = GeographicConcentration()
        res = gcc.calculate_gcc({"A": 1.0, "B": 3.0, "C": 2.0})
        shares = res["per_region"]["evidence_share"].tolist()
        assert shares == sorted(shares, reverse=True)

    def test_needs_two_regions(self):
        gcc = GeographicConcentration()
        with pytest.raises(ValueError):
            gcc.calculate_gcc({"A": 1.0})

    def test_negative_value_raises(self):
        gcc = GeographicConcentration()
        with pytest.raises(ValueError):
            gcc.calculate_gcc({"A": -1.0, "B": 2.0})


class TestReferenceData:
    def test_reference_sums_to_one(self):
        assert sum(WHO_REGION_IHD_BURDEN.values()) == pytest.approx(1.0, abs=1e-6)

    def test_count_shares_match_who_ghe_2023(self):
        # WHO Global Health Estimates 2023, IHD DALYs by WHO region, 2023
        # (thousands): shares of the regional total, to 4 decimals.
        expected = {"AFRO": 0.0552, "AMRO": 0.1147, "EMRO": 0.0969,
                    "EURO": 0.1789, "SEARO": 0.2478, "WPRO": 0.3065}
        for region, share in expected.items():
            assert WHO_REGION_IHD_BURDEN[region] == pytest.approx(share, abs=5e-5)

    def test_rate_shares_match_who_ghe_2023(self):
        from equimed_dss import WHO_REGION_IHD_BURDEN_RATE
        expected = {"AFRO": 0.0604, "AMRO": 0.1493, "EMRO": 0.1619,
                    "EURO": 0.2579, "SEARO": 0.1846, "WPRO": 0.1858}
        assert sum(WHO_REGION_IHD_BURDEN_RATE.values()) == pytest.approx(1.0, abs=1e-9)
        for region, share in expected.items():
            assert WHO_REGION_IHD_BURDEN_RATE[region] == pytest.approx(share, abs=5e-5)

    def test_region_codes_map_gho_to_bundled_keys(self):
        from equimed_dss import WHO_REGION_CODES
        assert sorted(WHO_REGION_CODES.values()) == sorted(WHO_REGION_IHD_BURDEN)

    def test_reference_usable_as_burden(self):
        bemi = BurdenEvidenceMismatch()
        ev = {k: (0.0 if k in ("AFRO", "SEARO") else 1.0) for k in WHO_REGION_IHD_BURDEN}
        res = bemi.calculate_bemi(evidence_counts=ev, burden_shares=WHO_REGION_IHD_BURDEN)
        assert 0.0 < res["bemi"] <= 1.0
        assert res["most_underserved_region"] in ("AFRO", "SEARO")
