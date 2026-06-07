"""Tests for geographic-equity metrics (BEMI, GCC)."""
import numpy as np
import pandas as pd
import pytest

from equimed_dss.geographic import BurdenEvidenceMismatch


class TestBurdenEvidenceMismatch:
    def test_identical_distributions_give_zero(self):
        bemi = BurdenEvidenceMismatch()
        ev = {"AFRO": 0.25, "EURO": 0.25, "AMRO": 0.25, "WPRO": 0.25}
        res = bemi.calculate_bemi(ev, burden=ev)
        assert res["bemi_index"] == pytest.approx(0.0, abs=1e-9)

    def test_disjoint_support_gives_one(self):
        bemi = BurdenEvidenceMismatch()
        ev = {"AFRO": 1.0, "EURO": 0.0}
        bu = {"AFRO": 0.0, "EURO": 1.0}
        res = bemi.calculate_bemi(ev, burden=bu)
        assert res["bemi_index"] == pytest.approx(1.0, abs=1e-9)

    def test_known_mismatch_value(self):
        # evidence concentrated in two regions; burden spread to four.
        bemi = BurdenEvidenceMismatch()
        ev = {"AFRO": 0.0, "SEARO": 0.0, "EURO": 0.5, "AMRO": 0.5}
        bu = {"AFRO": 0.18, "SEARO": 0.18, "EURO": 0.32, "AMRO": 0.32}
        res = bemi.calculate_bemi(ev, burden=bu)
        assert res["bemi_index"] == pytest.approx(0.36, abs=1e-9)
        assert isinstance(res["per_region"], pd.DataFrame)
        assert set(res["per_region"]["region"]) == {"AFRO", "SEARO", "EURO", "AMRO"}

    def test_missing_reference_raises(self):
        bemi = BurdenEvidenceMismatch()
        with pytest.raises(ValueError):
            bemi.calculate_bemi({"AFRO": 1.0})

    def test_zero_total_raises(self):
        bemi = BurdenEvidenceMismatch()
        with pytest.raises(ValueError):
            bemi.calculate_bemi({"AFRO": 0.0, "EURO": 0.0}, burden={"AFRO": 1.0, "EURO": 0.0})


from equimed_dss.geographic import GeographicConcentration


class TestGeographicConcentration:
    def test_uniform_low_gini_high_entropy(self):
        gcc = GeographicConcentration()
        res = gcc.calculate_gcc({f"R{i}": 1.0 for i in range(6)})
        assert res["gini"] == pytest.approx(0.0, abs=1e-9)
        assert res["normalized_entropy"] == pytest.approx(1.0, abs=1e-9)
        assert res["concentration"] == pytest.approx(0.0, abs=1e-9)

    def test_single_region_gini_one_entropy_zero(self):
        # The sample correction R/(R-1) is required for gini == 1 here.
        gcc = GeographicConcentration()
        ev = {f"R{i}": (1.0 if i == 0 else 0.0) for i in range(6)}
        res = gcc.calculate_gcc(ev)
        assert res["gini"] == pytest.approx(1.0, abs=1e-9)
        assert res["normalized_entropy"] == pytest.approx(0.0, abs=1e-9)

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


from equimed_dss.geographic import WHO_REGION_IHD_BURDEN


class TestReferenceData:
    def test_reference_sums_to_one(self):
        total = sum(WHO_REGION_IHD_BURDEN.values())
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_reference_usable_as_default(self):
        bemi = BurdenEvidenceMismatch(burden_reference=WHO_REGION_IHD_BURDEN)
        # evidence with nothing in AFRO/SEARO -> positive mismatch there
        ev = {k: (0.0 if k in ("AFRO", "SEARO") else 1.0) for k in WHO_REGION_IHD_BURDEN}
        res = bemi.calculate_bemi(ev)
        assert 0.0 < res["bemi_index"] <= 1.0
        assert res["most_underserved_region"] in ("AFRO", "SEARO")
