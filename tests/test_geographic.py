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
