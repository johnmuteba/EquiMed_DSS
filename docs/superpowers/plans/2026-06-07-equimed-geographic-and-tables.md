# EquiMed-DSS v1.1.0: Geographic Metrics + Statistical Tables Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a geographic-equity metric module (BEMI + GCC) and a reporting layer that turns existing statistics results into publication-ready tables, then ship as v1.1.0.

**Architecture:** Two new additive subpackages, `equimed_dss/geographic/` (two metric classes following the existing `calculate_*`-returns-dict pattern) and `equimed_dss/reporting/` (pure presentation: dict -> tidy DataFrame -> exporter). No existing public signatures change. Spec: `docs/superpowers/specs/2026-06-07-equimed-geographic-and-tables-design.md`.

**Tech Stack:** Python 3.10+, numpy, pandas, scipy (existing); add `tabulate` (for `DataFrame.to_markdown`). Tests via pytest. Run all Python with `/usr/bin/python3` on the login node (the `~/env` venv has no pandas).

**Conventions to follow (from existing code):**
- Metric classes use `def __init__(self): pass` and `calculate_<name>(self, ...) -> Dict[str, Any]`, returning a dict that includes an `"interpretation"` sub-dict.
- Tests are class-based (`class TestX:` with `test_*` methods); use `np.random.RandomState(seed)`, never `np.random.seed`.
- `pyproject.toml` has `testpaths = ["tests"]`, `addopts = "-v --tb=short"`.

**Run tests with:** `/usr/bin/python3 -m pytest <path> -v`

---

### Task 1: BEMI — Burden-Evidence Mismatch metric

**Files:**
- Create: `equimed_dss/geographic/__init__.py`
- Create: `equimed_dss/geographic/burden_evidence.py`
- Test: `tests/test_geographic.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_geographic.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/usr/bin/python3 -m pytest tests/test_geographic.py -v`
Expected: FAIL with `ModuleNotFoundError` / `ImportError: cannot import name 'BurdenEvidenceMismatch'`.

- [ ] **Step 3: Write minimal implementation**

Create `equimed_dss/geographic/burden_evidence.py`:

```python
"""Burden-Evidence Mismatch Index (BEMI).

BEMI is the total-variation distance between a corpus's geographic *evidence*
distribution and a disease-*burden* distribution over the same regions:

    BEMI = 0.5 * sum_r |evidence_share_r - burden_share_r|,  range [0, 1].

0 means evidence tracks burden perfectly; 1 means disjoint support. It equals
the fraction of evidence mass that would need geographic reallocation to match
burden. Bounds proven and verified 2026-06-07.
"""
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


class BurdenEvidenceMismatch:
    """Geographic Burden-Evidence Mismatch Index (BEMI)."""

    def __init__(self, burden_reference: Optional[Dict[str, float]] = None):
        # Optional default burden distribution; can also be passed per call.
        self.burden_reference = burden_reference

    def calculate_bemi(
        self,
        evidence: Dict[str, float],
        burden: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Compute BEMI between an evidence distribution and a burden reference.

        Args:
            evidence: region -> evidence count or share.
            burden: region -> burden count or share. If None, uses the
                reference passed at construction.

        Returns:
            Dict with bemi_index (float), per_region (DataFrame),
            most_underserved_region (str), and interpretation.
        """
        burden = burden if burden is not None else self.burden_reference
        if burden is None:
            raise ValueError(
                "A burden reference must be provided (per call or at construction)."
            )

        regions = sorted(set(evidence) | set(burden))
        a = np.array([float(evidence.get(r, 0.0)) for r in regions])
        b = np.array([float(burden.get(r, 0.0)) for r in regions])
        if a.sum() <= 0 or b.sum() <= 0:
            raise ValueError("Evidence and burden must each have a positive total.")

        a = a / a.sum()
        b = b / b.sum()
        mismatch = a - b
        ratio = np.divide(a, b, out=np.full_like(a, np.nan), where=b > 0)
        bemi_index = float(0.5 * np.abs(mismatch).sum())

        table = pd.DataFrame(
            {
                "region": regions,
                "evidence_share": a,
                "burden_share": b,
                "mismatch": mismatch,
                "ratio": ratio,
            }
        )
        underserved = str(table.loc[table["mismatch"].idxmin(), "region"])

        return {
            "bemi_index": bemi_index,
            "per_region": table,
            "most_underserved_region": underserved,
            "interpretation": {
                "range": "BEMI in [0, 1]; 0 = evidence tracks burden, 1 = disjoint",
                "index_meaning": (
                    f"{bemi_index * 100:.1f}% of evidence mass would need "
                    "geographic reallocation to match burden"
                ),
            },
        }
```

Create `equimed_dss/geographic/__init__.py`:

```python
"""Geographic-equity metrics for EquiMed-DSS."""
from .burden_evidence import BurdenEvidenceMismatch

__all__ = ["BurdenEvidenceMismatch"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/usr/bin/python3 -m pytest tests/test_geographic.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add equimed_dss/geographic/__init__.py equimed_dss/geographic/burden_evidence.py tests/test_geographic.py
git commit -m "feat(geographic): add Burden-Evidence Mismatch Index (BEMI)"
```

---

### Task 2: GCC — Geographic Concentration of Coverage

**Files:**
- Create: `equimed_dss/geographic/concentration.py`
- Modify: `equimed_dss/geographic/__init__.py`
- Test: `tests/test_geographic.py` (append a test class)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_geographic.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/usr/bin/python3 -m pytest tests/test_geographic.py::TestGeographicConcentration -v`
Expected: FAIL with `ImportError: cannot import name 'GeographicConcentration'`.

- [ ] **Step 3: Write minimal implementation**

Create `equimed_dss/geographic/concentration.py`:

```python
"""Geographic Concentration of Coverage (GCC).

Two complementary descriptors of how a corpus's evidence is spread across
regions:

- Gini (sample-corrected): G* = (R / (R-1)) * G_raw, range [0, 1].
  0 = perfectly even, 1 = all evidence in one region. The R/(R-1) factor is
  required because raw Gini for R categories maxes out at (R-1)/R.
- Normalized Shannon entropy: H_norm = -sum_r p_r ln p_r / ln R, range [0, 1].
  1 = even coverage, 0 = single-region concentration.

Gini and entropy run in opposite directions; `concentration = 1 - H_norm` is
exposed so a single "higher = more concentrated" reading is available.
Verified 2026-06-07.
"""
from typing import Any, Dict

import numpy as np
import pandas as pd


class GeographicConcentration:
    """Geographic Concentration of Coverage (GCC)."""

    def __init__(self):
        pass

    def calculate_gcc(self, evidence: Dict[str, float]) -> Dict[str, Any]:
        """Compute concentration descriptors for a geographic evidence vector.

        Args:
            evidence: region -> evidence count or share (non-negative).

        Returns:
            Dict with gini, normalized_entropy, concentration, n_regions,
            per_region (DataFrame), and interpretation.
        """
        regions = sorted(evidence)
        x = np.array([float(evidence[r]) for r in regions])
        if np.any(x < 0):
            raise ValueError("Evidence values must be non-negative.")
        if x.sum() <= 0:
            raise ValueError("Evidence total must be positive.")
        R = len(x)
        if R < 2:
            raise ValueError("Need at least 2 regions to measure concentration.")

        gini_raw = np.abs(x[:, None] - x[None, :]).sum() / (2 * R * x.sum())
        gini = float(gini_raw * R / (R - 1))

        p = x / x.sum()
        nz = p[p > 0]
        entropy = float(-(nz * np.log(nz)).sum() / np.log(R))
        concentration = float(1.0 - entropy)

        table = (
            pd.DataFrame({"region": regions, "evidence_share": p})
            .sort_values("evidence_share", ascending=False)
            .reset_index(drop=True)
        )

        return {
            "gini": gini,
            "normalized_entropy": entropy,
            "concentration": concentration,
            "n_regions": R,
            "per_region": table,
            "interpretation": {
                "gini": "0 = even coverage, 1 = single-region (sample-corrected)",
                "entropy": "1 = even coverage, 0 = single-region concentration",
            },
        }
```

Update `equimed_dss/geographic/__init__.py`:

```python
"""Geographic-equity metrics for EquiMed-DSS."""
from .burden_evidence import BurdenEvidenceMismatch
from .concentration import GeographicConcentration

__all__ = ["BurdenEvidenceMismatch", "GeographicConcentration"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/usr/bin/python3 -m pytest tests/test_geographic.py -v`
Expected: all geographic tests pass (10 total).

- [ ] **Step 5: Commit**

```bash
git add equimed_dss/geographic/concentration.py equimed_dss/geographic/__init__.py tests/test_geographic.py
git commit -m "feat(geographic): add Geographic Concentration of Coverage (GCC)"
```

---

### Task 3: Burden reference constant (real numbers, no fabrication)

**Files:**
- Create: `equimed_dss/geographic/reference_data.py`
- Modify: `equimed_dss/geographic/__init__.py`
- Test: `tests/test_geographic.py` (append)

**IMPORTANT (manuscript-integrity rule):** Do NOT invent precise GBD numbers and present them as authoritative. Before writing the constant, source real per-WHO-region IHD burden shares from the user's own pipeline output. Check, in order:
1. `ls /mnt/datalab-hpc2-slurm/home/jmwamba/tri_corpora_env/result -R | grep -i -E "geograph|burden|distinctive"` and read any region->burden file the `geography_distinctive_layer.py` step produced.
2. If a real share table is found, transcribe those values verbatim into the constant and cite the source file in the docstring.
3. If none is found, define the constant as clearly-labelled ILLUSTRATIVE values (name it with an `EXAMPLE_` prefix, docstring says "illustrative; replace with sourced GBD/pipeline values before any published analysis") and leave a NOTE in CHANGELOG so the user replaces it.

- [ ] **Step 1: Source the real values**

Run: `grep -rniE "AFRO|SEARO|burden|region" /mnt/datalab-hpc2-slurm/home/jmwamba/tri_corpora_env/geography_distinctive_layer.py | head -40`
and inspect `result/` for a produced burden table. Record the real shares (they should reflect the manuscript's "AFRO+SEARO ~36% IHD burden" finding).

- [ ] **Step 2: Write the failing test**

Append to `tests/test_geographic.py`:

```python
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
```

- [ ] **Step 3: Run test to verify it fails**

Run: `/usr/bin/python3 -m pytest tests/test_geographic.py::TestReferenceData -v`
Expected: FAIL with `ImportError: cannot import name 'WHO_REGION_IHD_BURDEN'`.

- [ ] **Step 4: Write the constant**

REAL values sourced (2026-06-07) from the tri-corpora pipeline's
`geography_distinctive_layer.py` (`GBD_IHD_DALY_PER_100K_BY_REGION`, Roth GA
et al., 2020 GBD Compare for IHD). Normalized shares give AFRO+SEARO = 0.361,
matching the manuscript's geographic-gap finding. Store the raw DALY values
(auditable, identical to the pipeline) and derive the shares from them.

Create `equimed_dss/geographic/reference_data.py`:

```python
"""Reference disease-burden distributions for geographic metrics.

GBD_IHD_DALY_PER_100K_BY_REGION: age-standardized ischaemic heart disease
(IHD) DALYs per 100,000 by WHO region. Source: Roth GA et al., 2020 (GBD
Compare for IHD), values rounded to the nearest 100. Aggregate published
statistics (not patient-level), DUA-safe to bundle. This is the same reference
used by the tri-corpora pipeline's geography_distinctive_layer step.

WHO_REGION_IHD_BURDEN: the above normalized to shares summing to 1. AFRO and
SEARO together carry ~36% of global IHD burden, matching the manuscript's
geographic-gap finding.
"""

GBD_IHD_DALY_PER_100K_BY_REGION = {
    "AFRO": 2730,
    "AMRO": 2070,
    "EMRO": 4200,
    "EURO": 3550,
    "SEARO": 3850,
    "WPRO": 1830,
}

_TOTAL = sum(GBD_IHD_DALY_PER_100K_BY_REGION.values())
WHO_REGION_IHD_BURDEN = {
    region: dalys / _TOTAL
    for region, dalys in GBD_IHD_DALY_PER_100K_BY_REGION.items()
}
```

Update `equimed_dss/geographic/__init__.py`:

```python
"""Geographic-equity metrics for EquiMed-DSS."""
from .burden_evidence import BurdenEvidenceMismatch
from .concentration import GeographicConcentration
from .reference_data import WHO_REGION_IHD_BURDEN

__all__ = [
    "BurdenEvidenceMismatch",
    "GeographicConcentration",
    "WHO_REGION_IHD_BURDEN",
]
```

- [ ] **Step 5: Run test + commit**

Run: `/usr/bin/python3 -m pytest tests/test_geographic.py -v`
Expected: all pass.

```bash
git add equimed_dss/geographic/reference_data.py equimed_dss/geographic/__init__.py tests/test_geographic.py
git commit -m "feat(geographic): add WHO-region IHD burden reference constant"
```

---

### Task 4: Reporting tables (dict -> tidy DataFrame)

**Files:**
- Create: `equimed_dss/reporting/__init__.py`
- Create: `equimed_dss/reporting/tables.py`
- Test: `tests/test_reporting.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_reporting.py`:

```python
"""Tests for the reporting/table layer."""
import numpy as np
import pandas as pd
import pytest

from equimed_dss.reporting import (
    hierarchical_coefficients_table,
    mediation_effects_table,
    network_centrality_table,
    geographic_table,
)


def test_hierarchical_table_shape():
    results = {
        "icc": 0.12, "variance_between_groups": 0.5, "variance_within_groups": 3.5,
        "total_variance": 4.0, "r_squared_marginal": 0.05, "aic": 100.0,
        "bic": 110.0, "n_groups": 10, "n_observations": 200,
    }
    df = hierarchical_coefficients_table(results)
    assert list(df.columns) == ["term", "value"]
    assert "ICC" in set(df["term"])
    assert df.loc[df["term"] == "ICC", "value"].iloc[0] == 0.12


def test_mediation_table_rows_and_flag():
    results = {
        "total_effect": 1.0, "direct_effect": 0.4, "indirect_effect": 0.6,
        "proportion_mediated": 0.6, "indirect_ci_lower": 0.2,
        "indirect_ci_upper": 0.9,
        "interpretation": {"mediation_type": "Partial mediation (complementary)"},
    }
    df = mediation_effects_table(results)
    assert list(df["effect"]) == ["total", "direct", "indirect"]
    assert df.loc[df["effect"] == "indirect", "proportion_mediated"].iloc[0] == 0.6
    assert df.attrs["proportion_mediated_flag"] == ""


def test_mediation_table_flags_out_of_range_pm():
    results = {
        "total_effect": 0.1, "direct_effect": -0.5, "indirect_effect": 0.6,
        "proportion_mediated": 6.0, "indirect_ci_lower": 0.2,
        "indirect_ci_upper": 0.9,
        "interpretation": {"mediation_type": "Partial mediation (competitive)"},
    }
    df = mediation_effects_table(results)
    assert "out-of-range" in df.attrs["proportion_mediated_flag"]


def test_network_table_one_row_per_node():
    results = {
        "degree_centrality": {"a": 1.0, "b": 0.5},
        "betweenness_centrality": {"a": 0.0, "b": 0.0},
        "closeness_centrality": {"a": 1.0, "b": 0.66},
        "clustering_coefficients": {"a": 0.0, "b": 0.0},
    }
    df = network_centrality_table(results)
    assert list(df.columns) == ["node", "degree", "betweenness", "closeness", "clustering"]
    assert len(df) == 2


def test_geographic_table_passthrough():
    per_region = pd.DataFrame({"region": ["A", "B"], "evidence_share": [0.6, 0.4]})
    df = geographic_table({"per_region": per_region})
    assert df.equals(per_region)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/usr/bin/python3 -m pytest tests/test_reporting.py -v`
Expected: FAIL with `ModuleNotFoundError: equimed_dss.reporting`.

- [ ] **Step 3: Write minimal implementation**

Create `equimed_dss/reporting/tables.py`:

```python
"""Convert EquiMed-DSS statistics result dicts into tidy DataFrames.

Pure presentation layer: these functions reshape the dicts already returned by
`equimed_dss.statistics` and `equimed_dss.geographic`; they do not recompute
any statistic.
"""
from typing import Any, Dict

import pandas as pd


def hierarchical_coefficients_table(results: Dict[str, Any]) -> pd.DataFrame:
    """Variance components and fit statistics from HierarchicalLinearModeling."""
    rows = [
        ("ICC", results.get("icc")),
        ("variance_between_groups", results.get("variance_between_groups")),
        ("variance_within_groups", results.get("variance_within_groups")),
        ("total_variance", results.get("total_variance")),
        ("r_squared_marginal", results.get("r_squared_marginal")),
        ("AIC", results.get("aic")),
        ("BIC", results.get("bic")),
        ("n_groups", results.get("n_groups")),
        ("n_observations", results.get("n_observations")),
    ]
    return pd.DataFrame(rows, columns=["term", "value"])


def mediation_effects_table(results: Dict[str, Any]) -> pd.DataFrame:
    """Total/direct/indirect effects from MediationAnalysis.

    proportion_mediated is reported as-is; a flag in df.attrs marks the case
    where it falls outside [0, 1] (competitive/unstable mediation).
    """
    pm = results.get("proportion_mediated", float("nan"))
    mtype = results.get("interpretation", {}).get("mediation_type", "")
    rows = [
        ("total", results.get("total_effect"), None, None, None, ""),
        ("direct", results.get("direct_effect"), None, None, None, ""),
        (
            "indirect",
            results.get("indirect_effect"),
            results.get("indirect_ci_lower"),
            results.get("indirect_ci_upper"),
            pm,
            mtype,
        ),
    ]
    df = pd.DataFrame(
        rows,
        columns=[
            "effect", "estimate", "ci_lower", "ci_upper",
            "proportion_mediated", "classification",
        ],
    )
    in_range = isinstance(pm, (int, float)) and 0.0 <= pm <= 1.0
    df.attrs["proportion_mediated_flag"] = (
        "" if in_range else "out-of-range (competitive/unstable mediation)"
    )
    return df


def network_centrality_table(results: Dict[str, Any]) -> pd.DataFrame:
    """One row per node with degree/betweenness/closeness/clustering."""
    deg = results.get("degree_centrality", {})
    bet = results.get("betweenness_centrality", {})
    clo = results.get("closeness_centrality", {})
    clu = results.get("clustering_coefficients", {})
    nodes = sorted(deg)
    return pd.DataFrame(
        {
            "node": nodes,
            "degree": [deg.get(n) for n in nodes],
            "betweenness": [bet.get(n) for n in nodes],
            "closeness": [clo.get(n) for n in nodes],
            "clustering": [clu.get(n) for n in nodes],
        }
    )


def geographic_table(result: Dict[str, Any]) -> pd.DataFrame:
    """Return the per-region table from a BEMI or GCC result."""
    return result["per_region"].copy()
```

Create `equimed_dss/reporting/__init__.py`:

```python
"""Reporting/table layer for EquiMed-DSS."""
from .tables import (
    geographic_table,
    hierarchical_coefficients_table,
    mediation_effects_table,
    network_centrality_table,
)

__all__ = [
    "hierarchical_coefficients_table",
    "mediation_effects_table",
    "network_centrality_table",
    "geographic_table",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/usr/bin/python3 -m pytest tests/test_reporting.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add equimed_dss/reporting/__init__.py equimed_dss/reporting/tables.py tests/test_reporting.py
git commit -m "feat(reporting): add tidy-DataFrame tables for stats results"
```

---

### Task 5: export_table helper + tabulate dependency

**Files:**
- Create: `equimed_dss/reporting/export.py`
- Modify: `equimed_dss/reporting/__init__.py`
- Modify: `pyproject.toml` (add `tabulate` dependency)
- Test: `tests/test_reporting.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_reporting.py`:

```python
from equimed_dss.reporting import export_table


def test_export_markdown_nonempty():
    df = pd.DataFrame({"term": ["ICC"], "value": [0.123456]})
    out = export_table(df, fmt="markdown", decimals=3)
    assert "ICC" in out
    assert "0.123" in out


def test_export_latex_nonempty():
    df = pd.DataFrame({"term": ["ICC"], "value": [0.123456]})
    out = export_table(df, fmt="latex", decimals=2)
    assert "tabular" in out
    assert "0.12" in out


def test_export_writes_file(tmp_path):
    df = pd.DataFrame({"term": ["ICC"], "value": [0.5]})
    p = tmp_path / "t.md"
    export_table(df, fmt="markdown", path=str(p))
    assert p.read_text().strip() != ""


def test_export_bad_format_raises():
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(ValueError):
        export_table(df, fmt="csv")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/usr/bin/python3 -m pytest tests/test_reporting.py -k export -v`
Expected: FAIL with `ImportError: cannot import name 'export_table'`.

- [ ] **Step 3: Write minimal implementation**

Create `equimed_dss/reporting/export.py`:

```python
"""Render a DataFrame to markdown / LaTeX / HTML with consistent rounding."""
from typing import Optional

import pandas as pd

_FORMATS = ("markdown", "latex", "html")


def export_table(
    df: pd.DataFrame,
    fmt: str = "markdown",
    path: Optional[str] = None,
    decimals: int = 3,
) -> str:
    """Render a table for slides or the manuscript.

    Args:
        df: any tidy DataFrame (e.g. from equimed_dss.reporting.tables).
        fmt: one of "markdown", "latex", "html".
        path: if given, also write the rendered string to this path.
        decimals: rounding applied to numeric columns before rendering.

    Returns:
        The rendered table as a string.
    """
    if fmt not in _FORMATS:
        raise ValueError(f"Unknown fmt {fmt!r}; use one of {_FORMATS}.")

    rounded = df.copy()
    num_cols = rounded.select_dtypes(include="number").columns
    rounded[num_cols] = rounded[num_cols].round(decimals)

    if fmt == "markdown":
        rendered = rounded.to_markdown(index=False)
    elif fmt == "latex":
        rendered = rounded.to_latex(index=False)
    else:  # html
        rendered = rounded.to_html(index=False)

    if path is not None:
        with open(path, "w") as fh:
            fh.write(rendered)
    return rendered
```

Update `equimed_dss/reporting/__init__.py` to also export `export_table`:

```python
"""Reporting/table layer for EquiMed-DSS."""
from .export import export_table
from .tables import (
    geographic_table,
    hierarchical_coefficients_table,
    mediation_effects_table,
    network_centrality_table,
)

__all__ = [
    "hierarchical_coefficients_table",
    "mediation_effects_table",
    "network_centrality_table",
    "geographic_table",
    "export_table",
]
```

- [ ] **Step 4: Add tabulate dependency**

In `pyproject.toml`, find the `dependencies = [` list (project deps, near the top, includes numpy/pandas/scipy) and add:

```toml
    "tabulate>=0.9.0",
```

Then install it for the test run: `/usr/bin/python3 -m pip install --user --break-system-packages "tabulate>=0.9.0"`

- [ ] **Step 5: Run test to verify it passes + commit**

Run: `/usr/bin/python3 -m pytest tests/test_reporting.py -v`
Expected: all pass (9 total).

```bash
git add equimed_dss/reporting/export.py equimed_dss/reporting/__init__.py pyproject.toml tests/test_reporting.py
git commit -m "feat(reporting): add export_table (markdown/latex/html) + tabulate dep"
```

---

### Task 6: Top-level re-exports

**Files:**
- Modify: `equimed_dss/__init__.py`
- Test: `tests/test_basic.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_basic.py`:

```python
def test_top_level_geographic_and_reporting_imports():
    import equimed_dss
    from equimed_dss import (
        BurdenEvidenceMismatch,
        GeographicConcentration,
        export_table,
    )
    assert equimed_dss.BurdenEvidenceMismatch is BurdenEvidenceMismatch
    assert callable(export_table)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/usr/bin/python3 -m pytest tests/test_basic.py::test_top_level_geographic_and_reporting_imports -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Add re-exports**

In `equimed_dss/__init__.py`, after the `from .__version__ import (...)` block, add:

```python
from .geographic import (
    BurdenEvidenceMismatch,
    GeographicConcentration,
    WHO_REGION_IHD_BURDEN,
)
from .reporting import (
    export_table,
    geographic_table,
    hierarchical_coefficients_table,
    mediation_effects_table,
    network_centrality_table,
)
```

and add these names to the existing `__all__` list:

```python
    "BurdenEvidenceMismatch",
    "GeographicConcentration",
    "WHO_REGION_IHD_BURDEN",
    "export_table",
    "geographic_table",
    "hierarchical_coefficients_table",
    "mediation_effects_table",
    "network_centrality_table",
```

- [ ] **Step 4: Run test to verify it passes + commit**

Run: `/usr/bin/python3 -m pytest tests/test_basic.py -v`
Expected: pass.

```bash
git add equimed_dss/__init__.py tests/test_basic.py
git commit -m "feat: re-export geographic + reporting at top level"
```

---

### Task 7: More example scripts

**Files:**
- Create: `examples/example_geographic.py`
- Create: `examples/example_statistics_tables.py`

- [ ] **Step 1: Write the geographic example**

Create `examples/example_geographic.py`:

```python
"""Geographic-equity metrics demo (BEMI + GCC).

Uses illustrative sample shares. REAL-DATA HOOK: to render the actual
manuscript numbers on your machine, replace `evidence` below with your
corpus's per-WHO-region evidence counts (e.g. loaded from your
geography_distinctive_layer output) and pass your sourced burden reference.
"""
from equimed_dss.geographic import (
    BurdenEvidenceMismatch,
    GeographicConcentration,
    WHO_REGION_IHD_BURDEN,
)
from equimed_dss.reporting import export_table, geographic_table


def main():
    # Illustrative evidence distribution (replace via the real-data hook).
    evidence = {
        "AFRO": 1, "SEARO": 2, "EURO": 120, "AMRO": 90, "EMRO": 8, "WPRO": 15,
    }

    bemi = BurdenEvidenceMismatch(burden_reference=WHO_REGION_IHD_BURDEN)
    bemi_res = bemi.calculate_bemi(evidence)
    print(f"BEMI index: {bemi_res['bemi_index']:.3f}")
    print(f"Most underserved region: {bemi_res['most_underserved_region']}")
    print(export_table(geographic_table(bemi_res), fmt="markdown"))

    gcc = GeographicConcentration()
    gcc_res = gcc.calculate_gcc(evidence)
    print(f"\nGini (sample-corrected): {gcc_res['gini']:.3f}")
    print(f"Normalized entropy: {gcc_res['normalized_entropy']:.3f}")
    print(f"Concentration (1 - entropy): {gcc_res['concentration']:.3f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `/usr/bin/python3 examples/example_geographic.py`
Expected: prints a BEMI index in [0,1], a markdown table, and GCC values. No traceback.

- [ ] **Step 3: Write the statistics-tables example**

Create `examples/example_statistics_tables.py`:

```python
"""Render hierarchical / mediation / network results as publication tables.

Uses bundled illustrative sample data. REAL-DATA HOOK: replace the sample
DataFrames with your real result frames (or load a saved result dict) to render
the actual manuscript tables for slides.
"""
import numpy as np
import pandas as pd

from equimed_dss.statistics import (
    HierarchicalLinearModeling,
    MediationAnalysis,
    NetworkStatistics,
)
from equimed_dss.reporting import (
    export_table,
    hierarchical_coefficients_table,
    mediation_effects_table,
    network_centrality_table,
)


def _sample_hierarchical_df(rng):
    rows = []
    for g in range(8):
        group_effect = rng.normal(0, 2)
        for _ in range(25):
            x = rng.normal(0, 1)
            rows.append(
                {"group": g, "x": x, "outcome": group_effect + 0.5 * x + rng.normal(0, 1)}
            )
    return pd.DataFrame(rows)


def main():
    rng = np.random.RandomState(42)

    # --- Hierarchical ---
    # fit_model returns the dict (icc, variance components, aic, bic); note
    # calculate_icc returns a bare float, so use fit_model here.
    hlm = HierarchicalLinearModeling()
    df = _sample_hierarchical_df(rng)
    hlm_res = hlm.fit_model(
        df, outcome_var="outcome", level1_predictors=["x"], level2_var="group"
    )
    print("Hierarchical (variance components):")
    print(export_table(hierarchical_coefficients_table(hlm_res), fmt="markdown"))

    # --- Mediation ---  (arg is treatment_var, not independent_var)
    n = 300
    x = rng.normal(0, 1, n)
    m = 0.5 * x + rng.normal(0, 1, n)
    y = 0.3 * x + 0.4 * m + rng.normal(0, 1, n)
    med = MediationAnalysis()
    med_res = med.analyze_mediation(
        pd.DataFrame({"X": x, "M": m, "Y": y}),
        treatment_var="X", mediator_var="M", outcome_var="Y",
    )
    print("\nMediation effects:")
    med_table = mediation_effects_table(med_res)
    print(export_table(med_table, fmt="markdown"))
    flag = med_table.attrs.get("proportion_mediated_flag", "")
    if flag:
        print(f"NOTE: {flag}")

    # --- Network ---  (takes a numpy adjacency matrix + node_labels)
    net = NetworkStatistics()
    labels = ["DFR", "ECS", "ICC", "HER"]
    adjacency = np.array(
        [
            [0.0, 0.6, 0.2, 0.0],
            [0.6, 0.0, 0.0, 0.5],
            [0.2, 0.0, 0.0, 0.0],
            [0.0, 0.5, 0.0, 0.0],
        ]
    )
    net_res = net.analyze_network(adjacency, node_labels=labels)
    print("\nNetwork centralities:")
    print(export_table(network_centrality_table(net_res), fmt="markdown"))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run it**

Run: `/usr/bin/python3 examples/example_statistics_tables.py`
Expected: three markdown tables print with no traceback. (Signatures verified against
`equimed_dss/statistics/*.py` 2026-06-07: `fit_model(data, outcome_var,
level1_predictors, level2_var)`, `analyze_mediation(data, treatment_var,
mediator_var, outcome_var)`, `analyze_network(adjacency_matrix, node_labels)`.)

- [ ] **Step 5: Commit**

```bash
git add examples/example_geographic.py examples/example_statistics_tables.py
git commit -m "docs(examples): add geographic + statistics-tables examples"
```

---

### Task 8: Version bump, docs, CHANGELOG

**Files:**
- Modify: `equimed_dss/__version__.py`
- Modify: `CHANGELOG.md`
- Modify: `README.md`
- Modify: `docs/METRICS_GUIDE.md`
- Modify: `docs/API_REFERENCE.md`

- [ ] **Step 1: Bump version**

In `equimed_dss/__version__.py`, change:

```python
__version__ = "1.0.2"
```

to:

```python
__version__ = "1.1.0"
```

- [ ] **Step 2: Add CHANGELOG entry**

At the top of `CHANGELOG.md` (under any header), add:

```markdown
## [1.1.0] - 2026-06-07

### Added
- `equimed_dss.geographic`: Burden-Evidence Mismatch Index (BEMI, the
  total-variation distance between evidence and disease-burden distributions)
  and Geographic Concentration of Coverage (GCC, sample-corrected Gini +
  normalized Shannon entropy). Bundled `WHO_REGION_IHD_BURDEN` reference.
- `equimed_dss.reporting`: tidy-DataFrame tables for hierarchical, mediation,
  and network results, plus `export_table` (markdown / LaTeX / HTML).
- Examples: `example_geographic.py`, `example_statistics_tables.py`.

### Notes
- `WHO_REGION_IHD_BURDEN` values must be confirmed against the sourced
  GBD / geography_distinctive_layer output before any published analysis.
- `proportion_mediated` in the mediation table is reported unclamped and
  flagged when it falls outside [0, 1] (competitive/unstable mediation).
```

- [ ] **Step 3: Update README feature list + METRICS_GUIDE + API_REFERENCE**

In `README.md`, add a bullet under the features/metrics list mentioning the geographic metrics (BEMI, GCC) and the reporting/table layer.

In `docs/METRICS_GUIDE.md`, add a "Geographic Equity" section documenting BEMI (formula `0.5*sum|a-b|`, range [0,1], total-variation interpretation) and GCC (sample-corrected Gini `R/(R-1)*G_raw`, normalized entropy `H/lnR`, opposite directions).

In `docs/API_REFERENCE.md`, add entries for `BurdenEvidenceMismatch.calculate_bemi`, `GeographicConcentration.calculate_gcc`, `WHO_REGION_IHD_BURDEN`, the four table functions, and `export_table`.

- [ ] **Step 4: No-em-dash check (user rule)**

Run: `grep -rnP "\x{2014}" CHANGELOG.md README.md docs/METRICS_GUIDE.md docs/API_REFERENCE.md docs/VIGNETTE.md`
Expected: no output. If any line matches, replace the em dash with a comma/colon/parenthesis.

- [ ] **Step 5: Commit**

```bash
git add equimed_dss/__version__.py CHANGELOG.md README.md docs/METRICS_GUIDE.md docs/API_REFERENCE.md
git commit -m "docs: v1.1.0 version bump, CHANGELOG, geographic + tables docs"
```

---

### Task 9: Full verification + build

**Files:** none (verification only)

- [ ] **Step 1: Run the full test suite**

Run: `/usr/bin/python3 -m pytest -v`
Expected: all tests pass (existing + new geographic/reporting/basic tests). If any pre-existing test fails for an unrelated reason, note it but do not fix outside scope.

- [ ] **Step 2: Build sdist + wheel**

Run:
```bash
/usr/bin/python3 -m pip install --user --break-system-packages build twine
/usr/bin/python3 -m build
```
Expected: `dist/equimed_dss-1.1.0.tar.gz` and `dist/equimed_dss-1.1.0-py3-none-any.whl` created.

- [ ] **Step 3: twine check**

Run: `/usr/bin/python3 -m twine check dist/equimed_dss-1.1.0*`
Expected: both artifacts `PASSED`.

- [ ] **Step 4: Smoke-import from the wheel**

Run:
```bash
/usr/bin/python3 -c "import equimed_dss; print(equimed_dss.__version__); from equimed_dss import BurdenEvidenceMismatch, GeographicConcentration, export_table; print('ok')"
```
Expected: prints `1.1.0` then `ok`.

- [ ] **Step 5: Commit any build config changes**

```bash
git add -A
git commit -m "chore: v1.1.0 build verification" || echo "nothing to commit"
```

---

### Task 10: Delivery (TestPyPI + GitHub) — needs user tokens

**Files:** none (release only). **Do NOT run until the user provides tokens at delivery time; handle them transiently, never write them to a file or commit them.**

- [ ] **Step 1: Upload 1.1.0 to TestPyPI**

With the TestPyPI token the user provides:
```bash
TWINE_USERNAME=__token__ TWINE_PASSWORD=<testpypi-token> \
  /usr/bin/python3 -m twine upload --repository-url https://test.pypi.org/legacy/ dist/equimed_dss-1.1.0*
```
Expected: upload succeeds; verify at `https://test.pypi.org/project/equimed-dss/1.1.0/`.

- [ ] **Step 2: Merge feature branch to master**

```bash
git checkout master
git merge --no-ff feature/v1.1.0-geographic-and-tables -m "Merge v1.1.0: geographic metrics + statistical tables"
```

- [ ] **Step 3: Push to GitHub with the user's PAT (transient)**

```bash
git push "https://<github-pat>@github.com/johnmuteba/EquiMed_DSS.git" master
```
Expected: master updated on GitHub. Then remind the user to revoke the PAT.

- [ ] **Step 4: Handoff**

Report: TestPyPI 1.1.0 live, GitHub master updated. User tests 1.1.0 locally from TestPyPI, then decides on real PyPI. Manuscript revision is the downstream step (separate work, governed by [[feedback-manuscript-integrity]]).

---

## Self-Review notes

- **Spec coverage:** geographic BEMI (T1) + GCC (T3 reference, T2 metric), reporting tables (T4) + exporters (T5), examples with real-data hook (T7), tests (T1-T6), docs + version (T8), verification (T9), TestPyPI + GitHub delivery (T10). All spec sections mapped.
- **Formula correctness:** BEMI bounds and GCC sample-corrected Gini encoded directly in tests (disjoint=1, single-region Gini=1), matching the verified spec.
- **Integrity:** Task 3 forbids inventing GBD numbers and routes to the user's real pipeline output; em-dash check in Task 8.
- **Type consistency:** table functions consume the exact dict keys returned by the existing statistics classes (`icc`, `total_effect`, `indirect_ci_lower`, `degree_centrality`, etc., confirmed against source) and the `per_region` key produced by BEMI/GCC.
- **Open confirmation:** Task 7 Step 4 flags that `calculate_icc` / `analyze_mediation` / `analyze_network` argument names must be confirmed against source while writing the example.
