#!/usr/bin/env python3
"""
regression_bugfix_report.py
===========================
Produces a professionally formatted report (printed and written to
docs/REGRESSION_BUGFIX_REPORT.md) containing:

  1. A bug-fix verification table for the four v1.5.0 correctness fixes
     (DFR Wilson CI, JSD consistency, HAFG normalization, sample-SD / no-mutation),
     each with the recomputed statistic.
  2. A hierarchical (mixed-effects) regression COEFFICIENT table
     (term, estimate, SE, t, p, 95% CI) rendered by the reporting layer.
  3. A mediation effects table (direct / indirect / total with bootstrap CI).

All inputs are seeded synthetic data for illustration; the point is the
table formatting and that the corrected metrics behave as documented.

Run:  PYTHONPATH=. python3 examples/regression_bugfix_report.py
"""
import os

import numpy as np
import pandas as pd

from equimed_dss.domain1 import DecisionFlipRate, InterRaterReliability
from equimed_dss.domain2 import HarmAdjustedFairnessGap, IntersectionalBiasScore
from equimed_dss.domain3 import TemporalFairnessDrift
from equimed_dss.appendix.advanced_metrics import JensenShannonDivergence
from equimed_dss.appendix.info_theory import AdvancedInfoTheoryMetrics
from equimed_dss.statistics import HierarchicalLinearModeling, MediationAnalysis
from equimed_dss.reporting import (
    export_table,
    hierarchical_coefficients_table,
    mediation_effects_table,
)

RNG = np.random.default_rng(20260613)


def bugfix_verification_table() -> pd.DataFrame:
    """Recompute one statistic per fix and tabulate before/after behaviour."""
    dfr = DecisionFlipRate().calculate_dfr([0] * 8 + [1, 1], [0] * 10)
    a = JensenShannonDivergence().calculate_jsd([0.7, 0.2, 0.1], [0.1, 0.2, 0.7])["jsd"]
    b = AdvancedInfoTheoryMetrics().calculate_jsd([0.7, 0.2, 0.1], [0.1, 0.2, 0.7])
    hafg = HarmAdjustedFairnessGap().calculate_hafg({"fn": 5, "fp": 10}, {"fn": 2, "fp": 5})
    ba = InterRaterReliability().bland_altman_analysis(
        np.array([[3.0, 2.0], [5.0, 4.0], [6.0, 2.0]])
    )["Judge1-Judge2"]

    rows = [
        ["DecisionFlipRate CI", "percentile of 0/1 vector ≈ [0, 1]",
         "Wilson 95% interval",
         f"flip={dfr['flip_rate']:.2f}, CI=({dfr['ci_lower']:.3f}, {dfr['ci_upper']:.3f})"],
        ["JSD consistency", "distance vs divergence (disagreed)",
         "JS divergence, base 2, [0,1]",
         f"advanced={a:.4f} == info_theory={b:.4f}"],
        ["HAFG normalization", "raw |H1−H2| in [0, ∞)",
         "|H1−H2|/max(H1,H2) in [0,1]",
         f"hafg={hafg['hafg']:.4f} (abs gap={hafg['absolute_harm_gap']:.0f})"],
        ["Sample SD (Bland-Altman/TFD)", "population SD (ddof=0)",
         "sample SD (ddof=1)",
         f"SD_diff={ba['std_difference']:.4f}"],
    ]
    return pd.DataFrame(rows, columns=["Fix", "Before (bug)", "After (v1.5.0)", "Recomputed value"])


def fit_demo_hlm() -> dict:
    """Seeded nested data: patients in hospitals, real fixed effect on age."""
    n_hosp, per = 25, 40
    rows = []
    for h in range(n_hosp):
        u = RNG.normal(0, 0.6)  # hospital random intercept
        for _ in range(per):
            age = RNG.normal(0, 1)
            acuity = RNG.normal(0, 1)
            y = 0.40 * age - 0.25 * acuity + u + RNG.normal(0, 0.8)
            rows.append({"hospital": f"H{h:02d}", "age": age, "acuity": acuity, "y": y})
    df = pd.DataFrame(rows)
    return HierarchicalLinearModeling().fit_model(
        data=df, outcome_var="y",
        level1_predictors=["age", "acuity"], level2_var="hospital",
    )


def run_demo_mediation() -> dict:
    """Seeded data with a genuine indirect path X -> M -> Y."""
    n = 600
    x = RNG.normal(0, 1, n)
    m = 0.6 * x + RNG.normal(0, 0.8, n)          # a1 = 0.6
    y = 0.2 * x + 0.7 * m + RNG.normal(0, 0.8, n)  # b2 = 0.7
    df = pd.DataFrame({"x": x, "m": m, "y": y})
    return MediationAnalysis(n_bootstrap=2000, random_state=7).analyze_mediation(
        data=df, treatment_var="x", mediator_var="m", outcome_var="y",
    )


def section(title: str, df: pd.DataFrame) -> str:
    return f"### {title}\n\n{export_table(df, fmt='markdown')}\n"


def main() -> None:
    bug = bugfix_verification_table()
    hlm = fit_demo_hlm()
    med = run_demo_mediation()
    coef = hierarchical_coefficients_table(hlm)
    medt = mediation_effects_table(med)

    md = ["# EquiMed-DSS v1.5.0 - Regression & Bug-fix Verification Report",
          "",
          "_Seeded synthetic data; tables demonstrate the corrected metric "
          "behaviour and the reporting layer's regression tables._",
          "",
          section("1. Bug-fix verification (four v1.5.0 fixes)", bug),
          section("2. Hierarchical mixed-effects regression coefficients", coef),
          f"_ICC = {hlm.get('icc', float('nan')):.3f}; "
          f"n = {hlm.get('n_observations')}, groups = {hlm.get('n_groups')}; "
          f"AIC = {hlm.get('aic', float('nan')):.1f}, BIC = {hlm.get('bic', float('nan')):.1f}._",
          "",
          section("3. Mediation effects (bootstrap 95% CI)", medt),
          f"_Proportion mediated = {med.get('proportion_mediated', float('nan')):.1%}; "
          f"indirect 95% CI = ({med.get('indirect_ci_lower', float('nan')):.3f}, "
          f"{med.get('indirect_ci_upper', float('nan')):.3f})._",
          ""]
    report = "\n".join(md)

    print(report)
    out = os.path.join(os.path.dirname(__file__), "..", "docs", "REGRESSION_BUGFIX_REPORT.md")
    with open(os.path.abspath(out), "w") as f:
        f.write(report)
    # also drop a LaTeX version of the coefficient table for manuscripts
    export_table(coef, fmt="latex",
                 path=os.path.abspath(os.path.join(os.path.dirname(out), "hlm_coefficients.tex")))
    print(f"\nWrote {os.path.abspath(out)} and hlm_coefficients.tex")


if __name__ == "__main__":
    main()
