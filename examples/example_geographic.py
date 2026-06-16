"""Geographic-equity metrics demo (BEMI + GCC) and equity visualisations.

Uses illustrative sample counts. REAL-DATA HOOK: to render the actual
manuscript numbers on your machine, replace ``evidence`` below with your
corpus's per-WHO-region study/case counts (e.g. loaded from your
geography_distinctive_layer output) and pass your sourced burden reference.

This example also demonstrates the two equity figures added in v1.5.2:
``plot_geographic_dumbbell`` (the Cleveland/dumbbell chart used for manuscript
Figure 4/5) and ``plot_equity_radar``. Both return a Matplotlib ``Figure`` and,
when ``save_path`` is given, also write the file.
"""
import matplotlib
matplotlib.use("Agg")  # headless-safe; remove for interactive viewing

from equimed_dss.geographic import (
    BurdenEvidenceMismatch,
    GeographicConcentration,
    WHO_REGION_IHD_BURDEN,
)
from equimed_dss.reporting import export_table, geographic_table
from equimed_dss.utils import plot_equity_radar, plot_geographic_dumbbell


def main():
    # Illustrative evidence distribution (replace via the real-data hook).
    evidence = {"AFRO": 5, "AMRO": 40, "EURO": 30, "SEARO": 3, "WPRO": 10, "EMRO": 2}

    bemi = BurdenEvidenceMismatch()
    bemi_result = bemi.calculate_bemi(
        evidence_counts=evidence, burden_shares=WHO_REGION_IHD_BURDEN
    )
    print(f"BEMI: {bemi_result['bemi']:.3f}  (0 = aligned, 1 = disjoint)")
    print(f"Most under-served region: {bemi_result['most_underserved_region']}")

    gcc = GeographicConcentration()
    gcc_result = gcc.calculate_gcc(evidence)
    print(f"Gini* (G*): {gcc_result['gini_corrected']:.3f}")
    print(f"H_norm: {gcc_result['entropy_normalized']:.3f}")

    # Combined geographic summary table, exported in three formats.
    df = geographic_table(bemi_result, gcc_result)
    print("\nCombined geographic summary:")
    print(export_table(df, fmt="markdown"))

    # ------------------------------------------------------------------
    # Visualisation 1: geographic dumbbell (burden vs evidence per region).
    # Shares need not be pre-normalised; the function normalises counts.
    # The evidence shares below are the manuscript's verified corpus geography
    # (AMRO 78.0%, EURO 10.5%, WPRO 7.7%, EMRO 3.7%, AFRO 0.2%, SEARO 0%).
    # ------------------------------------------------------------------
    evidence_shares = {
        "AMRO": 0.780, "EURO": 0.105, "WPRO": 0.077,
        "EMRO": 0.037, "AFRO": 0.002, "SEARO": 0.000,
    }
    fig_db = plot_geographic_dumbbell(
        burden_shares=WHO_REGION_IHD_BURDEN,
        evidence_shares=evidence_shares,
        title="IHD burden vs evidence supply by WHO region",
        burden_label="IHD burden share",
        evidence_label="Evidence share",
        save_path="example_geographic_dumbbell.png",
    )
    print("\nSaved dumbbell chart -> example_geographic_dumbbell.png")

    # ------------------------------------------------------------------
    # Visualisation 2: equity radar across EquiMed-DSS domains (illustrative
    # domain equity sub-scores, 0-1; reference ring at 0.8).
    # ------------------------------------------------------------------
    domain_scores = {
        "Representation": 0.21,
        "Calibration": 0.46,
        "Prompt robustness": 0.73,
        "Geographic": 0.33,
        "Governance": 0.16,
    }
    fig_radar = plot_equity_radar(
        domain_scores,
        title="EquiMed-DSS equity radar (illustrative)",
        reference=0.8,
        save_path="example_equity_radar.png",
    )
    print("Saved equity radar  -> example_equity_radar.png")
    return fig_db, fig_radar


if __name__ == "__main__":
    main()
