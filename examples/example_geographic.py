"""Geographic-equity metrics demo (BEMI + GCC) and equity visualisations.

Uses illustrative sample counts. To analyse your own corpus, replace
``evidence`` below with its per-WHO-region study or case counts and, if needed,
pass a burden reference of your own (the bundled one is WHO Global Health
Estimates 2023, ischaemic heart disease).

This example also demonstrates two equity figures: ``plot_geographic_dumbbell``
(a Cleveland/dumbbell chart of burden against evidence share) and
``plot_equity_radar``. Both return a Matplotlib ``Figure`` and, when
``save_path`` is given, also write the file.
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
    # The evidence shares are the illustrative counts used above.
    # ------------------------------------------------------------------
    total = sum(evidence.values())
    evidence_shares = {region: n / total for region, n in evidence.items()}
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
