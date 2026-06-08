"""Geographic-equity metrics demo (BEMI + GCC).

Uses illustrative sample counts. REAL-DATA HOOK: to render the actual
manuscript numbers on your machine, replace ``evidence`` below with your
corpus's per-WHO-region study/case counts (e.g. loaded from your
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


if __name__ == "__main__":
    main()
