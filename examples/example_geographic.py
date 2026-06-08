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
