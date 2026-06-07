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
