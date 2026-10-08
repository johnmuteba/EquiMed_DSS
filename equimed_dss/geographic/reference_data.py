"""Reference disease-burden distributions for the geographic metrics.

Source: World Health Organization. Global Health Estimates 2023: disease burden
by cause, age, sex, by country and by region, 2000-2023. Geneva: World Health
Organization; 2026. Workbook "Global Health Estimates 2023: DALYs by cause and
region, 2000-2023" (WHO regions), year 2023, cause "Ischaemic heart disease".
Aggregate published statistics, not patient-level data.

WHO_GHE2023_IHD_DALYS_THOUSANDS: ischaemic heart disease (IHD) DALYs by WHO
region in 2023, in thousands, as published (rounded here to the nearest DALY).
The six regions sum to the published global total.

WHO_GHE2023_POPULATION_THOUSANDS: total population (all ages) by WHO region in
2023, in thousands, from the same workbook.

WHO_REGION_IHD_BURDEN: each region's share of the regional IHD DALYs (count
share). It asks whether evidence is where the patients are, and is the default
burden target for ``BurdenEvidenceMismatch``.

WHO_REGION_IHD_BURDEN_RATE: each region's crude IHD DALY rate (DALYs per
person) as a share of the summed regional rates. A population-size-independent
alternative; a young age structure lowers a region's crude rate.

Keys are the WHO regional-office acronyms (AFRO, AMRO, EMRO, EURO, SEARO,
WPRO). The WHO Global Health Observatory codes the same regions AFR, AMR, EMR,
EUR, SEAR and WPR; ``WHO_REGION_CODES`` maps those codes to the keys used here.

Changed in 1.10.0: versions up to 1.9.5 bundled age-standardised IHD DALY rates
attributed to the GBD 2019 study whose regional values could not be traced to a
published table. They were replaced by the WHO figures above.
"""

WHO_GHE2023_IHD_DALYS_THOUSANDS = {
    "AFRO": 11696.757,
    "AMRO": 24317.667,
    "EMRO": 20537.260,
    "EURO": 37936.997,
    "SEARO": 52552.457,
    "WPRO": 64999.038,
}

WHO_GHE2023_POPULATION_THOUSANDS = {
    "AFRO": 1236605.866,
    "AMRO": 1039858.310,
    "EMRO": 809901.418,
    "EURO": 939115.518,
    "SEARO": 1817153.989,
    "WPRO": 2232975.234,
}

WHO_REGION_CODES = {
    "AFR": "AFRO",
    "AMR": "AMRO",
    "EMR": "EMRO",
    "EUR": "EURO",
    "SEAR": "SEARO",
    "WPR": "WPRO",
}

_TOTAL_DALYS = sum(WHO_GHE2023_IHD_DALYS_THOUSANDS.values())
WHO_REGION_IHD_BURDEN = {
    region: dalys / _TOTAL_DALYS
    for region, dalys in WHO_GHE2023_IHD_DALYS_THOUSANDS.items()
}

_RATES = {
    region: WHO_GHE2023_IHD_DALYS_THOUSANDS[region]
    / WHO_GHE2023_POPULATION_THOUSANDS[region]
    for region in WHO_GHE2023_IHD_DALYS_THOUSANDS
}
_TOTAL_RATE = sum(_RATES.values())
WHO_REGION_IHD_BURDEN_RATE = {
    region: rate / _TOTAL_RATE for region, rate in _RATES.items()
}
