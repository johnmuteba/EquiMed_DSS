"""Geographic-equity metrics for EquiMed-DSS."""

from .burden_evidence import BurdenEvidenceMismatch
from .concentration import GeographicConcentration
from .reference_data import (
    WHO_GHE2023_IHD_DALYS_THOUSANDS,
    WHO_GHE2023_POPULATION_THOUSANDS,
    WHO_REGION_CODES,
    WHO_REGION_IHD_BURDEN,
    WHO_REGION_IHD_BURDEN_RATE,
)

__all__ = [
    "BurdenEvidenceMismatch",
    "GeographicConcentration",
    "WHO_GHE2023_IHD_DALYS_THOUSANDS",
    "WHO_GHE2023_POPULATION_THOUSANDS",
    "WHO_REGION_CODES",
    "WHO_REGION_IHD_BURDEN",
    "WHO_REGION_IHD_BURDEN_RATE",
]
