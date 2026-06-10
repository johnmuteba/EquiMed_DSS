"""
EquiMed-DSS: A Comprehensive Library for Clinical AI Fairness Assessment

This package provides 19 novel metrics for evaluating reliability, equity,
governance, and intersectionality in clinical AI systems.

Domains:
    - domain1: Reliability & Calibration metrics (DFR, ECS, ICC)
    - domain2: Fairness, Equity & Ethics metrics (HER, HAFG, ERI, IBS)
    - domain3: Governance & Transparency metrics (TFD, ATS, GCI)
    - appendix: Advanced metrics (BCI, SPA, MIC, JSD, WD, NM, TS, RCS)
    - statistics: Statistical analyses (HLM, Mediation, Network, Reliability)
    - utils: Data utilities and visualizations

Example:
    >>> from equimed_dss.domain2 import HierarchicalEquityRatio
    >>> her = HierarchicalEquityRatio()
    >>> scores = her.calculate_her({'White': 0.85, 'Black': 0.78})

For more information, see: https://github.com/johnmuteba/EquiMed_DSS
"""

from .__version__ import (
    __author__,
    __copyright__,
    __description__,
    __license__,
    __title__,
    __version__,
    __version_info__,
)
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
from .domain4 import (
    SemanticParityGap,
    ClinicalHallucinationRate,
    InstructionalVulnerabilityIndex,
    GeographicRepresentationIndex,
)
from .domain5 import (
    IntersectionalCalibrationError,
    WeightedClinicalHarmAdjustedFairnessGap,
    LexicalDiversityDisparityIndex,
    RecommendationEntropyGap,
    CounterfactualParityScore,
    ClinicalInformationDensityRatio,
    DiagnosticCompletenessIndex,
    UncertaintyQuantificationGap,
    GeographicRepresentationBiasIndex,
    HealthcareSystemStratifiedFairness,
    IntersectionalShapleyFairnessValue,
    SemanticRobustnessParityIndex,
)

__all__ = [
    "__version__",
    "__version_info__",
    "__title__",
    "__description__",
    "__author__",
    "__license__",
    "__copyright__",
    "BurdenEvidenceMismatch",
    "GeographicConcentration",
    "WHO_REGION_IHD_BURDEN",
    "export_table",
    "geographic_table",
    "hierarchical_coefficients_table",
    "mediation_effects_table",
    "network_centrality_table",
    "SemanticParityGap",
    "ClinicalHallucinationRate",
    "InstructionalVulnerabilityIndex",
    "GeographicRepresentationIndex",
    "IntersectionalCalibrationError",
    "WeightedClinicalHarmAdjustedFairnessGap",
    "LexicalDiversityDisparityIndex",
    "RecommendationEntropyGap",
    "CounterfactualParityScore",
    "ClinicalInformationDensityRatio",
    "DiagnosticCompletenessIndex",
    "UncertaintyQuantificationGap",
    "GeographicRepresentationBiasIndex",
    "HealthcareSystemStratifiedFairness",
    "IntersectionalShapleyFairnessValue",
    "SemanticRobustnessParityIndex",
]
