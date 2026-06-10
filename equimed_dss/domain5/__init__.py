"""
Domain 5: Novel fairness metrics for health-AI evaluation (technical supplement).

Twelve metrics extending fairness assessment to intersectionality, clinical-harm
weighting, geographic bias, healthcare-system stratification, and robustness:
    - IntersectionalCalibrationError (ICE)
    - WeightedClinicalHarmAdjustedFairnessGap (wHAFG)
    - LexicalDiversityDisparityIndex (LDDI)
    - RecommendationEntropyGap (REG)
    - CounterfactualParityScore (CPS)
    - ClinicalInformationDensityRatio (CIDR)
    - DiagnosticCompletenessIndex (DCI)
    - UncertaintyQuantificationGap (UQG)
    - GeographicRepresentationBiasIndex (GRBI)
    - HealthcareSystemStratifiedFairness (HSSF)
    - IntersectionalShapleyFairnessValue (ISFV)
    - SemanticRobustnessParityIndex (SRPI)
"""
from .calibration import IntersectionalCalibrationError
from .harm import WeightedClinicalHarmAdjustedFairnessGap
from .text import (
    LexicalDiversityDisparityIndex,
    RecommendationEntropyGap,
    ClinicalInformationDensityRatio,
    DiagnosticCompletenessIndex,
    UncertaintyQuantificationGap,
)
from .counterfactual import CounterfactualParityScore, SemanticRobustnessParityIndex
from .geographic_bias import GeographicRepresentationBiasIndex
from .system import HealthcareSystemStratifiedFairness
from .shapley import IntersectionalShapleyFairnessValue

__all__ = [
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
