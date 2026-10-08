"""
EquiMed-DSS: A Comprehensive Library for Clinical AI Fairness Assessment

This package provides 37 metrics (26 in five core domains, 2 geographic and
9 in an advanced appendix) for evaluating reliability, equity, governance,
representation, robustness, and intersectionality in clinical AI systems.

Domains:
    - domain1: Reliability & robustness (DecisionFlipRate, EmbeddingConsistencyScore,
      InterRaterReliability/ICC)
    - domain2: Fairness, equity & ethics (HER, HAFG, ERI, IBS)
    - domain3: Governance & transparency (TFD, ATS, GCI)
    - domain4: Representation & robustness (SPG, CHR, IVI, GRI)
    - domain5: Technical-supplement fairness (ICE, wHAFG, LDDI, REG, CPS, CIDR,
      DCI, UQG, GRBI, HSSF, ISFV, SRPI)
    - geographic: Burden-Evidence Mismatch (BEMI), Geographic Concentration (GCC),
      and WHO Global Health Estimates 2023 IHD burden reference shares
    - appendix: Advanced metrics (BCI, SPA, MIC, JSD, WD, NM, TS, RCS)
    - statistics: HLM/MAIHDA, mediation, network, reliability
    - reporting: tidy result tables (markdown / LaTeX / HTML)
    - utils: data utilities and visualizations

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
from .domain4 import (
    ClinicalHallucinationRate,
    GeographicRepresentationIndex,
    InstructionalVulnerabilityIndex,
    SemanticParityGap,
)
from .domain5 import (
    ClinicalInformationDensityRatio,
    CounterfactualParityScore,
    DiagnosticCompletenessIndex,
    GeographicRepresentationBiasIndex,
    HealthcareSystemStratifiedFairness,
    IntersectionalCalibrationError,
    IntersectionalShapleyFairnessValue,
    LexicalDiversityDisparityIndex,
    RecommendationEntropyGap,
    SemanticRobustnessParityIndex,
    UncertaintyQuantificationGap,
    WeightedClinicalHarmAdjustedFairnessGap,
)
from .geographic import (
    WHO_REGION_CODES,
    WHO_REGION_IHD_BURDEN,
    WHO_REGION_IHD_BURDEN_RATE,
    BurdenEvidenceMismatch,
    GeographicConcentration,
)
from .inference import (
    InferenceResult,
    MetricResult,
    bootstrap_ci,
    bootstrap_metric,
    permutation_test,
    proportion_ci,
    wilson_ci,
)
from .reporting import (
    export_table,
    geographic_table,
    hierarchical_coefficients_table,
    mediation_effects_table,
    network_centrality_table,
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
    "WHO_REGION_CODES",
    "WHO_REGION_IHD_BURDEN",
    "WHO_REGION_IHD_BURDEN_RATE",
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
    "InferenceResult",
    "MetricResult",
    "wilson_ci",
    "proportion_ci",
    "bootstrap_ci",
    "bootstrap_metric",
    "permutation_test",
]
