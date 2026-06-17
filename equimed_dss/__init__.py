"""
EquiMed-DSS: A Comprehensive Library for Clinical AI Fairness Assessment

This package provides 37 metrics across five domains, plus geographic and
advanced-appendix metrics, for evaluating reliability, equity, governance,
representation, robustness, and intersectionality in clinical AI systems.

Domains:
    - domain1: Reliability & robustness (DecisionFlipRate, EmbeddingConsistencyScore,
      InterRaterReliability/ICC)
    - domain2: Fairness, equity & ethics (HER, HAFG, ERI, IBS)
    - domain3: Governance & transparency (TFD, ATS, GCI)
    - domain4: Representation & robustness (SPG, CHR, IVI, GRI)
    - domain5: Technical-supplement fairness (ICE, wHAFG, LDDI, REG, CPS, CIDR,
      DCI, UQG, GRBI, HSSF, ISFV, SRPI)
    - geographic: Burden-Evidence Mismatch (BEMI), Geographic Concentration (GCC)
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
from .inference import (
    InferenceResult,
    wilson_ci,
    proportion_ci,
    bootstrap_ci,
    permutation_test,
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
    "InferenceResult",
    "wilson_ci",
    "proportion_ci",
    "bootstrap_ci",
    "permutation_test",
]
