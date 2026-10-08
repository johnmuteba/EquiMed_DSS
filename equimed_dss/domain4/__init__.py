"""
Domain 4: Representation, faithfulness, and vulnerability metrics.

Implements the PhD-framework metrics:
    - Semantic Parity Gap (SPG): latent representation bias by demographic.
    - Clinical Hallucination Rate (CHR): unsupported-claim rate in RAG outputs.
    - Instructional Vulnerability Index (IVI): susceptibility to bias-priming.
    - Geographic Representation Index (GRI) and Geographic Bias (GB).
"""

from .chr import ClinicalHallucinationRate
from .gri import GeographicRepresentationIndex
from .ivi import InstructionalVulnerabilityIndex
from .spg import SemanticParityGap

__all__ = [
    "SemanticParityGap",
    "ClinicalHallucinationRate",
    "InstructionalVulnerabilityIndex",
    "GeographicRepresentationIndex",
]
