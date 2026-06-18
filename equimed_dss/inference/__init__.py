"""Inference utilities: confidence intervals and hypothesis tests for metrics.

Turns EquiMed-DSS point-estimate metrics into interval estimates with explicit
methods and sample sizes, and provides cluster-aware resampling so repeated
evaluations of the same patient/visit do not inflate precision.
"""
from equimed_dss.inference.resampling import (
    InferenceResult,
    MetricResult,
    bootstrap_ci,
    bootstrap_metric,
    permutation_test,
    proportion_ci,
    wilson_ci,
)

__all__ = [
    "InferenceResult",
    "MetricResult",
    "wilson_ci",
    "proportion_ci",
    "bootstrap_ci",
    "bootstrap_metric",
    "permutation_test",
]
