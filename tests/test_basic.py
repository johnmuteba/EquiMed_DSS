"""Basic tests to ensure the package imports correctly."""

import pytest


def test_import_domain1():
    """Test that domain1 modules can be imported."""
    from equimed_dss.domain1 import (
        DecisionFlipRate,
        EmbeddingConsistencyScore,
        InterRaterReliability,
    )

    assert DecisionFlipRate is not None
    assert EmbeddingConsistencyScore is not None
    assert InterRaterReliability is not None


def test_import_domain2():
    """Test that domain2 modules can be imported."""
    from equimed_dss.domain2 import (
        EthicalRiskIndex,
        HarmAdjustedFairnessGap,
        HierarchicalEquityRatio,
        IntersectionalBiasScore,
    )

    assert HierarchicalEquityRatio is not None
    assert HarmAdjustedFairnessGap is not None
    assert EthicalRiskIndex is not None
    assert IntersectionalBiasScore is not None


def test_import_domain3():
    """Test that domain3 modules can be imported."""
    from equimed_dss.domain3 import (
        AuditTraceabilityScore,
        GovernanceComplianceIndex,
        TemporalFairnessDrift,
    )

    assert TemporalFairnessDrift is not None
    assert AuditTraceabilityScore is not None
    assert GovernanceComplianceIndex is not None


def test_import_appendix():
    """Test that appendix modules can be imported."""
    from equimed_dss.appendix import (
        AdvancedInfoTheoryMetrics,
        AdvancedNetworkAnalysis,
        AdvancedNetworkMetrics,
        AdvancedReliabilityMetrics,
    )

    assert AdvancedReliabilityMetrics is not None
    assert AdvancedInfoTheoryMetrics is not None
    assert AdvancedNetworkMetrics is not None
    assert AdvancedNetworkAnalysis is not None


def test_package_version():
    """Test that the package has basic attributes."""
    import equimed_dss

    # Just verify the package imports
    assert equimed_dss is not None


def test_basic_functionality():
    """Test that basic functionality works on Python 3.8."""
    import numpy as np
    from equimed_dss.appendix.advanced_metrics import BootstrapConfidenceIntervals

    # Test basic functionality
    bci = BootstrapConfidenceIntervals(n_bootstrap=10, random_state=42)
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = bci.calculate_bci(data)

    # Verify expected keys exist
    assert "ci_lower" in result
    assert "ci_upper" in result
    assert "observed_statistic" in result
    assert isinstance(result["observed_statistic"], float)
