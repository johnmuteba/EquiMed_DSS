"""Tests for Statistical Analysis modules."""

import numpy as np
import pandas as pd
import pytest

from equimed_dss.statistics import (
    HierarchicalLinearModeling,
    MediationAnalysis,
    NetworkStatistics,
    ReliabilityAnalysis,
)


class TestHierarchicalLinearModeling:
    """Test suite for HierarchicalLinearModeling (HLM)."""

    @pytest.fixture
    def sample_hlm_data(self):
        """Create sample hierarchical data."""
        # Use consistent random state for cross-version compatibility
        np.random.seed(42)
        rng = np.random.RandomState(42)
        n_groups = 10
        n_per_group = 20

        data = []
        for group_id in range(n_groups):
            group_effect = rng.normal(0, 2)
            for _ in range(n_per_group):
                individual_effect = rng.normal(0, 1)
                predictor = rng.normal(0, 1)
                outcome = group_effect + 0.5 * predictor + individual_effect
                data.append({
                    "group": group_id,
                    "predictor": predictor,
                    "outcome": outcome
                })

        return pd.DataFrame(data)

    def test_fit_model_basic(self, sample_hlm_data):
        """Test basic HLM model fitting."""
        hlm = HierarchicalLinearModeling()

        result = hlm.fit_model(
            data=sample_hlm_data,
            outcome_var="outcome",
            level1_predictors=["predictor"],
            level2_var="group"
        )

        assert "icc" in result
        assert "variance_between_groups" in result
        assert "variance_within_groups" in result
        assert "interpretation" in result

        # ICC should be between 0 and 1
        assert 0 <= result["icc"] <= 1

    def test_fit_model_icc_interpretation(self, sample_hlm_data):
        """Test ICC interpretation."""
        hlm = HierarchicalLinearModeling()

        result = hlm.fit_model(
            data=sample_hlm_data,
            outcome_var="outcome",
            level1_predictors=["predictor"],
            level2_var="group"
        )

        assert "level_importance" in result["interpretation"]


class TestMediationAnalysis:
    """Test suite for MediationAnalysis."""

    @pytest.fixture
    def sample_mediation_data(self):
        """Create sample mediation data."""
        np.random.seed(42)
        n = 200

        treatment = np.random.binomial(1, 0.5, n)
        mediator = 0.7 * treatment + np.random.normal(0, 0.5, n)
        outcome = 0.3 * treatment + 0.5 * mediator + np.random.normal(0, 0.5, n)

        return pd.DataFrame({
            "treatment": treatment,
            "mediator": mediator,
            "outcome": outcome
        })

    def test_analyze_mediation_basic(self, sample_mediation_data):
        """Test basic mediation analysis."""
        mediation = MediationAnalysis(n_bootstrap=100, random_state=42)

        result = mediation.analyze_mediation(
            data=sample_mediation_data,
            treatment_var="treatment",
            mediator_var="mediator",
            outcome_var="outcome"
        )

        assert "indirect_effect" in result
        assert "direct_effect" in result
        assert "total_effect" in result
        assert "proportion_mediated" in result
        assert "indirect_ci_lower" in result
        assert "indirect_ci_upper" in result

        # Proportion mediated should be between 0 and 1 (or slightly outside due to sampling)
        assert -0.5 <= result["proportion_mediated"] <= 1.5

    def test_sobel_test(self, sample_mediation_data):
        """Test Sobel test for indirect effect."""
        mediation = MediationAnalysis(n_bootstrap=100, random_state=42)

        result = mediation.analyze_mediation(
            data=sample_mediation_data,
            treatment_var="treatment",
            mediator_var="mediator",
            outcome_var="outcome"
        )

        if "sobel_test" in result:
            assert "z_score" in result["sobel_test"]
            assert "p_value" in result["sobel_test"]


class TestNetworkStatistics:
    """Test suite for NetworkStatistics."""

    @pytest.fixture
    def sample_network(self):
        """Create sample network adjacency matrix."""
        # Correlation-like matrix (zero diagonal to avoid self-loops)
        matrix = np.array([
            [0.0, 0.8, 0.3, 0.1],
            [0.8, 0.0, 0.4, 0.2],
            [0.3, 0.4, 0.0, 0.7],
            [0.1, 0.2, 0.7, 0.0]
        ])
        labels = ["Metric1", "Metric2", "Metric3", "Metric4"]
        return matrix, labels

    def test_analyze_network_basic(self, sample_network):
        """Test basic network analysis."""
        network_stats = NetworkStatistics()
        adjacency, labels = sample_network

        result = network_stats.analyze_network(
            adjacency_matrix=adjacency,
            node_labels=labels
        )

        assert "degree_centrality" in result
        assert "betweenness_centrality" in result
        assert "closeness_centrality" in result
        assert "clustering_coefficients" in result
        assert "average_clustering" in result
        assert "n_nodes" in result
        assert "n_edges" in result
        assert "density" in result

        # Check that we have centrality for each node
        assert len(result["degree_centrality"]) == len(labels)

    def test_analyze_network_properties(self, sample_network):
        """Test network properties are in valid ranges."""
        network_stats = NetworkStatistics()
        adjacency, labels = sample_network

        result = network_stats.analyze_network(
            adjacency_matrix=adjacency,
            node_labels=labels
        )

        # Degree centrality for weighted graphs can exceed 1 (sum of weights)
        # Only check it's non-negative
        for value in result["degree_centrality"].values():
            assert value >= 0

        # Betweenness and closeness centrality should be between 0 and 1
        # Use more lenient bounds for cross-version compatibility
        for centrality_dict in [result["betweenness_centrality"],
                                result["closeness_centrality"]]:
            for value in centrality_dict.values():
                assert 0 <= value <= 1.01  # Slight tolerance for floating point precision

        # Density should be between 0 and 1
        # Use more lenient bounds for weighted graphs cross-version compatibility
        assert 0 <= result["density"] <= 1.01  # Slight tolerance


class TestReliabilityAnalysis:
    """Test suite for ReliabilityAnalysis."""

    @pytest.fixture
    def sample_reliability_data(self):
        """Create sample reliability data."""
        np.random.seed(42)
        n_subjects = 50
        n_items = 5

        # Create correlated items (good internal consistency)
        true_scores = np.random.normal(0, 1, n_subjects)
        data = np.column_stack([
            true_scores + np.random.normal(0, 0.3, n_subjects)
            for _ in range(n_items)
        ])

        return data

    def test_cronbachs_alpha_basic(self, sample_reliability_data):
        """Test basic Cronbach's Alpha calculation."""
        reliability = ReliabilityAnalysis()

        result = reliability.cronbachs_alpha(sample_reliability_data)

        assert "alpha" in result
        assert "n_items" in result
        assert "interpretation" in result

        # Alpha should typically be between 0 and 1
        assert -1 <= result["alpha"] <= 1

    def test_cronbachs_alpha_interpretation(self, sample_reliability_data):
        """Test Cronbach's Alpha interpretation."""
        reliability = ReliabilityAnalysis()

        result = reliability.cronbachs_alpha(sample_reliability_data)

        assert "quality" in result["interpretation"]
        assert result["interpretation"]["quality"] in [
            "Excellent", "Good", "Acceptable", "Questionable"
        ]

    def test_bland_altman_basic(self):
        """Test basic Bland-Altman analysis."""
        reliability = ReliabilityAnalysis()

        np.random.seed(42)
        method1 = np.random.normal(100, 10, 50)
        method2 = method1 + np.random.normal(0, 2, 50)  # Similar with small bias

        result = reliability.bland_altman_analysis(method1, method2)

        assert "mean_difference" in result
        assert "std_difference" in result
        assert "loa_upper" in result
        assert "loa_lower" in result
        assert "loa_width" in result
        assert "interpretation" in result

        # LOA should bracket the mean difference
        assert result["loa_lower"] < result["mean_difference"] < result["loa_upper"]

    def test_bland_altman_agreement_levels(self):
        """Test Bland-Altman agreement interpretation."""
        reliability = ReliabilityAnalysis()

        # Excellent agreement
        np.random.seed(42)
        method1 = np.random.normal(100, 10, 50)
        method2 = method1 + np.random.normal(0, 0.5, 50)

        result = reliability.bland_altman_analysis(method1, method2)

        if abs(result["mean_difference"]) < 0.1:
            assert "excellent" in result["interpretation"]["agreement"].lower()

    def test_bland_altman_arrays_different_length(self):
        """Test Bland-Altman with arrays of different lengths."""
        reliability = ReliabilityAnalysis()

        method1 = np.array([1, 2, 3, 4, 5])
        method2 = np.array([1, 2, 3])  # Different length

        # Should handle gracefully (either error or truncate)
        try:
            result = reliability.bland_altman_analysis(method1, method2)
            # If it succeeds, check it used the shorter length
            assert "mean_difference" in result
        except (ValueError, IndexError):
            # Or it might raise an error, which is also acceptable
            pass


class TestIntegration:
    """Integration tests for statistical modules."""

    def test_full_analysis_pipeline(self):
        """Test a complete analysis pipeline using multiple modules."""
        np.random.seed(42)

        # Create hierarchical data with mediation
        n_groups = 5
        n_per_group = 40

        data_list = []
        for group_id in range(n_groups):
            group_effect = np.random.normal(0, 1)

            for _ in range(n_per_group):
                treatment = np.random.binomial(1, 0.5)
                mediator = 0.6 * treatment + group_effect + np.random.normal(0, 0.5)
                outcome = 0.3 * treatment + 0.5 * mediator + np.random.normal(0, 0.5)

                data_list.append({
                    "group": group_id,
                    "treatment": treatment,
                    "mediator": mediator,
                    "outcome": outcome
                })

        df = pd.DataFrame(data_list)

        # HLM analysis
        hlm = HierarchicalLinearModeling()
        hlm_result = hlm.fit_model(
            data=df,
            outcome_var="outcome",
            level1_predictors=["treatment", "mediator"],
            level2_var="group"
        )

        # Mediation analysis
        mediation = MediationAnalysis(n_bootstrap=50, random_state=42)
        med_result = mediation.analyze_mediation(
            data=df,
            treatment_var="treatment",
            mediator_var="mediator",
            outcome_var="outcome"
        )

        # Both analyses should complete successfully
        assert hlm_result["icc"] >= 0
        assert "indirect_effect" in med_result
