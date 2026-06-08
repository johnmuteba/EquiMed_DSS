"""
Sample data generator for EquiMed_DSS library.

This module provides utilities to generate synthetic data for testing
and demonstrating the library's fairness metrics and analyses.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


class SampleDataGenerator:
    """
    Generate synthetic data for testing EquiMed_DSS metrics.

    This class creates realistic synthetic datasets for:
    - Clinical AI fairness evaluation
    - Demographic disparity analysis
    - Model calibration testing
    - Intersectional bias detection

    Example:
        >>> generator = SampleDataGenerator(random_state=42)
        >>> data = generator.generate_fairness_data(n_samples=1000)
        >>> print(data.columns.tolist())
        ['id', 'race', 'gender', 'age_group', 'prediction', 'actual', 'confidence']
    """

    # Default demographic categories
    RACE_CATEGORIES = ["White", "Black", "Hispanic", "Asian", "Native American"]
    GENDER_CATEGORIES = ["Male", "Female", "Non-binary"]
    AGE_GROUPS = ["Young Adult", "Middle Age", "Elderly"]
    SES_CATEGORIES = ["Low", "Middle", "High"]

    def __init__(self, random_state: Optional[int] = None):
        """
        Initialize the sample data generator.

        Args:
            random_state: Random seed for reproducibility
        """
        self.rng = np.random.RandomState(random_state)

    def generate_fairness_data(
        self,
        n_samples: int = 1000,
        include_bias: bool = True,
        bias_magnitude: float = 0.1,
    ) -> pd.DataFrame:
        """
        Generate synthetic data for fairness analysis.

        Args:
            n_samples: Number of samples to generate
            include_bias: Whether to include simulated demographic bias
            bias_magnitude: Strength of simulated bias (0-1)

        Returns:
            DataFrame with columns: id, race, gender, age_group, prediction, actual, confidence

        Example:
            >>> generator = SampleDataGenerator(random_state=42)
            >>> df = generator.generate_fairness_data(n_samples=500)
            >>> df.groupby('race')['prediction'].mean()
        """
        data = {
            "id": [f"patient_{i:05d}" for i in range(n_samples)],
            "race": self.rng.choice(self.RACE_CATEGORIES, n_samples),
            "gender": self.rng.choice(self.GENDER_CATEGORIES, n_samples),
            "age_group": self.rng.choice(self.AGE_GROUPS, n_samples),
        }

        # Generate base predictions
        base_predictions = self.rng.beta(2, 2, n_samples)

        # Add demographic bias if requested
        if include_bias:
            bias_adjustments = np.zeros(n_samples)
            # Simulate lower predictions for certain groups
            for i, race in enumerate(data["race"]):
                if race == "Black":
                    bias_adjustments[i] = -bias_magnitude * 0.8
                elif race == "Hispanic":
                    bias_adjustments[i] = -bias_magnitude * 0.5
                elif race == "Native American":
                    bias_adjustments[i] = -bias_magnitude * 0.6
            base_predictions = np.clip(base_predictions + bias_adjustments, 0, 1)

        data["prediction"] = base_predictions
        data["actual"] = (self.rng.random(n_samples) < base_predictions).astype(int)
        data["confidence"] = 0.5 + 0.5 * np.abs(base_predictions - 0.5)

        return pd.DataFrame(data)

    def generate_calibration_data(
        self, n_samples: int = 500, calibration_error: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate data for calibration testing.

        Args:
            n_samples: Number of samples
            calibration_error: Amount of miscalibration to simulate (0 = perfect)

        Returns:
            Tuple of (predictions, actuals) arrays

        Example:
            >>> generator = SampleDataGenerator(random_state=42)
            >>> predictions, actuals = generator.generate_calibration_data(n_samples=1000)
            >>> from equimed_dss.domain1 import ExpectedCalibrationScore
            >>> ecs = ExpectedCalibrationScore()
            >>> result = ecs.calculate_ecs(predictions, actuals)
        """
        predictions = self.rng.uniform(0.1, 0.9, n_samples)

        # Add calibration error
        biased_probs = predictions + calibration_error * (predictions - 0.5)
        biased_probs = np.clip(biased_probs, 0.01, 0.99)

        actuals = (self.rng.random(n_samples) < biased_probs).astype(int)

        return predictions, actuals

    def generate_reliability_data(
        self, n_subjects: int = 20, n_raters: int = 3, agreement_level: float = 0.8
    ) -> np.ndarray:
        """
        Generate inter-rater reliability data.

        Args:
            n_subjects: Number of subjects being rated
            n_raters: Number of raters
            agreement_level: Expected ICC (0-1)

        Returns:
            2D array of shape (n_subjects, n_raters)

        Example:
            >>> generator = SampleDataGenerator(random_state=42)
            >>> ratings = generator.generate_reliability_data(n_subjects=30, n_raters=4)
            >>> from equimed_dss.domain1 import IntraclassCorrelationCoefficient
            >>> icc = IntraclassCorrelationCoefficient()
            >>> result = icc.calculate_icc(ratings)
        """
        # True scores
        true_scores = self.rng.normal(50, 10, n_subjects)

        # Add rater noise based on agreement level
        noise_sd = 10 * (1 - agreement_level)
        ratings = np.zeros((n_subjects, n_raters))

        for r in range(n_raters):
            rater_bias = self.rng.normal(0, 5)
            ratings[:, r] = (
                true_scores + rater_bias + self.rng.normal(0, noise_sd, n_subjects)
            )

        return ratings

    def generate_temporal_data(
        self,
        n_timepoints: int = 50,
        drift_point: Optional[int] = None,
        drift_magnitude: float = 0.1,
    ) -> np.ndarray:
        """
        Generate temporal fairness data with optional drift.

        Args:
            n_timepoints: Number of time points
            drift_point: Time point where drift begins (None = no drift)
            drift_magnitude: Size of drift

        Returns:
            Array of fairness metric values over time

        Example:
            >>> generator = SampleDataGenerator(random_state=42)
            >>> time_series = generator.generate_temporal_data(n_timepoints=100, drift_point=60)
            >>> from equimed_dss.domain3 import TemporalFairnessDrift
            >>> tfd = TemporalFairnessDrift()
            >>> result = tfd.calculate_drift(time_series)
        """
        base_value = 0.85
        noise = self.rng.normal(0, 0.02, n_timepoints)

        if drift_point is not None:
            drift = np.zeros(n_timepoints)
            drift[drift_point:] = -drift_magnitude * np.linspace(
                0, 1, n_timepoints - drift_point
            )
            return base_value + noise + drift

        return base_value + noise

    def generate_group_performance_data(
        self, n_groups: int = 5, disparity_level: float = 0.1
    ) -> Dict[str, float]:
        """
        Generate group performance data for equity analysis.

        Args:
            n_groups: Number of demographic groups
            disparity_level: Amount of performance disparity (0 = equal)

        Returns:
            Dictionary mapping group names to performance scores

        Example:
            >>> generator = SampleDataGenerator(random_state=42)
            >>> group_scores = generator.generate_group_performance_data()
            >>> from equimed_dss.domain2 import HierarchicalEquityRatio
            >>> her = HierarchicalEquityRatio()
            >>> result = her.calculate_her(group_scores)
        """
        base_performance = 0.85
        groups = self.RACE_CATEGORIES[:n_groups]

        performance = {}
        for i, group in enumerate(groups):
            # Add varying disparity
            disparity = self.rng.uniform(-disparity_level, disparity_level)
            performance[group] = max(0.5, min(1.0, base_performance + disparity))

        return performance

    def generate_distribution_data(
        self, n_samples: int = 100, difference: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate two distributions for comparison.

        Args:
            n_samples: Samples per distribution
            difference: Mean difference between distributions

        Returns:
            Tuple of two arrays

        Example:
            >>> generator = SampleDataGenerator(random_state=42)
            >>> dist_a, dist_b = generator.generate_distribution_data(difference=0.2)
            >>> from equimed_dss.appendix import JensenShannonDivergence
            >>> jsd = JensenShannonDivergence()
            >>> result = jsd.calculate_jsd(dist_a, dist_b)
        """
        dist_a = self.rng.beta(5, 5, n_samples)
        dist_b = self.rng.beta(5 - difference * 2, 5 + difference * 2, n_samples)

        return dist_a, dist_b

    def generate_network_data(self, n_nodes: int = 10) -> np.ndarray:
        """
        Generate a correlation/adjacency matrix for network analysis.

        Args:
            n_nodes: Number of nodes in the network

        Returns:
            Symmetric adjacency matrix (n_nodes x n_nodes)

        Example:
            >>> generator = SampleDataGenerator(random_state=42)
            >>> adjacency = generator.generate_network_data(n_nodes=8)
            >>> from equimed_dss.appendix import NetworkModularity
            >>> nm = NetworkModularity()
            >>> result = nm.calculate_modularity(adjacency)
        """
        # Generate random correlation matrix
        random_matrix = self.rng.rand(n_nodes, n_nodes)
        symmetric = (random_matrix + random_matrix.T) / 2
        np.fill_diagonal(symmetric, 0)  # No self-loops

        return symmetric

    def generate_explanation_data(
        self, n_decisions: int = 50, quality_level: float = 0.7
    ) -> List[Dict[str, float]]:
        """
        Generate explanation quality data for transparency scoring.

        Args:
            n_decisions: Number of AI decisions with explanations
            quality_level: Base quality level (0-1)

        Returns:
            List of explanation quality dictionaries

        Example:
            >>> generator = SampleDataGenerator(random_state=42)
            >>> explanations = generator.generate_explanation_data(n_decisions=100)
            >>> from equimed_dss.appendix import TransparencyScore
            >>> ts = TransparencyScore()
            >>> result = ts.calculate_ts(explanations)
        """
        explanations = []
        for _ in range(n_decisions):
            noise = self.rng.uniform(-0.2, 0.2)
            base = max(0, min(1, quality_level + noise))

            explanations.append(
                {
                    "explanation_quality": max(
                        0, min(1, base + self.rng.uniform(-0.1, 0.1))
                    ),
                    "feature_importance": max(
                        0, min(1, base + self.rng.uniform(-0.1, 0.1))
                    ),
                    "interpretability": max(
                        0, min(1, base + self.rng.uniform(-0.1, 0.1))
                    ),
                }
            )

        return explanations

    def generate_perturbation_data(
        self, n_samples: int = 100, n_perturbations: int = 5, robustness: float = 0.9
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Generate original and perturbed predictions for robustness testing.

        Args:
            n_samples: Number of predictions
            n_perturbations: Number of perturbation sets
            robustness: Expected consistency rate (0-1)

        Returns:
            Tuple of (original_predictions, list of perturbed predictions)

        Example:
            >>> generator = SampleDataGenerator(random_state=42)
            >>> original, perturbed = generator.generate_perturbation_data()
            >>> from equimed_dss.appendix import RobustnessCertificationScore
            >>> rcs = RobustnessCertificationScore()
            >>> result = rcs.calculate_rcs(original, perturbed)
        """
        original = self.rng.randint(0, 2, n_samples)

        perturbed_list = []
        for _ in range(n_perturbations):
            perturbed = original.copy()
            # Flip some predictions based on robustness
            flip_mask = self.rng.random(n_samples) > robustness
            perturbed[flip_mask] = 1 - perturbed[flip_mask]
            perturbed_list.append(perturbed)

        return original, perturbed_list

    def generate_audit_logs(
        self, n_logs: int = 100, completeness: float = 0.9
    ) -> List[Dict[str, bool]]:
        """
        Generate audit log data for traceability scoring.

        Args:
            n_logs: Number of audit log entries
            completeness: Expected completeness rate (0-1)

        Returns:
            List of audit log dictionaries

        Example:
            >>> generator = SampleDataGenerator(random_state=42)
            >>> logs = generator.generate_audit_logs(n_logs=50)
            >>> from equimed_dss.domain3 import AuditTraceabilityScore
            >>> ats = AuditTraceabilityScore()
            >>> result = ats.calculate_ats(logs)
        """
        logs = []
        fields = ["timestamp", "user", "action", "details"]

        for _ in range(n_logs):
            log_entry = {}
            for field in fields:
                log_entry[field] = self.rng.random() < completeness
            logs.append(log_entry)

        return logs

    def generate_hierarchical_data(
        self, n_hospitals: int = 10, n_patients_per_hospital: int = 50
    ) -> pd.DataFrame:
        """
        Generate hierarchical/nested data for HLM analysis.

        Args:
            n_hospitals: Number of hospitals (level 2)
            n_patients_per_hospital: Patients per hospital (level 1)

        Returns:
            DataFrame with hospital_id, patient_id, and outcome columns

        Example:
            >>> generator = SampleDataGenerator(random_state=42)
            >>> data = generator.generate_hierarchical_data()
            >>> from equimed_dss.statistics import HierarchicalLinearModeling
            >>> hlm = HierarchicalLinearModeling()
        """
        data = []

        for h in range(n_hospitals):
            hospital_effect = self.rng.normal(0, 2)

            for p in range(n_patients_per_hospital):
                patient_data = {
                    "hospital_id": f"H{h:03d}",
                    "patient_id": f"P{h:03d}_{p:04d}",
                    "race": self.rng.choice(self.RACE_CATEGORIES),
                    "gender": self.rng.choice(self.GENDER_CATEGORIES),
                    "age": self.rng.randint(18, 90),
                    "predictor": self.rng.normal(0, 1),
                    "outcome": 50 + hospital_effect + self.rng.normal(0, 5),
                }
                data.append(patient_data)

        return pd.DataFrame(data)

    def generate_mediation_data(
        self, n_samples: int = 200, indirect_effect: float = 0.5
    ) -> pd.DataFrame:
        """
        Generate data for mediation analysis.

        Args:
            n_samples: Number of samples
            indirect_effect: Strength of indirect path (0-1)

        Returns:
            DataFrame with exposure, mediator, and outcome columns

        Example:
            >>> generator = SampleDataGenerator(random_state=42)
            >>> data = generator.generate_mediation_data()
            >>> from equimed_dss.statistics import MediationAnalysis
            >>> mediation = MediationAnalysis()
        """
        # X -> M -> Y model
        exposure = self.rng.normal(0, 1, n_samples)
        mediator = 0.5 * exposure + self.rng.normal(0, 0.5, n_samples)
        outcome = (
            0.3 * exposure  # Direct effect
            + indirect_effect * mediator  # Indirect effect
            + self.rng.normal(0, 0.5, n_samples)
        )

        return pd.DataFrame(
            {"exposure": exposure, "mediator": mediator, "outcome": outcome}
        )


def generate_sample_corpus(
    n_documents: int = 100, corpus_type: str = "peer_reviewed"
) -> pd.DataFrame:
    """
    Generate a sample corpus for text-based fairness analysis.

    Args:
        n_documents: Number of documents
        corpus_type: Type of corpus ('peer_reviewed', 'community', 'clinical')

    Returns:
        DataFrame with id, content, and demographic columns

    Example:
        >>> df = generate_sample_corpus(n_documents=50, corpus_type='clinical')
        >>> print(df.columns.tolist())
    """
    generator = SampleDataGenerator(random_state=42)

    templates = {
        "peer_reviewed": [
            "Study of {condition} in {race} {gender} patients.",
            "Clinical trial results for {condition} treatment in {age_group} population.",
            "Meta-analysis of {condition} outcomes across demographic groups.",
        ],
        "community": [
            "Patient forum discussion about {condition}.",
            "Community health resource for {condition} management.",
            "Support group information for {condition} patients.",
        ],
        "clinical": [
            "Patient presents with {condition}. Assessment pending.",
            "Follow-up visit for {condition}. Stable condition.",
            "Emergency admission for acute {condition} symptoms.",
        ],
    }

    conditions = ["diabetes", "hypertension", "heart disease", "asthma", "depression"]
    rng = np.random.RandomState(42)

    data = []
    for i in range(n_documents):
        race = rng.choice(SampleDataGenerator.RACE_CATEGORIES)
        gender = rng.choice(SampleDataGenerator.GENDER_CATEGORIES)
        age_group = rng.choice(SampleDataGenerator.AGE_GROUPS)
        condition = rng.choice(conditions)

        template = rng.choice(templates.get(corpus_type, templates["peer_reviewed"]))
        content = template.format(
            condition=condition, race=race, gender=gender, age_group=age_group
        )

        data.append(
            {
                "id": f"doc_{i:05d}",
                "content": content,
                "source": corpus_type,
                "race": race,
                "gender": gender,
                "age_group": age_group,
            }
        )

    return pd.DataFrame(data)


def generate_figure_data(random_state: int = 42) -> Dict[str, Dict]:
    """Generate ready-to-use sample inputs for the manuscript figure functions.

    Returns a dict keyed by figure name ("fig2" through "fig7"), each value being
    a dict in the exact structure the matching ``plot_figure*`` function expects.
    This lets you render every figure immediately, then swap in your own data by
    matching the same keys (documented in each plot function's docstring).

    Example:
        >>> from equimed_dss.utils import generate_figure_data
        >>> from equimed_dss.utils import plot_figure2_reliability_dashboard
        >>> figs = generate_figure_data()
        >>> plot_figure2_reliability_dashboard(figs["fig2"], save_path="figures/fig2.png")
    """
    rng = np.random.RandomState(random_state)

    def _sym(n: int) -> list:
        a = rng.uniform(0, 1, (n, n))
        a = (a + a.T) / 2
        np.fill_diagonal(a, 0)
        return a.tolist()

    corpora = ["peer_reviewed", "community", "mimic_iv", "mimic"]
    metric_names = ["DFR", "ECS", "HER", "IBS"]

    return {
        "fig2": {
            "icc_scores": {"Rater 1": 0.82, "Rater 2": 0.76, "Rater 3": 0.68},
            "cronbach_alpha": {"alpha": 0.88, "ci_lower": 0.81, "ci_upper": 0.93},
            "bland_altman": {
                "means": list(rng.uniform(0.4, 0.9, 40)),
                "diffs": list(rng.normal(0, 0.05, 40)),
                "loa_upper": 0.1,
                "loa_lower": -0.1,
            },
            "temporal": {
                "timepoints": [1, 2, 3, 4, 5],
                "reliability_scores": [0.80, 0.82, 0.79, 0.85, 0.83],
            },
        },
        "fig3": {
            "bias_gini": {
                "Peer-Reviewed": 0.12, "Community": 0.21, "MIMIC-IV": 0.18,
                "peer_reviewed": 0.12, "community": 0.21, "mimic_iv": 0.18,
            },
            "temporal_drift": {
                "timepoints": [1, 2, 3, 4],
                "peer_reviewed": [0.05, 0.07, 0.06, 0.08],
                "community": [0.12, 0.15, 0.11, 0.14],
                "mimic_iv": [0.09, 0.10, 0.08, 0.11],
                "mimic": [0.09, 0.10, 0.08, 0.11],
            },
            "clinical_harm": {
                "peer_reviewed": list(rng.uniform(0, 0.20, 30)),
                "community": list(rng.uniform(0, 0.30, 30)),
                "mimic_iv": list(rng.uniform(0, 0.25, 30)),
                "mimic": list(rng.uniform(0, 0.25, 30)),
            },
            "network_stability": {
                c: {"density": 0.5, "clustering": 0.4, "modularity": 0.3}
                for c in corpora
            },
        },
        "fig4": {
            "mediation": {
                "indirect_effect": 0.14, "direct_effect": 0.36,
                "indirect_ci_lower": 0.05, "indirect_ci_upper": 0.23,
                "direct_ci_lower": 0.25, "direct_ci_upper": 0.47,
                "total_ci_lower": 0.30, "total_ci_upper": 0.70,
            },
            "regression": {
                "predictors": ["race", "age", "ses"],
                "coefficients": [0.20, -0.10, 0.15],
                "ci_lower": [0.10, -0.20, 0.05],
                "ci_upper": [0.30, 0.00, 0.25],
            },
            "robustness": {
                "perturbation_levels": [0.0, 0.1, 0.2, 0.3],
                "rcs_scores": [1.0, 0.95, 0.88, 0.80],
            },
        },
        "fig5": {
            "ethical_risk": {
                "timepoints": [1, 2, 3, 4, 5],
                "eri_scores": [0.10, 0.12, 0.09, 0.14, 0.11],
                "violations": [0, 1, 0, 2, 1],
            },
            "rams": {"risk": 0.80, "accountability": 0.75, "monitoring": 0.70, "safety": 0.85},
            "fairness_ecosystem": {
                "components": ["Reliability", "Equity", "Governance", "Transparency"],
                "connections": _sym(4),
            },
        },
        "fig6": {
            k: {"adjacency_matrix": _sym(4), "metric_names": metric_names}
            for k in ["peer_reviewed", "community", "mimic", "combined"]
        },
        "fig7": {
            "fairness_matrix": rng.uniform(0.6, 0.95, (4, 3)).tolist(),
            "row_labels": ["White", "Black", "Hispanic", "Asian"],
            "col_labels": ["Female", "Male", "Non-binary"],
            "pareto_optimal": (rng.uniform(0, 1, (4, 3)) > 0.7).tolist(),
            "marginal_row": [0.86, 0.70, 0.79, 0.88],
            "marginal_col": [0.82, 0.78, 0.79],
        },
    }
