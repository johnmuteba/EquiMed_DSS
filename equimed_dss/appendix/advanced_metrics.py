"""
Advanced metrics from Appendix A of the EquiMed_DSS manuscript.

These 9 additional metrics complete the comprehensive 19-metric evaluation suite
for clinical AI fairness, reliability, and governance assessment.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import networkx as nx
import numpy as np
from scipy import stats
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance


class BootstrapConfidenceIntervals:
    """
    Appendix Metric: Bootstrap Confidence Intervals (BCI)

    Provides robust confidence intervals for performance metrics without
    distributional assumptions using bootstrap resampling.

    Reference: Manuscript Equation (11)
    """

    def __init__(self, n_bootstrap: int = 1000, random_state: Optional[int] = None):
        """
        Initialize Bootstrap CI calculator.

        Args:
            n_bootstrap: Number of bootstrap samples (default: 1000)
            random_state: Random seed for reproducibility
        """
        self.n_bootstrap = n_bootstrap
        self.rng = np.random.RandomState(random_state)

    def calculate_bci(
        self,
        data: np.ndarray,
        statistic: callable = np.mean,
        alpha: float = 0.05,
        method: str = "percentile",
    ) -> Dict[str, Any]:
        """
        Calculate bootstrap confidence intervals.

        Args:
            data: Input data array
            statistic: Function to compute statistic (default: mean)
            alpha: Significance level (default: 0.05 for 95% CI)
            method: 'percentile' or 'bca' (bias-corrected accelerated)

        Returns:
            Dictionary with CI bounds and bootstrap estimates

        Interpretation:
            - Narrow CI width (< 0.05): Stable, reliable performance
            - Wide CI: High uncertainty, requires more data
        """
        n = len(data)
        bootstrap_estimates = []

        # Generate bootstrap samples
        for _ in range(self.n_bootstrap):
            sample = self.rng.choice(data, size=n, replace=True)
            bootstrap_estimates.append(statistic(sample))

        bootstrap_estimates = np.array(bootstrap_estimates)

        # Calculate confidence intervals
        if method == "percentile":
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100
            ci_lower = np.percentile(bootstrap_estimates, lower_percentile)
            ci_upper = np.percentile(bootstrap_estimates, upper_percentile)
        else:
            raise ValueError(f"Method '{method}' not supported. Use 'percentile'.")

        observed_statistic = statistic(data)
        ci_width = ci_upper - ci_lower

        return {
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
            "ci_width": float(ci_width),
            "observed_statistic": float(observed_statistic),
            "bootstrap_mean": float(np.mean(bootstrap_estimates)),
            "bootstrap_std": float(np.std(bootstrap_estimates)),
            "n_bootstrap": self.n_bootstrap,
            "interpretation": {
                "range": f"[{ci_lower:.4f}, {ci_upper:.4f}]",
                "stability": "Stable" if ci_width < 0.05 else "Unstable",
                "verdict": (
                    "Excellent reliability (CI width < 0.05)"
                    if ci_width < 0.05
                    else (
                        "Acceptable reliability"
                        if ci_width < 0.1
                        else "Poor reliability (wide CI)"
                    )
                ),
            },
        }


class StatisticalPowerAnalysis:
    """
    Appendix Metric: Statistical Power Analysis (SPA)

    Determines minimum sample sizes needed to detect clinically meaningful
    differences between demographic groups with adequate statistical power.

    Reference: Manuscript Equation (12)
    """

    def calculate_sample_size(
        self,
        effect_size: float,
        alpha: float = 0.05,
        power: float = 0.8,
        alternative: str = "two-sided",
    ) -> Dict[str, Any]:
        """
        Calculate required sample size for given effect size and power.

        Args:
            effect_size: Cohen's d effect size
            alpha: Type I error rate (default: 0.05)
            power: Desired statistical power (default: 0.8)
            alternative: 'two-sided' or 'one-sided'

        Returns:
            Dictionary with required sample size and power analysis results

        Interpretation:
            - Power > 0.8: Adequate sensitivity to detect bias
            - Power < 0.8: Insufficient power, may miss important disparities
        """
        from statsmodels.stats.power import tt_ind_solve_power

        try:
            n_per_group = tt_ind_solve_power(
                effect_size=effect_size,
                alpha=alpha,
                power=power,
                alternative=alternative,
            )

            return {
                "n_per_group": int(np.ceil(n_per_group)),
                "total_n": int(np.ceil(n_per_group * 2)),
                "effect_size": float(effect_size),
                "alpha": alpha,
                "power": power,
                "interpretation": {
                    "range": "[0, 1]",
                    "achieved_power": power,
                    "verdict": (
                        "Adequate power (≥ 0.8)"
                        if power >= 0.8
                        else "Insufficient power (< 0.8)"
                    ),
                },
            }
        except Exception as e:
            return {
                "error": str(e),
                "effect_size": effect_size,
                "recommendation": "Consider increasing sample size or effect size",
            }

    def calculate_power(
        self, n: int, effect_size: float, alpha: float = 0.05
    ) -> Dict[str, Any]:
        """
        Calculate statistical power for given sample size.

        Args:
            n: Sample size per group
            effect_size: Cohen's d effect size
            alpha: Type I error rate

        Returns:
            Power analysis results
        """
        from statsmodels.stats.power import tt_ind_solve_power

        power = tt_ind_solve_power(
            effect_size=effect_size, nobs1=n, alpha=alpha, alternative="two-sided"
        )

        return {
            "power": float(power),
            "n_per_group": n,
            "effect_size": effect_size,
            "interpretation": {
                "verdict": ("Adequate power" if power >= 0.8 else "Insufficient power")
            },
        }


class BiasConcentrationIndex:
    """
    Appendix Metric: Bias Concentration Index (BCI)

    Measures whether bias affects all groups equally or concentrates
    in specific populations.

    Reference: Manuscript Equation (13)
    """

    def calculate_bci(
        self, group_bias_proportions: Union[List[float], np.ndarray]
    ) -> Dict[str, Any]:
        """
        Calculate Bias Concentration Index.

        Args:
            group_bias_proportions: Proportion of bias in each demographic group

        Returns:
            BCI score and interpretation

        Interpretation:
            - BCI near 1: Bias distributed evenly across groups
            - BCI near 0: Bias concentrated in specific groups (requires targeted intervention)
            - BCI < 0.3: Concentrated bias (HIGH CONCERN)
        """
        p = np.array(group_bias_proportions)
        n = len(p)

        if n == 0 or np.sum(p) == 0:
            return {"bci": 0.0, "interpretation": {"verdict": "No bias detected"}}

        # BCI = 1 - (sum of squared proportions / squared sum of proportions)
        numerator = np.sum(p**2)
        denominator = (np.sum(p)) ** 2

        bci = 1 - (numerator / denominator) if denominator != 0 else 0

        return {
            "bci": float(bci),
            "n_groups": n,
            "max_bias_proportion": float(np.max(p)),
            "min_bias_proportion": float(np.min(p)),
            "interpretation": {
                "range": "[0, 1]",
                "distribution": (
                    "Distributed bias"
                    if bci > 0.7
                    else "Moderate concentration" if bci > 0.3 else "Concentrated bias"
                ),
                "verdict": (
                    "Acceptable (distributed)"
                    if bci > 0.7
                    else (
                        "Monitor (moderate concentration)"
                        if bci > 0.3
                        else "Intervention required (concentrated bias)"
                    )
                ),
            },
        }


class MutualInformationContent:
    """
    Appendix Metric: Mutual Information Content (MIC)

    Quantifies information shared between demographic attributes and
    diagnostic outcomes, detecting inappropriate information leakage.

    Reference: Manuscript Equation (14)
    """

    def calculate_mic(
        self, demographics: np.ndarray, outcomes: np.ndarray
    ) -> Dict[str, Any]:
        """
        Calculate Mutual Information Content.

        Args:
            demographics: Array of demographic categories
            outcomes: Array of model outcomes/predictions

        Returns:
            MIC score and interpretation

        Interpretation:
            - MIC < 0.1: Minimal information leakage (good)
            - 0.1 ≤ MIC < 0.3: Moderate leakage (investigate)
            - MIC ≥ 0.3: Concerning leakage (demographics influence diagnoses)
        """
        from sklearn.metrics import mutual_info_score

        mi = mutual_info_score(demographics, outcomes)

        # Normalize by entropy
        from scipy.stats import entropy as scipy_entropy

        demo_entropy = scipy_entropy(np.bincount(demographics) / len(demographics))
        normalized_mi = mi / demo_entropy if demo_entropy > 0 else 0

        return {
            "mic": float(mi),
            "normalized_mic": float(normalized_mi),
            "interpretation": {
                "range": "[0, ∞)",
                "leakage_level": (
                    "Minimal" if mi < 0.1 else "Moderate" if mi < 0.3 else "Concerning"
                ),
                "verdict": (
                    "Acceptable (MIC < 0.1)"
                    if mi < 0.1
                    else (
                        "Investigate (0.1 ≤ MIC < 0.3)"
                        if mi < 0.3
                        else "Intervention required (MIC ≥ 0.3)"
                    )
                ),
            },
        }


class JensenShannonDivergence:
    """
    Appendix Metric: Jensen-Shannon Divergence (JSD)

    Measures distributional differences between demographic groups in
    model outputs (symmetric, bounded version of KL divergence).

    Reference: Manuscript Equation (15)
    """

    def calculate_jsd(
        self, distribution_p: np.ndarray, distribution_q: np.ndarray
    ) -> Dict[str, Any]:
        """
        Calculate Jensen-Shannon Divergence between two distributions.

        Args:
            distribution_p: First probability distribution
            distribution_q: Second probability distribution

        Returns:
            JSD score and interpretation

        Interpretation:
            - JSD < 0.1: Minimal distributional difference (good)
            - 0.1 ≤ JSD < 0.2: Moderate difference (monitor)
            - JSD ≥ 0.2: Significant difference (bias concern)
        """
        # Normalize to probability distributions
        p = np.array(distribution_p) / np.sum(distribution_p)
        q = np.array(distribution_q) / np.sum(distribution_q)

        jsd = jensenshannon(p, q)

        return {
            "jsd": float(jsd),
            "jsd_squared": float(jsd**2),
            "interpretation": {
                "range": "[0, 1]",
                "similarity": (
                    "Highly similar"
                    if jsd < 0.1
                    else (
                        "Moderately similar" if jsd < 0.2 else "Different distributions"
                    )
                ),
                "verdict": (
                    "Acceptable (JSD < 0.1)"
                    if jsd < 0.1
                    else (
                        "Monitor (0.1 ≤ JSD < 0.2)"
                        if jsd < 0.2
                        else "Bias concern (JSD ≥ 0.2)"
                    )
                ),
            },
        }


class WassersteinDistance:
    """
    Appendix Metric: Wasserstein Distance (WD)

    Provides robust distributional comparison resistant to outliers,
    measuring optimal transport distance between distributions.

    Reference: Manuscript Equation (16)
    """

    def calculate_wd(
        self, distribution_p: np.ndarray, distribution_q: np.ndarray
    ) -> Dict[str, Any]:
        """
        Calculate Wasserstein Distance (Earth Mover's Distance).

        Args:
            distribution_p: First distribution
            distribution_q: Second distribution

        Returns:
            WD score and interpretation

        Interpretation:
            - WD < 0.1: Minimal difference (equitable)
            - 0.1 ≤ WD < 0.25: Moderate difference (monitor)
            - WD ≥ 0.25: Substantial difference (calibration needed)
        """
        wd = wasserstein_distance(distribution_p, distribution_q)

        return {
            "wasserstein_distance": float(wd),
            "interpretation": {
                "range": "[0, ∞)",
                "difference_level": (
                    "Minimal"
                    if wd < 0.1
                    else "Moderate" if wd < 0.25 else "Substantial"
                ),
                "verdict": (
                    "Equitable (WD < 0.1)"
                    if wd < 0.1
                    else (
                        "Monitor (0.1 ≤ WD < 0.25)"
                        if wd < 0.25
                        else "Calibration needed (WD ≥ 0.25)"
                    )
                ),
            },
        }


class NetworkModularity:
    """
    Appendix Metric: Network Modularity (NM)

    Identifies clustered relationships among fairness metrics within
    each corpus using community detection.

    Reference: Manuscript Equation (17)
    """

    def calculate_modularity(self, adjacency_matrix: np.ndarray) -> Dict[str, Any]:
        """
        Calculate network modularity from correlation matrix.

        Args:
            adjacency_matrix: Adjacency/correlation matrix of metrics

        Returns:
            Modularity score and community structure

        Interpretation:
            - Modularity > 0.3: Strong clustering (coherent metric relationships)
            - 0.1 < Modularity ≤ 0.3: Moderate clustering
            - Modularity ≤ 0.1: Weak clustering
        """
        G = nx.from_numpy_array(np.abs(adjacency_matrix))

        # Detect communities using Louvain method
        try:
            from networkx.algorithms.community import (
                greedy_modularity_communities, modularity)

            communities = list(greedy_modularity_communities(G))
            Q = modularity(G, communities)

            return {
                "modularity": float(Q),
                "n_communities": len(communities),
                "community_sizes": [len(c) for c in communities],
                "interpretation": {
                    "range": "[-1, 1]",
                    "clustering_strength": (
                        "Strong" if Q > 0.3 else "Moderate" if Q > 0.1 else "Weak"
                    ),
                    "verdict": (
                        "Excellent (Q > 0.3)"
                        if Q > 0.3
                        else "Acceptable (Q > 0.1)" if Q > 0.1 else "Weak structure"
                    ),
                },
            }
        except Exception as e:
            return {
                "modularity": 0.0,
                "error": str(e),
                "interpretation": {"verdict": "Unable to compute modularity"},
            }


class TransparencyScore:
    """
    Appendix Metric: Transparency Score (TS)

    Measures clinician ability to understand AI reasoning through
    explanation quality, feature importance, and interpretability.

    Reference: Manuscript Equation (18)
    """

    def calculate_ts(self, explanations: List[Dict[str, float]]) -> Dict[str, Any]:
        """
        Calculate Transparency Score.

        Args:
            explanations: List of dicts with keys:
                'explanation_quality' (0-1)
                'feature_importance' (0-1)
                'interpretability' (0-1)

        Returns:
            TS score and interpretation

        Interpretation:
            - TS > 0.7: Adequate transparency for clinical use
            - 0.5 < TS ≤ 0.7: Moderate transparency (improvement needed)
            - TS ≤ 0.5: Poor transparency (not ready for deployment)
        """
        if not explanations:
            return {
                "ts": 0.0,
                "interpretation": {"verdict": "No explanations provided"},
            }

        scores = []
        for exp in explanations:
            e = exp.get("explanation_quality", 0)
            f = exp.get("feature_importance", 0)
            i = exp.get("interpretability", 0)
            avg_score = (e + f + i) / 3
            scores.append(avg_score)

        ts = np.mean(scores)

        return {
            "ts": float(ts),
            "n_decisions": len(explanations),
            "mean_explanation_quality": float(
                np.mean([e.get("explanation_quality", 0) for e in explanations])
            ),
            "mean_feature_importance": float(
                np.mean([e.get("feature_importance", 0) for e in explanations])
            ),
            "mean_interpretability": float(
                np.mean([e.get("interpretability", 0) for e in explanations])
            ),
            "interpretation": {
                "range": "[0, 1]",
                "transparency_level": (
                    "Adequate" if ts > 0.7 else "Moderate" if ts > 0.5 else "Poor"
                ),
                "verdict": (
                    "Clinical deployment ready (TS > 0.7)"
                    if ts > 0.7
                    else (
                        "Needs improvement (0.5 < TS ≤ 0.7)"
                        if ts > 0.5
                        else "Not ready for deployment (TS ≤ 0.5)"
                    )
                ),
            },
        }


class RobustnessCertificationScore:
    """
    Appendix Metric: Robustness Certification Score (RCS)

    Quantifies model stability under input variations typical in
    clinical practice (different documentation styles, measurement errors).

    Reference: Manuscript Equation (19)
    """

    def calculate_rcs(
        self,
        original_predictions: np.ndarray,
        perturbed_predictions: List[np.ndarray],
        epsilon: float = 0.1,
    ) -> Dict[str, Any]:
        """
        Calculate Robustness Certification Score.

        Args:
            original_predictions: Original model predictions
            perturbed_predictions: List of predictions under perturbations
            epsilon: Perturbation bound

        Returns:
            RCS score and interpretation

        Interpretation:
            - RCS > 0.8: Robust performance (clinical deployment ready)
            - 0.6 < RCS ≤ 0.8: Moderate robustness (monitor closely)
            - RCS ≤ 0.6: Poor robustness (requires improvement)
        """
        if not perturbed_predictions:
            return {
                "rcs": 0.0,
                "interpretation": {"verdict": "No perturbations provided"},
            }

        consistency_scores = []
        for perturbed in perturbed_predictions:
            # Calculate agreement between original and perturbed
            agreement = np.mean(original_predictions == perturbed)
            consistency_scores.append(agreement)

        rcs = np.mean(consistency_scores)
        rcs_std = np.std(consistency_scores)

        return {
            "rcs": float(rcs),
            "rcs_std": float(rcs_std),
            "n_perturbations": len(perturbed_predictions),
            "epsilon": epsilon,
            "min_consistency": float(np.min(consistency_scores)),
            "max_consistency": float(np.max(consistency_scores)),
            "interpretation": {
                "range": "[0, 1]",
                "robustness_level": (
                    "Robust" if rcs > 0.8 else "Moderate" if rcs > 0.6 else "Poor"
                ),
                "verdict": (
                    "Clinical deployment ready (RCS > 0.8)"
                    if rcs > 0.8
                    else (
                        "Monitor closely (0.6 < RCS ≤ 0.8)"
                        if rcs > 0.6
                        else "Requires improvement (RCS ≤ 0.6)"
                    )
                ),
            },
        }
