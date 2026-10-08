"""
Advanced appendix metrics for the EquiMed-DSS suite.

These nine metrics (bootstrap confidence intervals, statistical power, bias
concentration, mutual information, Jensen-Shannon divergence, Wasserstein
distance, network modularity, transparency, robustness certification)
complement the five core domains and the geographic module (37 metrics total).
"""

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

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
        statistic: Callable = np.mean,
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

        from equimed_dss.inference import MetricResult

        return MetricResult(
            {
                "ci_lower": float(ci_lower),
                "ci_upper": float(ci_upper),
                "ci_width": float(ci_width),
                "ci_method": "bootstrap",
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
            },
            name="BCI",
            value_key="observed_statistic",
        )


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

            from equimed_dss.inference import MetricResult

            # Required sample size is an analytic design quantity, not an estimate
            # from sampled data, so it carries no sampling CI (prints "unavailable").
            return MetricResult(
                {
                    "n_per_group": int(np.ceil(n_per_group)),
                    "total_n": int(np.ceil(n_per_group * 2)),
                    "effect_size": float(effect_size),
                    "alpha": alpha,
                    "power": power,
                    "interpretation": {
                        "range": "[0, 1]",
                        "achieved_power": power,
                        "verdict": (
                            "Adequate power (>= 0.8)"
                            if power >= 0.8
                            else "Insufficient power (< 0.8)"
                        ),
                    },
                },
                name="SampleSize",
                value_key="n_per_group",
            )
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

        from equimed_dss.inference import MetricResult

        # Achieved power is an analytic function of (n, effect size, alpha); it is
        # not estimated from sampled data, so it carries no sampling CI.
        return MetricResult(
            {
                "power": float(power),
                "n_per_group": n,
                "effect_size": effect_size,
                "interpretation": {
                    "verdict": (
                        "Adequate power" if power >= 0.8 else "Insufficient power"
                    )
                },
            },
            name="Power",
            value_key="power",
        )


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

        BCI = 1 - sum(p^2) / (sum p)^2 (one minus the Herfindahl index of the
        bias shares). With n groups its maximum is 1 - 1/n (equal shares), so
        ``bci_normalized`` = BCI / (1 - 1/n) rescales it to [0, 1] and the
        verdict uses the normalized value. Up to 1.9.5 the verdict used the raw
        BCI, so with two groups even perfectly equal shares (BCI = 0.5) could
        never be classed as distributed.

        Interpretation (bci_normalized):
            - near 1: Bias distributed evenly across groups
            - near 0: Bias concentrated in specific groups (requires targeted intervention)
            - < 0.3: Concentrated bias (HIGH CONCERN)
        """
        from equimed_dss.inference import MetricResult, bootstrap_ci

        p = np.array(group_bias_proportions)
        n = len(p)

        if n == 0 or np.sum(p) == 0:
            return MetricResult(
                {"bci": 0.0, "interpretation": {"verdict": "No bias detected"}},
                name="BiasConcentration",
                value_key="bci",
            )

        def _bci(vals):
            v = np.array(vals)
            den = (np.sum(v)) ** 2
            return float(1 - (np.sum(v**2) / den)) if den != 0 else 0.0

        # BCI = 1 - (sum of squared proportions / squared sum of proportions)
        bci = _bci(p)
        bci_max = 1.0 - 1.0 / n if n > 1 else 0.0
        bci_norm = float(bci / bci_max) if bci_max > 0 else 0.0

        out = {
            "bci": float(bci),
            "bci_normalized": bci_norm,
            "bci_max": float(bci_max),
            "n_groups": n,
            "max_bias_proportion": float(np.max(p)),
            "min_bias_proportion": float(np.min(p)),
            "interpretation": {
                "range": f"[0, {bci_max:.3f}] (1 - 1/n); bci_normalized in [0, 1]",
                "distribution": (
                    "Distributed bias"
                    if bci_norm > 0.7
                    else (
                        "Moderate concentration"
                        if bci_norm > 0.3
                        else "Concentrated bias"
                    )
                ),
                "verdict": (
                    "Acceptable (distributed)"
                    if bci_norm > 0.7
                    else (
                        "Monitor (moderate concentration)"
                        if bci_norm > 0.3
                        else "Intervention required (concentrated bias)"
                    )
                ),
            },
        }

        # Percentile bootstrap over the per-group bias proportions.
        if n >= 2:
            ci = bootstrap_ci(list(p), _bci, n_boot=1000, random_state=0)
            out["ci_lower"] = ci.ci_lower
            out["ci_upper"] = ci.ci_upper
            out["ci_method"] = ci.method
        return MetricResult(out, name="BiasConcentration", value_key="bci")


class MutualInformationContent:
    """
    Appendix Metric: Mutual Information Content (MIC)

    Mutual information between demographic attributes and diagnostic outcomes,
    detecting inappropriate information leakage. This is (normalized) mutual
    information -- NOT the Reshef Maximal Information Coefficient. Raw mutual
    information is unbounded and grows with the number of categories, so prefer
    ``normalized_mic`` for comparison across settings.
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
            - 0.1 <= MIC < 0.3: Moderate leakage (investigate)
            - MIC >= 0.3: Concerning leakage (demographics influence diagnoses)
        """
        from sklearn.metrics import mutual_info_score

        demographics = np.asarray(demographics)
        outcomes = np.asarray(outcomes)
        mi = mutual_info_score(demographics, outcomes)

        # Normalize by entropy
        from scipy.stats import entropy as scipy_entropy

        # Category frequencies from the labels themselves, so string categories
        # work (np.bincount, used up to 1.9.5, accepts only non-negative ints).
        _, demo_counts = np.unique(demographics, return_counts=True)
        demo_entropy = scipy_entropy(demo_counts / len(demographics))
        normalized_mi = mi / demo_entropy if demo_entropy > 0 else 0

        from equimed_dss.inference import MetricResult, bootstrap_ci

        out = {
            "mic": float(mi),
            "normalized_mic": float(normalized_mi),
            "interpretation": {
                "range": "[0, inf)",
                "leakage_level": (
                    "Minimal" if mi < 0.1 else "Moderate" if mi < 0.3 else "Concerning"
                ),
                "verdict": (
                    "Acceptable (MIC < 0.1)"
                    if mi < 0.1
                    else (
                        "Investigate (0.1 <= MIC < 0.3)"
                        if mi < 0.3
                        else "Intervention required (MIC >= 0.3)"
                    )
                ),
            },
        }

        # Percentile bootstrap over paired (demographic, outcome) observations.
        n_obs = len(demographics)
        if n_obs >= 2:
            idx = list(range(n_obs))
            ci = bootstrap_ci(
                idx,
                lambda i: float(
                    mutual_info_score(demographics[list(i)], outcomes[list(i)])
                ),
                n_boot=1000,
                random_state=0,
            )
            out["ci_lower"] = ci.ci_lower
            out["ci_upper"] = ci.ci_upper
            out["ci_method"] = ci.method
        return MetricResult(out, name="MIC", value_key="mic")


class JensenShannonDivergence:
    """
    Appendix Metric: Jensen-Shannon Divergence (JSD)

    Measures distributional differences between demographic groups in
    model outputs (symmetric, bounded version of KL divergence). Returns the
    Jensen-Shannon DIVERGENCE in base 2 (range [0, 1]); ``jsd_distance`` is the
    corresponding metric distance (its square root).
    """

    def calculate_jsd(
        self, distribution_p: np.ndarray, distribution_q: np.ndarray
    ) -> Dict[str, Any]:
        """
        Calculate Jensen-Shannon Divergence between two distributions.

        Args:
            distribution_p: First probability distribution (or counts) over a
                fixed set of categories or bins
            distribution_q: Second distribution over the SAME categories, in the
                same order. To compare two samples of values, histogram both on
                common bins first; raw samples would be compared position by
                position, which is meaningless.

        Returns:
            JSD score and interpretation

        Interpretation:
            - JSD < 0.1: Minimal distributional difference (good)
            - 0.1 <= JSD < 0.2: Moderate difference (monitor)
            - JSD >= 0.2: Significant difference (bias concern)
        """
        # Normalize to probability distributions
        p = np.array(distribution_p) / np.sum(distribution_p)
        q = np.array(distribution_q) / np.sum(distribution_q)

        # Jensen-Shannon DIVERGENCE in base 2 (range [0, 1]). scipy's
        # jensenshannon returns the metric DISTANCE (sqrt of the divergence) in
        # the given base, so we square it; jsd_distance exposes the distance.
        jsd_distance = float(jensenshannon(p, q, base=2))
        jsd = jsd_distance**2

        from equimed_dss.inference import MetricResult

        # JSD is computed between two already-aggregated probability distributions;
        # without the underlying per-observation samples there is no sampling
        # distribution to bootstrap, so it prints "CI unavailable".
        return MetricResult(
            {
                "jsd": float(jsd),
                "jsd_distance": jsd_distance,
                "interpretation": {
                    "range": "[0, 1]",
                    "similarity": (
                        "Highly similar"
                        if jsd < 0.1
                        else (
                            "Moderately similar"
                            if jsd < 0.2
                            else "Different distributions"
                        )
                    ),
                    "verdict": (
                        "Acceptable (JSD < 0.1)"
                        if jsd < 0.1
                        else (
                            "Monitor (0.1 <= JSD < 0.2)"
                            if jsd < 0.2
                            else "Bias concern (JSD >= 0.2)"
                        )
                    ),
                },
            },
            name="JSD",
            value_key="jsd",
        )


class WassersteinDistance:
    """
    Appendix Metric: Wasserstein Distance (WD)

    Provides robust distributional comparison resistant to outliers,
    measuring optimal transport distance between distributions.

    Reference: Manuscript Equation (16)
    """

    def calculate_wd(
        self,
        distribution_p: np.ndarray,
        distribution_q: np.ndarray,
        support: Optional[Sequence[float]] = None,
    ) -> Dict[str, Any]:
        """
        Calculate Wasserstein Distance (Earth Mover's Distance).

        By default the two inputs are SAMPLES of values (e.g. predicted risks in
        two groups), not probability vectors: [0.2, 0.8] and [0.8, 0.2] are the
        same sample and have distance 0. To compare two histograms, pass the
        bin probabilities as ``distribution_p`` / ``distribution_q`` and the bin
        locations as ``support``.

        Args:
            distribution_p: First sample (or histogram weights with ``support``)
            distribution_q: Second sample (or histogram weights with ``support``)
            support: optional bin locations shared by both histograms

        Returns:
            WD score and interpretation

        Interpretation (heuristic, for values on a 0-1 scale such as risks):
            - WD < 0.1: Minimal difference (equitable)
            - 0.1 <= WD < 0.25: Moderate difference (monitor)
            - WD >= 0.25: Substantial difference (calibration needed)
        """
        from equimed_dss.inference import MetricResult

        p_arr = np.asarray(distribution_p, dtype=float)
        q_arr = np.asarray(distribution_q, dtype=float)
        if support is not None:
            x = np.asarray(support, dtype=float)
            if not (x.shape == p_arr.shape == q_arr.shape):
                raise ValueError(
                    "support, distribution_p and distribution_q must have the same length."
                )
            wd = wasserstein_distance(x, x, p_arr, q_arr)
            return MetricResult(
                {
                    "wasserstein_distance": float(wd),
                    "input": "histograms on a shared support",
                    "interpretation": {
                        "range": "[0, inf), in the units of the support",
                        "note": "No CI: the inputs are aggregated histograms.",
                    },
                },
                name="WD",
                value_key="wasserstein_distance",
            )
        wd = wasserstein_distance(p_arr, q_arr)

        out = {
            "wasserstein_distance": float(wd),
            "interpretation": {
                "range": "[0, inf)",
                "difference_level": (
                    "Minimal"
                    if wd < 0.1
                    else "Moderate" if wd < 0.25 else "Substantial"
                ),
                "verdict": (
                    "Equitable (WD < 0.1)"
                    if wd < 0.1
                    else (
                        "Monitor (0.1 <= WD < 0.25)"
                        if wd < 0.25
                        else "Calibration needed (WD >= 0.25)"
                    )
                ),
            },
        }

        # The two inputs are treated as samples by scipy's Wasserstein distance,
        # so a percentile bootstrap that resamples each sample independently gives
        # an honest CI for the distance.
        if p_arr.size >= 2 and q_arr.size >= 2:
            rng = np.random.default_rng(0)
            boots = []
            for _ in range(1000):
                bp = p_arr[rng.integers(0, p_arr.size, size=p_arr.size)]
                bq = q_arr[rng.integers(0, q_arr.size, size=q_arr.size)]
                boots.append(float(wasserstein_distance(bp, bq)))
            lo, hi = np.percentile(boots, [2.5, 97.5])
            out["ci_lower"] = float(lo)
            out["ci_upper"] = float(hi)
            out["ci_method"] = "bootstrap"
        return MetricResult(out, name="WD", value_key="wasserstein_distance")


class NetworkModularity:
    """
    Appendix Metric: Network Modularity (NM)

    Identifies clustered relationships among fairness metrics within
    each corpus using community detection (Newman modularity Q over greedy
    Clauset-Newman-Moore communities).
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
            - 0.1 < Modularity <= 0.3: Moderate clustering
            - Modularity <= 0.1: Weak clustering
        """
        from equimed_dss.inference import MetricResult

        # Absolute weights, and no self-loops: the diagonal of a correlation
        # matrix (1) is not an edge. Communities are found and scored with the
        # same edge weights. (Up to 1.9.5 the diagonal was kept and the greedy
        # search ignored the weights while the score used them.)
        A = np.abs(np.asarray(adjacency_matrix, dtype=float))
        np.fill_diagonal(A, 0.0)
        G = nx.from_numpy_array(A)

        # Detect communities with greedy modularity (Clauset-Newman-Moore)
        try:
            from networkx.algorithms.community import (
                greedy_modularity_communities,
                modularity,
            )

            if G.number_of_edges() == 0:
                raise ValueError("the network has no edges")
            communities = list(greedy_modularity_communities(G, weight="weight"))
            Q = modularity(G, communities, weight="weight")

            def _modularity_of(sub_A):
                sub_A = sub_A.copy()
                np.fill_diagonal(sub_A, 0.0)
                gg = nx.from_numpy_array(sub_A)
                comms = list(greedy_modularity_communities(gg, weight="weight"))
                return float(modularity(gg, comms, weight="weight"))

            # Node-resampling bootstrap: resample node indices with replacement and
            # recompute modularity on the induced subgraph, giving a stability CI.
            n_nodes = A.shape[0]
            ci_lower = ci_upper = None
            ci_method = None
            if n_nodes >= 3:
                rng = np.random.default_rng(0)
                boots = []
                for _ in range(200):
                    idx = rng.integers(0, n_nodes, size=n_nodes)
                    try:
                        boots.append(_modularity_of(A[np.ix_(idx, idx)]))
                    except Exception:
                        continue
                if boots:
                    lo, hi = np.percentile(boots, [2.5, 97.5])
                    ci_lower, ci_upper, ci_method = float(lo), float(hi), "bootstrap"

            res = {
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
            if ci_lower is not None:
                res["ci_lower"] = ci_lower
                res["ci_upper"] = ci_upper
                res["ci_method"] = ci_method
            return MetricResult(res, name="NM", value_key="modularity")
        except Exception as e:
            import warnings

            warnings.warn(
                f"Modularity could not be computed ({e}); returning NaN.",
                UserWarning,
                stacklevel=2,
            )
            return MetricResult(
                {
                    "modularity": float("nan"),
                    "error": str(e),
                    "interpretation": {"verdict": "Unable to compute modularity"},
                },
                name="NM",
                value_key="modularity",
            )


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
            - 0.5 < TS <= 0.7: Moderate transparency (improvement needed)
            - TS <= 0.5: Poor transparency (not ready for deployment)
        """
        from equimed_dss.inference import MetricResult, bootstrap_ci

        if not explanations:
            return MetricResult(
                {
                    "ts": 0.0,
                    "interpretation": {"verdict": "No explanations provided"},
                },
                name="TS",
                value_key="ts",
            )

        scores = []
        for exp in explanations:
            e = exp.get("explanation_quality", 0)
            f = exp.get("feature_importance", 0)
            i = exp.get("interpretability", 0)
            avg_score = (e + f + i) / 3
            scores.append(avg_score)

        ts = np.mean(scores)

        out = {
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
                        "Needs improvement (0.5 < TS <= 0.7)"
                        if ts > 0.5
                        else "Not ready for deployment (TS <= 0.5)"
                    )
                ),
            },
        }

        # TS is a mean of per-decision [0, 1] transparency scores; a percentile
        # bootstrap over decisions gives its 95% CI.
        if len(scores) >= 2:
            ci = bootstrap_ci(
                list(scores), lambda s: float(np.mean(s)), n_boot=1000, random_state=0
            )
            out["ci_lower"] = ci.ci_lower
            out["ci_upper"] = ci.ci_upper
            out["ci_method"] = ci.method
        return MetricResult(out, name="TS", value_key="ts")


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
            epsilon: Perturbation bound (recorded in the output only; it does
                not enter the score, which is the mean agreement)

        Returns:
            RCS score and interpretation

        Interpretation:
            - RCS > 0.8: Robust performance (clinical deployment ready)
            - 0.6 < RCS <= 0.8: Moderate robustness (monitor closely)
            - RCS <= 0.6: Poor robustness (requires improvement)
        """
        from equimed_dss.inference import MetricResult, bootstrap_ci

        if not perturbed_predictions:
            return MetricResult(
                {
                    "rcs": 0.0,
                    "interpretation": {"verdict": "No perturbations provided"},
                },
                name="RCS",
                value_key="rcs",
            )

        # Element-wise agreement. Inputs are converted to arrays: with plain
        # lists, == compared whole lists (one True/False), so up to 1.9.5 any
        # single difference gave an agreement of 0 for that perturbation.
        original = np.asarray(original_predictions)
        consistency_scores = []
        for perturbed in perturbed_predictions:
            perturbed = np.asarray(perturbed)
            if perturbed.shape != original.shape:
                raise ValueError(
                    "Each perturbed prediction set must have the shape of "
                    f"original_predictions {original.shape}; got {perturbed.shape}."
                )
            agreement = np.mean(original == perturbed)
            consistency_scores.append(agreement)

        rcs = np.mean(consistency_scores)
        rcs_std = np.std(consistency_scores)

        out = {
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
                        "Monitor closely (0.6 < RCS <= 0.8)"
                        if rcs > 0.6
                        else "Requires improvement (RCS <= 0.6)"
                    )
                ),
            },
        }

        # RCS is the mean per-perturbation agreement; a percentile bootstrap over
        # perturbations gives its 95% CI.
        if len(consistency_scores) >= 2:
            ci = bootstrap_ci(
                list(consistency_scores),
                lambda s: float(np.mean(s)),
                n_boot=1000,
                random_state=0,
            )
            out["ci_lower"] = ci.ci_lower
            out["ci_upper"] = ci.ci_upper
            out["ci_method"] = ci.method
        return MetricResult(out, name="RCS", value_key="rcs")
