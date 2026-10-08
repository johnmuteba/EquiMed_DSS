"""
Advanced appendix metrics for the EquiMed-DSS suite.

These nine metrics (bootstrap confidence intervals, statistical power, bias
concentration, mutual information, Jensen-Shannon divergence, Wasserstein
distance, network modularity, transparency, robustness certification)
complement the five core domains and the geographic module (37 metrics total).
"""

import warnings
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

    See docs/Metric_Math_Derivations.md, Metric 29.
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
                        "Narrow interval (width < 0.05)"
                        if ci_width < 0.05
                        else (
                            "Moderate interval width (< 0.1)"
                            if ci_width < 0.1
                            else "Wide interval (>= 0.1)"
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

    See docs/Metric_Math_Derivations.md, Metric 30.
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
                    # Two equal groups: the total is twice the rounded-up group
                    # size (up to 1.9.5, ceil(2n) could be one short of it).
                    "total_n": 2 * int(np.ceil(n_per_group)),
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

    Measures whether bias is spread evenly across groups or concentrated in a
    few. It describes the DISTRIBUTION of bias, not its amount: an even spread of
    a large bias scores as "evenly distributed".

    See docs/Metric_Math_Derivations.md, Metric 31.
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
            - near 1: bias spread evenly across groups
            - near 0: bias concentrated in a few groups

        No confidence interval is reported: the inputs are one fixed value per
        group, not a sample (up to 1.9.5 they were resampled as if they were).

        Raises:
            ValueError: for negative or non-finite proportions.
        """
        from equimed_dss.inference import MetricResult

        p = np.array(group_bias_proportions, dtype=float)
        n = len(p)
        if not np.all(np.isfinite(p)) or np.any(p < 0):
            raise ValueError("group_bias_proportions must be finite and non-negative.")

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
                    "Evenly distributed (normalized > 0.7)"
                    if bci_norm > 0.7
                    else (
                        "Moderately concentrated (normalized 0.3 to 0.7)"
                        if bci_norm > 0.3
                        else "Concentrated (normalized <= 0.3)"
                    )
                ),
                "note": "Describes how bias is distributed, not how large it is.",
            },
        }
        return MetricResult(out, name="BiasConcentration", value_key="bci")


class MutualInformationContent:
    """
    Appendix Metric: Mutual Information Content (MIC)

    Mutual information between demographic categories and DISCRETE outcomes
    (for example a decision or a risk category). This is (normalized) mutual
    information, NOT the Reshef Maximal Information Coefficient. Raw mutual
    information is unbounded, grows with the number of categories and is biased
    upward in small samples, so the result reports a permutation null.

    An association can reflect clinical need or case mix rather than an
    inappropriate use of demographics; assess clinically relevant adjustment
    before interpreting it as bias.
    """

    def calculate_mic(
        self, demographics: np.ndarray, outcomes: np.ndarray
    ) -> Dict[str, Any]:
        """
        Calculate Mutual Information Content.

        Args:
            demographics: Array of demographic categories
            outcomes: Array of DISCRETE model outcomes (labels, decisions or
                binned scores). Continuous values must be binned first: when
                every value is unique, mutual information approaches the
                demographic entropy simply because each value identifies a row.

        Returns:
            MIC (nats), normalized MIC, a 95% bootstrap CI, and a permutation
            null: ``mic_null_mean`` (the value expected with no association, the
            small-sample bias) and ``p_value_permutation`` (200 shuffles of the
            outcomes, seed 0).

        Raises:
            ValueError: for mismatched or empty inputs, missing values, or
                non-integer numeric outcomes.
        """
        import pandas as pd
        from sklearn.metrics import mutual_info_score

        demographics = np.asarray(demographics)
        outcomes = np.asarray(outcomes)
        if len(demographics) != len(outcomes) or len(outcomes) == 0:
            raise ValueError("demographics and outcomes must be non-empty and paired.")
        if (
            pd.isna(pd.Series(list(demographics))).any()
            or pd.isna(pd.Series(list(outcomes))).any()
        ):
            raise ValueError(
                "demographics and outcomes must not contain missing values."
            )
        if np.issubdtype(outcomes.dtype, np.floating) and not np.all(
            outcomes == np.round(outcomes)
        ):
            raise ValueError(
                "outcomes look continuous (non-integer numbers); bin them into "
                "categories before computing mutual information."
            )
        n_obs = len(outcomes)
        if len(np.unique(outcomes)) > n_obs / 2:
            warnings.warn(
                "Most outcome values are unique; mutual information is then inflated "
                "towards the demographic entropy. Use coarser outcome categories.",
                UserWarning,
                stacklevel=2,
            )
        mi = mutual_info_score(demographics, outcomes)

        # Normalize by entropy
        from scipy.stats import entropy as scipy_entropy

        # Category frequencies from the labels themselves, so string categories
        # work (np.bincount, used up to 1.9.5, accepts only non-negative ints).
        _, demo_counts = np.unique(demographics, return_counts=True)
        demo_entropy = scipy_entropy(demo_counts / len(demographics))
        normalized_mi = mi / demo_entropy if demo_entropy > 0 else 0

        from equimed_dss.inference import MetricResult, bootstrap_ci

        rng = np.random.default_rng(0)
        null = np.array(
            [
                mutual_info_score(demographics, rng.permutation(outcomes))
                for _ in range(200)
            ]
        )
        p_perm = float((1 + np.sum(null >= mi - 1e-12)) / (1 + len(null)))
        association = (
            "above the permutation null (p < 0.05)"
            if p_perm < 0.05
            else "not distinguishable from the permutation null"
        )
        out = {
            "mic": float(mi),
            "normalized_mic": float(normalized_mi),
            "mic_null_mean": float(null.mean()),
            "p_value_permutation": p_perm,
            "interpretation": {
                "range": "[0, inf) nats",
                "leakage_level": association,
                "verdict": association,
                "note": (
                    "An association may reflect clinical need or case mix; adjust "
                    "for clinically relevant factors before calling it bias."
                ),
            },
        }

        # Percentile bootstrap over paired (demographic, outcome) observations.
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
        p = np.asarray(distribution_p, dtype=float)
        q = np.asarray(distribution_q, dtype=float)
        if p.ndim != 1 or p.shape != q.shape or p.size == 0:
            raise ValueError(
                "distributions must be non-empty 1D arrays of equal length."
            )
        for name, arr in (("distribution_p", p), ("distribution_q", q)):
            if not np.all(np.isfinite(arr)) or np.any(arr < 0) or arr.sum() <= 0:
                raise ValueError(
                    f"{name} must be finite, non-negative and have a positive sum."
                )
        # Normalize to probability distributions
        p = p / p.sum()
        q = q / q.sum()

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
                        "Below 0.1 (heuristic cut-off)"
                        if jsd < 0.1
                        else (
                            "Between 0.1 and 0.2 (heuristic cut-offs)"
                            if jsd < 0.2
                            else "At least 0.2 (heuristic cut-off)"
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

    Optimal-transport (earth mover's) distance between two samples of values,
    in the units of those values.

    See docs/Metric_Math_Derivations.md, Metric 34.
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

        Interpretation: the distance is in the units of the inputs, so there is
        no universal cut-off; compare it with a difference that matters
        clinically, in the same units.
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
                "range": "[0, inf), in the units of the inputs",
                "difference_level": "not graded (depends on the units)",
                "verdict": (
                    "No universal threshold: compare with a clinically meaningful "
                    "difference in the same units."
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

    def calculate_modularity(
        self,
        adjacency_matrix: np.ndarray,
        observations: Optional[np.ndarray] = None,
        n_boot: int = 200,
    ) -> Dict[str, Any]:
        """
        Calculate network modularity from a correlation (or adjacency) matrix.

        Args:
            adjacency_matrix: Adjacency/correlation matrix of metrics
            observations: optional raw data behind a correlation matrix, shape
                (n_observations, n_metrics). When given, a 95% CI is computed by
                resampling OBSERVATIONS, recomputing the absolute correlation
                matrix and its modularity. Without it no CI is reported: the
                matrix is a summary, and resampling metric nodes (as versions up
                to 1.9.5 did) does not describe sampling uncertainty.
            n_boot: bootstrap replicates when ``observations`` is given.

        Returns:
            Modularity score and community structure

        Interpretation (conventional, Newman):
            - Modularity > 0.3: Strong community structure
            - 0.1 < Modularity <= 0.3: Moderate community structure
            - Modularity <= 0.1: Weak community structure
        """
        from networkx.algorithms.community import (
            greedy_modularity_communities,
            modularity,
        )

        from equimed_dss.inference import MetricResult

        def _q(mat):
            # Absolute weights, and no self-loops: the diagonal of a correlation
            # matrix (1) is not an edge. Communities are found and scored with
            # the same edge weights. (Up to 1.9.5 the diagonal was kept and the
            # greedy search ignored the weights while the score used them.)
            A = np.abs(np.asarray(mat, dtype=float))
            np.fill_diagonal(A, 0.0)
            G = nx.from_numpy_array(A)
            if G.number_of_edges() == 0:
                raise ValueError("the network has no edges")
            comms = list(greedy_modularity_communities(G, weight="weight"))
            return float(modularity(G, comms, weight="weight")), comms

        try:
            Q, communities = _q(adjacency_matrix)
        except Exception as e:
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
                    "Strong community structure (Q > 0.3)"
                    if Q > 0.3
                    else (
                        "Moderate community structure (Q > 0.1)"
                        if Q > 0.1
                        else "Weak community structure"
                    )
                ),
            },
        }

        if observations is not None:
            X = np.asarray(observations, dtype=float)
            k = np.asarray(adjacency_matrix).shape[0]
            if X.ndim != 2 or X.shape[1] != k or X.shape[0] < 3:
                raise ValueError(
                    "observations must have shape (n_observations >= 3, n_metrics)."
                )
            rng = np.random.default_rng(0)
            boots, failed = [], 0
            for _ in range(int(n_boot)):
                idx = rng.integers(0, X.shape[0], size=X.shape[0])
                with np.errstate(invalid="ignore", divide="ignore"):
                    C = np.corrcoef(X[idx], rowvar=False)
                if not np.all(np.isfinite(C)):
                    failed += 1
                    continue
                try:
                    boots.append(_q(C)[0])
                except ValueError:
                    failed += 1
            if len(boots) >= 2:
                lo, hi = np.percentile(boots, [2.5, 97.5])
                res["ci_lower"] = float(lo)
                res["ci_upper"] = float(hi)
                res["ci_method"] = "bootstrap (observations)"
                res["n_boot_failed"] = failed
        return MetricResult(res, name="NM", value_key="modularity")


class TransparencyScore:
    """
    Appendix Metric: Transparency Score (TS)

    The mean of three ratings supplied by the user (explanation quality, feature
    importance, interpretability), each in [0, 1]. It reflects transparency only
    as well as those ratings do and does not establish readiness for clinical
    use.

    See docs/Metric_Math_Derivations.md, Metric 36.
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

        Interpretation (heuristic cut-offs): above 0.7, between 0.5 and 0.7, at
        most 0.5.

        Raises:
            ValueError: if no explanations are given, a rating is missing (up to
                1.9.5 a missing rating silently counted as 0) or a rating is
                outside [0, 1].
        """
        from equimed_dss.inference import MetricResult, bootstrap_ci

        if not explanations:
            raise ValueError("explanations is empty, so TS is undefined.")
        keys = ("explanation_quality", "feature_importance", "interpretability")
        for j, exp in enumerate(explanations):
            for k in keys:
                if k not in exp:
                    raise ValueError(f"explanation {j} has no '{k}' rating.")
                v = float(exp[k])
                if not np.isfinite(v) or v < 0 or v > 1:
                    raise ValueError(f"explanation {j}: '{k}' must lie in [0, 1].")

        scores = []
        for exp in explanations:
            e = exp["explanation_quality"]
            f = exp["feature_importance"]
            i = exp["interpretability"]
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
                    "Above 0.7 (heuristic cut-off)"
                    if ts > 0.7
                    else (
                        "Between 0.5 and 0.7 (heuristic cut-offs)"
                        if ts > 0.5
                        else "At most 0.5 (heuristic cut-off)"
                    )
                ),
                "note": (
                    "A mean of three subjective ratings; it does not establish "
                    "readiness for clinical use."
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


class ObservedPerturbationAgreement:
    """
    Appendix Metric: Observed Perturbation Agreement (formerly the Robustness
    Certification Score, RCS)

    The mean element-wise agreement between the original predictions and the
    predictions under each perturbation (different documentation styles,
    measurement errors, ...). It describes the stability observed on the
    perturbations tried; it is not a certified robustness bound.

    See docs/Metric_Math_Derivations.md, Metric 37.
    """

    def calculate_agreement(
        self,
        original_predictions: np.ndarray,
        perturbed_predictions: List[np.ndarray],
        epsilon: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Calculate the observed perturbation agreement.

        Args:
            original_predictions: Original model predictions
            perturbed_predictions: List of prediction sets, one per perturbation,
                each with the shape of ``original_predictions``
            epsilon: optional perturbation size, recorded in the output only (it
                does not enter the score)

        Returns:
            ``agreement`` (also returned as ``rcs`` for compatibility), its SD,
            minimum and maximum over perturbations, and a 95% bootstrap CI over
            perturbations.

        Raises:
            ValueError: if no perturbations are given (up to 1.9.5 this returned
                0) or shapes differ.
        """
        from equimed_dss.inference import MetricResult, bootstrap_ci

        if not perturbed_predictions:
            raise ValueError(
                "perturbed_predictions is empty, so agreement is undefined."
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
            consistency_scores.append(np.mean(original == perturbed))

        agreement = float(np.mean(consistency_scores))
        level = (
            "High agreement (> 0.8)"
            if agreement > 0.8
            else (
                "Moderate agreement (0.6 to 0.8)"
                if agreement > 0.6
                else "Low agreement (<= 0.6)"
            )
        )
        out = {
            "agreement": agreement,
            "rcs": agreement,
            "rcs_std": float(np.std(consistency_scores)),
            "n_perturbations": len(perturbed_predictions),
            "epsilon": epsilon,
            "min_consistency": float(np.min(consistency_scores)),
            "max_consistency": float(np.max(consistency_scores)),
            "interpretation": {
                "range": "[0, 1]",
                "robustness_level": level,
                "verdict": level,
                "note": (
                    "Observed agreement on the perturbations tried; not a "
                    "certified robustness guarantee."
                ),
            },
        }

        # Mean per-perturbation agreement; a percentile bootstrap over
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
        return MetricResult(out, name="Agreement", value_key="agreement")

    def calculate_rcs(
        self,
        original_predictions: np.ndarray,
        perturbed_predictions: List[np.ndarray],
        epsilon: float = 0.1,
    ) -> Dict[str, Any]:
        """Backward-compatible name for :meth:`calculate_agreement`."""
        return self.calculate_agreement(
            original_predictions, perturbed_predictions, epsilon
        )


class RobustnessCertificationScore(ObservedPerturbationAgreement):
    """Deprecated name for :class:`ObservedPerturbationAgreement`.

    Kept so existing code runs; it will be removed in 2.0. The metric measures
    observed agreement under perturbation and certifies nothing.
    """

    def __init__(self):
        warnings.warn(
            "RobustnessCertificationScore is deprecated and will be removed in 2.0; "
            "use ObservedPerturbationAgreement (it measures observed agreement and "
            "certifies nothing).",
            DeprecationWarning,
            stacklevel=2,
        )
