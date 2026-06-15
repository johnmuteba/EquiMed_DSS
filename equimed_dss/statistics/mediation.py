"""
Mediation Analysis for Causal Pathway Investigation

Product-of-coefficients mediation:
- M = a0 + a1*X + e1            (mediator model)
- Y = b0 + b1*X + b2*M + e2     (outcome model)
- indirect effect = a1 * b2; total effect = direct + indirect;
- proportion mediated = indirect / total.

The indirect-effect confidence interval is obtained by nonparametric
bootstrap; a CI that excludes zero indicates a significant indirect pathway.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


class MediationAnalysis:
    """
    Causal mediation analysis to identify direct and indirect bias pathways.

    Reveals how bias operates through intermediate variables rather than
    direct demographic associations.
    """

    def __init__(self, n_bootstrap: int = 1000, random_state: Optional[int] = None):
        """
        Initialize mediation analyzer.

        Args:
            n_bootstrap: Number of bootstrap samples for CI estimation
            random_state: Random seed for reproducibility
        """
        self.n_bootstrap = n_bootstrap
        self.rng = np.random.RandomState(random_state)
        self.results = None

    def analyze_mediation(
        self,
        data: pd.DataFrame,
        treatment_var: str,
        mediator_var: str,
        outcome_var: str,
        covariates: Optional[List[str]] = None,
        alpha: float = 0.05,
    ) -> Dict[str, Any]:
        """
        Perform complete mediation analysis.

        Args:
            data: DataFrame containing all variables
            treatment_var: Independent variable (X)
            mediator_var: Mediator variable (M)
            outcome_var: Dependent variable (Y)
            covariates: List of covariate names to control for
            alpha: Significance level for confidence intervals

        Returns:
            Dict with direct effects, indirect effects, and proportions

        Example:
            >>> med = MediationAnalysis(n_bootstrap=1000)
            >>> results = med.analyze_mediation(
            ...     data=df,
            ...     treatment_var='demographic_group',
            ...     mediator_var='access_to_care',
            ...     outcome_var='diagnostic_accuracy'
            ... )
            >>> print(f"Proportion mediated: {results['proportion_mediated']:.1%}")
        """
        df = data.copy()

        # Prepare covariate string for formulas
        cov_str = ""
        if covariates:
            cov_str = " + " + " + ".join(covariates)

        try:
            from sklearn.linear_model import LinearRegression

            # Step 1: Mediator model (M ~ X + covariates)
            X_med = df[[treatment_var] + (covariates if covariates else [])].values
            y_med = df[mediator_var].values

            med_model = LinearRegression()
            med_model.fit(X_med, y_med)
            alpha_1 = med_model.coef_[0]  # Effect of X on M

            # Step 2: Outcome model (Y ~ X + M + covariates)
            X_out = df[
                [treatment_var, mediator_var] + (covariates if covariates else [])
            ].values
            y_out = df[outcome_var].values

            out_model = LinearRegression()
            out_model.fit(X_out, y_out)
            beta_1 = out_model.coef_[0]  # Direct effect of X on Y
            beta_2 = out_model.coef_[1]  # Effect of M on Y

            # Step 3: Total effect (Y ~ X + covariates, without mediator)
            X_total = df[[treatment_var] + (covariates if covariates else [])].values
            total_model = LinearRegression()
            total_model.fit(X_total, y_out)
            total_effect = total_model.coef_[0]

            # Calculate effects
            indirect_effect = alpha_1 * beta_2
            direct_effect = beta_1
            proportion_mediated = (
                indirect_effect / total_effect if abs(total_effect) > 1e-10 else 0
            )

            # Bootstrap confidence intervals
            indirect_boots = []
            for _ in range(self.n_bootstrap):
                boot_idx = self.rng.choice(len(df), size=len(df), replace=True)
                boot_data = df.iloc[boot_idx]

                # Fit mediator model
                X_med_boot = boot_data[
                    [treatment_var] + (covariates if covariates else [])
                ].values
                y_med_boot = boot_data[mediator_var].values
                med_boot = LinearRegression().fit(X_med_boot, y_med_boot)
                a1_boot = med_boot.coef_[0]

                # Fit outcome model
                X_out_boot = boot_data[
                    [treatment_var, mediator_var] + (covariates if covariates else [])
                ].values
                y_out_boot = boot_data[outcome_var].values
                out_boot = LinearRegression().fit(X_out_boot, y_out_boot)
                b2_boot = out_boot.coef_[1]

                indirect_boots.append(a1_boot * b2_boot)

            indirect_boots = np.array(indirect_boots)
            ci_lower = np.percentile(indirect_boots, (alpha / 2) * 100)
            ci_upper = np.percentile(indirect_boots, (1 - alpha / 2) * 100)

            self.results = {
                "total_effect": float(total_effect),
                "direct_effect": float(direct_effect),
                "indirect_effect": float(indirect_effect),
                "proportion_mediated": float(proportion_mediated),
                "alpha_1": float(alpha_1),  # X -> M
                "beta_1": float(direct_effect),  # X -> Y (direct)
                "beta_2": float(beta_2),  # M -> Y
                "indirect_ci_lower": float(ci_lower),
                "indirect_ci_upper": float(ci_upper),
                "interpretation": {
                    "mediation_type": self._classify_mediation(
                        direct_effect, indirect_effect, ci_lower, ci_upper
                    ),
                    "proportion_description": f"{proportion_mediated*100:.1f}% of effect is mediated",
                    "clinical_implication": self._interpret_mediation(
                        proportion_mediated
                    ),
                    "significance": (
                        "Significant" if ci_lower * ci_upper > 0 else "Non-significant"
                    ),
                },
            }

            return self.results

        except Exception as e:
            return {
                "error": str(e),
                "message": "Mediation analysis failed. Check your data and variable names.",
            }

    def _classify_mediation(
        self, direct: float, indirect: float, ci_lower: float, ci_upper: float
    ) -> str:
        """Classify type of mediation based on effects."""
        indirect_significant = ci_lower * ci_upper > 0

        if indirect_significant:
            if abs(direct) < 1e-10:
                return "Complete mediation"
            elif direct * indirect > 0:
                return "Partial mediation (complementary)"
            else:
                return "Partial mediation (competitive)"
        else:
            return "No mediation"

    def _interpret_mediation(self, proportion: float) -> str:
        """Interpret clinical implications of mediation proportion."""
        if proportion > 0.7:
            return (
                "Bias primarily operates through indirect pathways. "
                "Interventions should target intermediate mechanisms, not just "
                "direct demographic factors."
            )
        elif proportion > 0.3:
            return (
                "Mixed direct and indirect effects. Both pathway types "
                "should be addressed in bias mitigation strategies."
            )
        else:
            return (
                "Bias primarily operates through direct pathways. "
                "Traditional fairness interventions targeting direct "
                "associations may be effective."
            )

    def calculate_sobel_test(
        self, alpha_1: float, beta_2: float, se_alpha: float, se_beta: float
    ) -> Dict[str, float]:
        """
        Calculate Sobel test for mediation significance.

        Args:
            alpha_1: Effect of X on M
            beta_2: Effect of M on Y
            se_alpha: Standard error of alpha_1
            se_beta: Standard error of beta_2

        Returns:
            Dict with test statistic and p-value
        """
        # Indirect effect
        indirect = alpha_1 * beta_2

        # Sobel standard error
        se_sobel = np.sqrt((alpha_1**2) * (se_beta**2) + (beta_2**2) * (se_alpha**2))

        # Z-statistic
        z_stat = indirect / se_sobel if se_sobel > 0 else 0
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

        return {
            "indirect_effect": float(indirect),
            "se": float(se_sobel),
            "z_statistic": float(z_stat),
            "p_value": float(p_value),
            "significant": p_value < 0.05,
        }
