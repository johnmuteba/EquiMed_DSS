"""
Mediation Analysis for Causal Pathway Investigation

Product-of-coefficients mediation:
- M = a0 + a1*X + e1            (mediator model)
- Y = b0 + b1*X + b2*M + e2     (outcome model)
- indirect effect = a1 * b2; total effect = direct + indirect;
- proportion mediated = indirect / total.

The indirect- and direct-effect confidence intervals are obtained by
nonparametric bootstrap; a CI that excludes zero indicates a significant
pathway. The mediation type follows Zhao, Lynch and Chen (J Consum Res 2010):
indirect effect significant and direct effect not, complete (indirect-only)
mediation; both significant, partial mediation, complementary when they share
a sign and competitive when they do not; indirect effect not significant, no
mediation. Reading the effects as causal requires no unmeasured confounding of
the treatment-mediator, treatment-outcome and mediator-outcome relations.
"""

import warnings
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
            treatment_var: Independent variable (X): numeric, boolean, or a
                category with exactly two levels (coded 0 for the first level in
                sorted order and 1 for the other; reported as
                ``treatment_coding``). With more than two categories, analyse one
                two-group contrast at a time.
            mediator_var: Mediator variable (M), numeric or boolean
            outcome_var: Dependent variable (Y), numeric or boolean
            covariates: List of covariate names to control for; categorical
                covariates are dummy-coded (first level as reference)
            alpha: Significance level for confidence intervals

        Returns:
            Dict with direct, indirect and total effects, their bootstrap CIs,
            the proportion mediated (NaN when the total effect is zero, where it
            is undefined) and an interpretation. The effects are ASSOCIATIONS
            from linear models (``estimand``); reading them as causal effects
            requires no unmeasured confounding of the X-M, X-Y and M-Y
            relations (see Imai, Keele and Tingley, Psychol Methods 2010).

        Raises:
            ValueError: for missing columns, missing values, a non-numeric
                mediator or outcome, or a treatment with more than two categories.

        Example:
            >>> med = MediationAnalysis(n_bootstrap=1000)
            >>> results = med.analyze_mediation(
            ...     data=df,
            ...     treatment_var='demographic_group',  # two categories
            ...     mediator_var='access_to_care',
            ...     outcome_var='diagnostic_accuracy'
            ... )
            >>> print(f"Proportion mediated: {results['proportion_mediated']:.1%}")
        """
        df, covariates, coding = self._prepare(
            data.copy(), treatment_var, mediator_var, outcome_var, covariates
        )

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
            # Undefined when the total effect is zero (up to 1.9.5 it was set to 0).
            proportion_mediated = (
                indirect_effect / total_effect
                if abs(total_effect) > 1e-10
                else float("nan")
            )

            # Bootstrap confidence intervals (indirect and direct effects)
            indirect_boots = []
            direct_boots = []
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
                direct_boots.append(out_boot.coef_[0])

            indirect_boots = np.array(indirect_boots)
            ci_lower = np.percentile(indirect_boots, (alpha / 2) * 100)
            ci_upper = np.percentile(indirect_boots, (1 - alpha / 2) * 100)
            direct_ci_lower = np.percentile(direct_boots, (alpha / 2) * 100)
            direct_ci_upper = np.percentile(direct_boots, (1 - alpha / 2) * 100)

            self.results = {
                "estimand": "associational (product of linear-model coefficients)",
                "treatment_coding": coding,
                "total_effect": float(total_effect),
                "direct_effect": float(direct_effect),
                "indirect_effect": float(indirect_effect),
                "proportion_mediated": float(proportion_mediated),
                "alpha_1": float(alpha_1),  # X -> M
                "beta_1": float(direct_effect),  # X -> Y (direct)
                "beta_2": float(beta_2),  # M -> Y
                "indirect_ci_lower": float(ci_lower),
                "indirect_ci_upper": float(ci_upper),
                "direct_ci_lower": float(direct_ci_lower),
                "direct_ci_upper": float(direct_ci_upper),
                "interpretation": {
                    "mediation_type": self._classify_mediation(
                        direct_effect,
                        indirect_effect,
                        ci_lower,
                        ci_upper,
                        direct_ci_lower,
                        direct_ci_upper,
                    ),
                    "proportion_description": (
                        f"{proportion_mediated*100:.1f}% of the total association "
                        "runs through the mediator in this linear model"
                        if np.isfinite(proportion_mediated)
                        else "Proportion mediated undefined (total effect is zero)"
                    ),
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
            warnings.warn(
                f"Mediation analysis failed ({type(e).__name__}: {e}).",
                UserWarning,
                stacklevel=2,
            )
            return {
                "error": str(e),
                "message": "Mediation analysis failed. Check your data and variable names.",
            }

    def _classify_mediation(
        self,
        direct: float,
        indirect: float,
        ci_lower: float,
        ci_upper: float,
        direct_ci_lower: Optional[float] = None,
        direct_ci_upper: Optional[float] = None,
    ) -> str:
        """Classify the type of mediation (Zhao, Lynch and Chen 2010).

        Significance is judged from the bootstrap CIs. Without a direct-effect
        CI, the direct effect counts as absent only if it is exactly zero (the
        rule used up to 1.9.5, under which complete mediation was practically
        never reported).
        """
        indirect_significant = ci_lower * ci_upper > 0
        if direct_ci_lower is not None and direct_ci_upper is not None:
            direct_significant = direct_ci_lower * direct_ci_upper > 0
        else:
            direct_significant = abs(direct) >= 1e-10

        if indirect_significant:
            if not direct_significant:
                return "Complete mediation"
            elif direct * indirect > 0:
                return "Partial mediation (complementary)"
            else:
                return "Partial mediation (competitive)"
        else:
            return "No mediation"

    def _interpret_mediation(self, proportion: float) -> str:
        """Describe what the proportion does and does not show.

        Up to 1.9.5 fixed cut-offs of the proportion were turned into advice on
        which interventions to make; a linear decomposition cannot support that.
        """
        return (
            "Associational decomposition from linear models. Reading it causally "
            "requires no unmeasured confounding of the treatment-mediator, "
            "treatment-outcome and mediator-outcome relations, and the proportion "
            "does not by itself indicate which intervention would work."
        )

    @staticmethod
    def _prepare(df, treatment_var, mediator_var, outcome_var, covariates):
        """Check the analysis columns and encode categorical variables."""
        covariates = list(covariates or [])
        cols = [treatment_var, mediator_var, outcome_var] + covariates
        absent = [c for c in cols if c not in df.columns]
        if absent:
            raise ValueError(f"columns not found: {absent}")
        n_missing = int(df[cols].isna().any(axis=1).sum())
        if n_missing:
            raise ValueError(
                f"{n_missing} rows have missing values in the analysis columns; drop "
                "or impute them first."
            )
        coding = None
        t = df[treatment_var]
        if pd.api.types.is_bool_dtype(t):
            df[treatment_var] = t.astype(int)
        elif not pd.api.types.is_numeric_dtype(t):
            levels = sorted(t.astype(str).unique())
            if len(levels) != 2:
                raise ValueError(
                    f"treatment '{treatment_var}' has {len(levels)} categories; this "
                    "analysis needs a numeric or two-level exposure, so analyse one "
                    "two-group contrast at a time."
                )
            df[treatment_var] = (t.astype(str) == levels[1]).astype(int)
            coding = {"reference (0)": levels[0], "exposed (1)": levels[1]}
        for c in (mediator_var, outcome_var):
            if pd.api.types.is_bool_dtype(df[c]):
                df[c] = df[c].astype(int)
            elif not pd.api.types.is_numeric_dtype(df[c]):
                raise ValueError(f"'{c}' must be numeric (the models are linear).")
        cov_cols = []
        for c in covariates:
            if pd.api.types.is_bool_dtype(df[c]):
                df[c] = df[c].astype(int)
                cov_cols.append(c)
            elif pd.api.types.is_numeric_dtype(df[c]):
                cov_cols.append(c)
            else:
                dummies = pd.get_dummies(
                    df[c].astype(str), prefix=c, drop_first=True, dtype=float
                )
                df = pd.concat([df, dummies], axis=1)
                cov_cols.extend(dummies.columns)
        return df, cov_cols, coding

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
