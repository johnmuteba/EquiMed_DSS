"""
Hierarchical Linear Modeling (HLM) / Mixed Effects Models

Implements variance decomposition and hierarchical analysis as described in
Manuscript Section 2.4: Yij = γ00 + γ10Xij + u0j + rij

Key Finding from Manuscript: 55.8% of outcome variance at hospital/institutional
levels, not patient characteristics.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats


class HierarchicalLinearModeling:
    """
    Hierarchical Linear Modeling for nested data analysis.

    Assesses variance components to determine whether fairness outcomes
    are driven by individual-level (patient) or group-level (hospital/institution)
    factors.
    """

    def __init__(self):
        """Initialize HLM analyzer."""
        self.model_fitted = False
        self.results = None

    def fit_model(
        self,
        data: pd.DataFrame,
        outcome_var: str,
        level1_predictors: List[str],
        level2_var: str,
        level2_predictors: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Fit hierarchical linear model.

        Args:
            data: DataFrame with nested structure
            outcome_var: Dependent variable name
            level1_predictors: Individual-level predictors
            level2_var: Grouping variable (e.g., 'hospital_id')
            level2_predictors: Group-level predictors (optional)

        Returns:
            Dict with variance components and model statistics

        Example:
            >>> hlm = HierarchicalLinearModeling()
            >>> results = hlm.fit_model(
            ...     data=df,
            ...     outcome_var='fairness_score',
            ...     level1_predictors=['race', 'gender', 'ses'],
            ...     level2_var='hospital_id'
            ... )
            >>> print(f"Hospital-level variance: {results['icc']:.1%}")
        """
        try:
            import statsmodels.formula.api as smf
        except ImportError:
            raise ImportError(
                "statsmodels is required for HLM. "
                "Install with: pip install statsmodels"
            )

        # Prepare data
        df = data.copy()

        # Fit null model (unconditional model) for ICC calculation
        formula_null = f"{outcome_var} ~ 1"
        try:
            # Try to use mixed linear model
            from statsmodels.regression.mixed_linear_model import MixedLM

            # Null model
            null_model = MixedLM.from_formula(
                formula_null, data=df, groups=df[level2_var]
            ).fit()

            # Full model with level-1 predictors
            formula_full = f"{outcome_var} ~ {' + '.join(level1_predictors)}"
            full_model = MixedLM.from_formula(
                formula_full, data=df, groups=df[level2_var]
            ).fit()

            # Calculate ICC
            var_between = null_model.cov_re.iloc[0, 0]
            var_within = null_model.scale
            icc = var_between / (var_between + var_within)

            # Variance explained
            total_var_null = var_between + var_within
            total_var_full = full_model.cov_re.iloc[0, 0] + full_model.scale
            r_squared_marginal = 1 - (total_var_full / total_var_null)

            self.model_fitted = True
            self.results = {
                "icc": float(icc),
                "variance_between_groups": float(var_between),
                "variance_within_groups": float(var_within),
                "total_variance": float(total_var_null),
                "r_squared_marginal": float(r_squared_marginal),
                "aic": float(full_model.aic),
                "bic": float(full_model.bic),
                "n_groups": int(df[level2_var].nunique()),
                "n_observations": len(df),
                "interpretation": {
                    "icc_description": (
                        f"{icc*100:.1f}% of variance is between groups"
                    ),
                    "level_importance": (
                        "Group-level factors dominate"
                        if icc > 0.25
                        else (
                            "Individual-level factors dominate"
                            if icc < 0.10
                            else "Mixed influence"
                        )
                    ),
                    "clinical_implication": (
                        "Institutional interventions needed"
                        if icc > 0.25
                        else "Individual-level interventions appropriate"
                    ),
                },
            }

            return self.results

        except Exception as e:
            # Fallback to simpler variance decomposition
            return self._simple_variance_decomposition(df, outcome_var, level2_var)

    def _simple_variance_decomposition(
        self, df: pd.DataFrame, outcome_var: str, group_var: str
    ) -> Dict[str, Any]:
        """Simple ANOVA-based variance decomposition."""
        # Calculate group means
        group_means = df.groupby(group_var)[outcome_var].mean()
        grand_mean = df[outcome_var].mean()

        # Between-group variance
        n_per_group = df.groupby(group_var).size()
        ss_between = np.sum(n_per_group * (group_means - grand_mean) ** 2)

        # Within-group variance
        ss_within = np.sum((df[outcome_var] - df[group_var].map(group_means)) ** 2)

        # Degrees of freedom
        n_groups = df[group_var].nunique()
        n_total = len(df)
        df_between = n_groups - 1
        df_within = n_total - n_groups

        # Mean squares
        ms_between = ss_between / df_between if df_between > 0 else 0
        ms_within = ss_within / df_within if df_within > 0 else 0

        # ICC calculation
        icc = (ms_between - ms_within) / (
            ms_between + ms_within * (n_per_group.mean() - 1)
        )
        icc = max(0, min(1, icc))  # Bound between 0 and 1

        return {
            "icc": float(icc),
            "variance_between_groups": float(ms_between),
            "variance_within_groups": float(ms_within),
            "n_groups": n_groups,
            "n_observations": n_total,
            "interpretation": {
                "icc_description": f"{icc*100:.1f}% of variance is between groups",
                "level_importance": (
                    "Group-level factors dominate"
                    if icc > 0.25
                    else (
                        "Individual-level factors dominate"
                        if icc < 0.10
                        else "Mixed influence"
                    )
                ),
            },
        }

    def decompose_variance(
        self, data: pd.DataFrame, outcome_var: str, group_var: str
    ) -> Dict[str, float]:
        """
        Simple variance decomposition using ANOVA.

        Args:
            data: DataFrame with outcome and grouping variables
            outcome_var: Name of outcome variable
            group_var: Name of grouping variable

        Returns:
            Dict with variance components
        """
        return self._simple_variance_decomposition(data, outcome_var, group_var)

    def calculate_icc(
        self, data: pd.DataFrame, outcome_var: str, group_var: str
    ) -> float:
        """
        Calculate Intraclass Correlation Coefficient.

        Args:
            data: DataFrame with nested data
            outcome_var: Outcome variable name
            group_var: Grouping variable name

        Returns:
            ICC value between 0 and 1
        """
        result = self._simple_variance_decomposition(data, outcome_var, group_var)
        return result["icc"]
