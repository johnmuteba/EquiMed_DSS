from typing import Any, Dict, List, Tuple

import numpy as np


def _wilson_ci(n_success: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score 95% interval for a binomial proportion.

    The flip rate is a proportion, so its uncertainty is a binomial confidence
    interval, not a percentile of the 0/1 indicator vector.
    """
    if n == 0:
        return 0.0, 0.0
    p = n_success / n
    denom = 1.0 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


class DecisionFlipRate:
    """
    Domain 1: Reliability and Robustness Assessment
    Metric 3: Decision Flip Rate (DFR)

    Quantifies diagnostic instability under input variations (e.g., demographic flips).
    """

    def __init__(self):
        pass

    def calculate_dfr(
        self, original_decisions: List[Any], counterfactual_decisions: List[Any]
    ) -> Dict[str, float]:
        """
        Calculate Decision Flip Rate.

        Args:
            original_decisions: List of original decisions (e.g., binary labels 0/1 or class names).
            counterfactual_decisions: List of decisions after input perturbation.

        Returns:
            Dictionary containing flip rate and confidence intervals.
        """
        if len(original_decisions) != len(counterfactual_decisions):
            raise ValueError("Input lists must have the same length")

        n_samples = len(original_decisions)
        flips = [
            1 if o != c else 0
            for o, c in zip(original_decisions, counterfactual_decisions)
        ]

        n_flipped = int(np.sum(flips))
        flip_rate = float(np.mean(flips))
        ci_lower, ci_upper = _wilson_ci(n_flipped, n_samples)

        # Interpretation
        if flip_rate < 0.05:
            verdict = "Excellent Stability"
        elif flip_rate < 0.15:
            verdict = "Moderate Stability"
        else:
            verdict = "High Instability"

        return {
            "flip_rate": flip_rate,
            "n_flipped": n_flipped,
            "n_samples": n_samples,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "interpretation": {
                "range": "[0, 1]",
                "ideal": "Lower is better (close to 0)",
                "verdict": verdict,
            },
        }
