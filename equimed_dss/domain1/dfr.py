import numpy as np
from typing import Dict, List, Any

class DecisionFlipRate:
    """
    Domain 1: Reliability and Robustness Assessment
    Metric 3: Decision Flip Rate (DFR)
    
    Quantifies diagnostic instability under input variations (e.g., demographic flips).
    """
    
    def __init__(self):
        pass
        
    def calculate_dfr(self, original_decisions: List[Any], counterfactual_decisions: List[Any]) -> Dict[str, float]:
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
        flips = [1 if o != c else 0 for o, c in zip(original_decisions, counterfactual_decisions)]
        
        flip_rate = float(np.mean(flips))
        
        # Interpretation
        if flip_rate < 0.05:
            verdict = "Excellent Stability"
        elif flip_rate < 0.15:
            verdict = "Moderate Stability"
        else:
            verdict = "High Instability"
            
        return {
            'flip_rate': flip_rate,
            'ci_lower': float(np.percentile(flips, 2.5)),
            'ci_upper': float(np.percentile(flips, 97.5)),
            'interpretation': {
                'range': '[0, 1]',
                'ideal': 'Lower is better (close to 0)',
                'verdict': verdict
            }
        }
