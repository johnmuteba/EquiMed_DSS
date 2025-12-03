import numpy as np
from typing import Dict, Union

class HarmAdjustedFairnessGap:
    """
    Domain 2: Fairness, Equity, and Ethics Assessment
    Metric 5: Harm-Adjusted Fairness Gap (HAFG)
    
    Quantifies fairness weighted by potential clinical harm (cost of errors).
    """
    
    def __init__(self, cost_fn: float = 10.0, cost_fp: float = 3.0):
        """
        Initialize with costs for False Negatives and False Positives.
        
        Args:
            cost_fn: Cost of a false negative (default: 10).
            cost_fp: Cost of a false positive (default: 3).
        """
        self.cost_fn = cost_fn
        self.cost_fp = cost_fp
        
    def calculate_hafg(self, 
                       group1_errors: Dict[str, int], 
                       group2_errors: Dict[str, int]) -> Dict[str, float]:
        """
        Calculate HAFG between two groups (e.g., Marginalized vs Privileged).
        
        Args:
            group1_errors: Dict with 'fn' (count) and 'fp' (count) for group 1.
            group2_errors: Dict with 'fn' (count) and 'fp' (count) for group 2.
            
        Returns:
            Dictionary containing harm for each group and the gap.
        """
        harm1 = group1_errors.get('fn', 0) * self.cost_fn + group1_errors.get('fp', 0) * self.cost_fp
        harm2 = group2_errors.get('fn', 0) * self.cost_fn + group2_errors.get('fp', 0) * self.cost_fp
        
        gap = abs(harm1 - harm2)
        
        return {
            'harm_group1': float(harm1),
            'harm_group2': float(harm2),
            'hafg': float(gap),
            'ratio': float(harm1 / harm2) if harm2 > 0 else float('inf'),
            'interpretation': {
                'range': '[0, inf)',
                'ideal': 'Lower gap is better (close to 0)',
                'verdict': "Significant Harm Disparity" if gap > 10 else "Acceptable" # Arbitrary threshold for example
            }
        }
