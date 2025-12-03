import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union, Any

class EquiMedDataset:
    """
    Standardized dataset container for EquiMed_DSS.
    
    Facilitates loading data from common formats (DataFrame, CSV) and 
    converting it into the specific structures required by EquiMed metrics.
    """
    
    def __init__(self, 
                 data: pd.DataFrame, 
                 target_col: str, 
                 prediction_col: Optional[str] = None,
                 sensitive_cols: Optional[List[str]] = None,
                 judge_cols: Optional[List[str]] = None,
                 timestamp_col: Optional[str] = None):
        """
        Initialize the dataset.
        
        Args:
            data: The raw data as a Pandas DataFrame.
            target_col: Name of the column containing ground truth labels.
            prediction_col: Name of the column containing model predictions.
            sensitive_cols: List of columns containing sensitive attributes (e.g., 'race', 'gender').
            judge_cols: List of columns containing scores from different judges (for ICC).
            timestamp_col: Name of the column containing timestamps (for TFD).
        """
        self.df = data.copy()
        self.target_col = target_col
        self.prediction_col = prediction_col
        self.sensitive_cols = sensitive_cols or []
        self.judge_cols = judge_cols or []
        self.timestamp_col = timestamp_col
        
        # Basic validation
        if target_col not in self.df.columns:
            raise ValueError(f"Target column '{target_col}' not found in data.")
            
    @classmethod
    def from_csv(cls, filepath: str, **kwargs):
        """
        Load data from a CSV file.
        
        Args:
            filepath: Path to the CSV file.
            **kwargs: Arguments passed to pd.read_csv and the constructor.
        """
        # Separate read_csv args from constructor args
        read_csv_args = {k: v for k, v in kwargs.items() if k in ['sep', 'header', 'index_col', 'usecols']}
        constructor_args = {k: v for k, v in kwargs.items() if k not in read_csv_args}
        
        df = pd.read_csv(filepath, **read_csv_args)
        return cls(df, **constructor_args)

    def to_reliability_matrix(self) -> np.ndarray:
        """
        Convert judge columns to a numpy matrix (n_samples, n_judges) for ICC.
        """
        if not self.judge_cols:
            raise ValueError("No judge columns specified.")
        return self.df[self.judge_cols].values

    def to_group_metrics(self, metric_fn: Any, sensitive_col: str) -> Dict[str, float]:
        """
        Calculate a metric for each subgroup within a sensitive attribute.
        
        Args:
            metric_fn: A function that takes (y_true, y_pred) and returns a float score.
            sensitive_col: The sensitive column to group by.
            
        Returns:
            Dictionary mapping group names to metric scores.
        """
        if sensitive_col not in self.df.columns:
            raise ValueError(f"Sensitive column '{sensitive_col}' not found.")
        if not self.prediction_col:
            raise ValueError("Prediction column not specified.")
            
        results = {}
        groups = self.df[sensitive_col].unique()
        
        for group in groups:
            group_data = self.df[self.df[sensitive_col] == group]
            if len(group_data) > 0:
                score = metric_fn(group_data[self.target_col], group_data[self.prediction_col])
                results[str(group)] = float(score)
                
        return results

    def to_time_series(self, metric_fn: Any, freq: str = 'M') -> List[float]:
        """
        Generate a time series of metric scores.
        
        Args:
            metric_fn: A function that takes (y_true, y_pred) and returns a float score.
            freq: Pandas frequency string (e.g., 'M' for month, 'D' for day).
            
        Returns:
            List of metric scores over time.
        """
        if not self.timestamp_col:
            raise ValueError("Timestamp column not specified.")
        if not self.prediction_col:
            raise ValueError("Prediction column not specified.")
            
        # Ensure timestamp is datetime
        self.df[self.timestamp_col] = pd.to_datetime(self.df[self.timestamp_col])
        
        results = []
        # Group by time period
        for _, period_data in self.df.set_index(self.timestamp_col).resample(freq):
            if len(period_data) > 0:
                score = metric_fn(period_data[self.target_col], period_data[self.prediction_col])
                results.append(float(score))
            else:
                # Handle empty periods if needed, or skip
                pass
                
        return results

    def to_error_dicts(self, sensitive_col: str, group1: str, group2: str) -> Dict[str, Dict[str, int]]:
        """
        Generate error counts (FN, FP) for two groups for HAFG.
        Assumes binary classification (0/1).
        """
        if not self.prediction_col:
            raise ValueError("Prediction column not specified.")
            
        def get_errors(group_df):
            y_true = group_df[self.target_col]
            y_pred = group_df[self.prediction_col]
            fn = ((y_true == 1) & (y_pred == 0)).sum()
            fp = ((y_true == 0) & (y_pred == 1)).sum()
            return {'fn': int(fn), 'fp': int(fp)}
            
        g1_data = self.df[self.df[sensitive_col] == group1]
        g2_data = self.df[self.df[sensitive_col] == group2]
        
        return {
            'group1': get_errors(g1_data),
            'group2': get_errors(g2_data)
        }
