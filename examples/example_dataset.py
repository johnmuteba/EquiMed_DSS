import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from equimed_dss.utils import EquiMedDataset
from equimed_dss.domain1 import InterRaterReliability
from equimed_dss.domain2 import HierarchicalEquityRatio, HarmAdjustedFairnessGap
from equimed_dss.domain3 import TemporalFairnessDrift

def main():
    print("=== EquiMedDataset Utility Demo ===")
    
    # 1. Create Synthetic Data (DataFrame)
    print("\n[1] Creating Synthetic DataFrame...")
    n_samples = 200
    dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='D')
    
    data = {
        'patient_id': range(n_samples),
        'diagnosis_true': np.random.randint(0, 2, n_samples),
        'diagnosis_pred': np.random.randint(0, 2, n_samples),
        'race': np.random.choice(['White', 'Black', 'Asian'], n_samples),
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'timestamp': dates,
        'judge1_score': np.random.normal(7, 1, n_samples),
        'judge2_score': np.random.normal(7.2, 1, n_samples),
        'judge3_score': np.random.normal(6.8, 1.2, n_samples)
    }
    df = pd.DataFrame(data)
    print(df.head())
    
    # 2. Initialize EquiMedDataset
    print("\n[2] Initializing EquiMedDataset...")
    dataset = EquiMedDataset(
        data=df,
        target_col='diagnosis_true',
        prediction_col='diagnosis_pred',
        sensitive_cols=['race', 'gender'],
        judge_cols=['judge1_score', 'judge2_score', 'judge3_score'],
        timestamp_col='timestamp'
    )
    
    # 3. Use with Domain 1 (ICC)
    print("\n[3] Converting for ICC (Domain 1)...")
    judge_matrix = dataset.to_reliability_matrix()
    icc = InterRaterReliability()
    icc_results = icc.calculate_icc_2_1(judge_matrix)
    print(f"ICC Score: {icc_results['score']:.3f}")
    
    # 4. Use with Domain 2 (HER)
    print("\n[4] Converting for HER (Domain 2)...")
    # Define a simple metric function (e.g., accuracy)
    def accuracy(y_true, y_pred):
        return accuracy_score(y_true, y_pred)
        
    group_scores = dataset.to_group_metrics(metric_fn=accuracy, sensitive_col='race')
    print(f"Group Accuracies: {group_scores}")
    
    her = HierarchicalEquityRatio()
    her_results = her.calculate_her(group_scores, reference_group='White')
    print(f"HER Results: {her_results}")
    
    # 5. Use with Domain 2 (HAFG)
    print("\n[5] Converting for HAFG (Domain 2)...")
    errors = dataset.to_error_dicts(sensitive_col='race', group1='Black', group2='White')
    hafg = HarmAdjustedFairnessGap()
    hafg_res = hafg.calculate_hafg(errors['group1'], errors['group2'])
    print(f"HAFG Results (Black vs White): {hafg_res}")
    
    # 6. Use with Domain 3 (TFD)
    print("\n[6] Converting for TFD (Domain 3)...")
    # Monthly accuracy
    ts_metrics = dataset.to_time_series(metric_fn=accuracy, freq='ME') # ME = Month End
    print(f"Monthly Accuracies: {ts_metrics}")
    
    tfd = TemporalFairnessDrift()
    drift_res = tfd.calculate_drift(ts_metrics)
    print(f"Drift Detected: {drift_res['drift_detected']}")

if __name__ == "__main__":
    main()
