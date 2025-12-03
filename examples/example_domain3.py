import numpy as np
from equimed_dss.domain3 import TemporalFairnessDrift, AuditTraceabilityScore, GovernanceComplianceIndex
from equimed_dss.utils.visualization import plot_control_chart

def main():
    print("=== Domain 3: Governance and Transparency ===")
    
    # 8. TFD
    print("\n8. Temporal Fairness Drift (TFD)")
    tfd_metric = TemporalFairnessDrift()
    time_series = np.random.normal(0.8, 0.05, 20).tolist()
    # Add drift
    time_series.extend(np.random.normal(0.6, 0.05, 5).tolist())
    
    tfd_results = tfd_metric.calculate_drift(time_series)
    print(f"Drift Detected: {tfd_results['drift_detected']}")
    print(f"Interpretation: {tfd_results['interpretation']}")
    
    # Visualization
    print("Generating Control Chart...")
    plot_control_chart(time_series, tfd_results['ucl'], tfd_results['lcl'], title="Temporal Fairness Drift Control Chart", save_path="tfd_control_chart.png")
    print("Plot saved to tfd_control_chart.png")
    
    # 9. ATS
    print("\n9. Audit Traceability Score (ATS)")
    ats_metric = AuditTraceabilityScore()
    ats_results = ats_metric.calculate_ats(n_traceable=950, n_total=1000)
    print(f"ATS Results: {ats_results['ats_score']:.3f}")
    print(f"Interpretation: {ats_results['interpretation']}")
    
    # 10. GCI
    print("\n10. Governance Compliance Index (GCI)")
    gci_metric = GovernanceComplianceIndex()
    compliance = {
        'Policy1': True,
        'Policy2': True,
        'Policy3': False,
        'Policy4': True
    }
    gci_results = gci_metric.calculate_gci(compliance)
    print(f"GCI Results: {gci_results['gci']:.3f}")
    print(f"Interpretation: {gci_results['interpretation']}")

if __name__ == "__main__":
    main()
