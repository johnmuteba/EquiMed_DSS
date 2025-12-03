import numpy as np

from equimed_dss.appendix import (
    AdvancedInfoTheoryMetrics,
    AdvancedNetworkMetrics,
    AdvancedReliabilityMetrics,
)


def main():
    print("=== Appendix Metrics ===")

    # Reliability
    print("\nA.1 Advanced Reliability")
    rel_metrics = AdvancedReliabilityMetrics()
    data = np.random.normal(10, 2, 100)
    bci = rel_metrics.calculate_bootstrap_ci(data.tolist())
    print(f"Bootstrap CI: {bci}")

    spa = rel_metrics.calculate_power_analysis(effect_size=0.5)
    print(f"Power Analysis (d=0.5): {spa}")

    # Info Theory
    print("\nA.2 Advanced Info Theory")
    info_metrics = AdvancedInfoTheoryMetrics()
    dist_p = [0.1, 0.4, 0.5]
    dist_q = [0.2, 0.3, 0.5]
    jsd = info_metrics.calculate_jsd(dist_p, dist_q)
    print(f"Jensen-Shannon Divergence: {jsd:.4f}")

    # Network
    print("\nA.3 Advanced Network")
    net_metrics = AdvancedNetworkMetrics()
    adj = np.array([[0, 1, 1, 0], [1, 0, 1, 0], [1, 1, 0, 1], [0, 0, 1, 0]])
    mod = net_metrics.calculate_modularity(adj)
    print(f"Network Modularity: {mod:.4f}")


if __name__ == "__main__":
    main()
