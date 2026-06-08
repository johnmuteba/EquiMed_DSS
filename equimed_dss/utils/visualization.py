import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns


def plot_network_graph(
    G: nx.Graph,
    title: str = "Network Graph",
    node_color: str = "skyblue",
    save_path: Optional[str] = None,
):
    """Plot a NetworkX graph."""
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color=node_color,
        node_size=1500,
        font_size=10,
        font_weight="bold",
        edge_color="gray",
        alpha=0.7,
    )
    labels = nx.get_edge_attributes(G, "weight")
    # Format weights
    labels = {k: f"{v:.2f}" for k, v in labels.items()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.title(title)
    plt.axis("off")
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path)
    # Always render the figure (inline in notebooks); saving does not suppress it.
    plt.show()


def plot_correlation_matrix(
    corr_matrix: pd.DataFrame,
    title: str = "Correlation Matrix",
    save_path: Optional[str] = None,
):
    """Plot correlation matrix heatmap."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1, center=0)
    plt.title(title)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path)
    # Always render the figure (inline in notebooks); saving does not suppress it.
    plt.show()


def plot_metric_distribution(
    data: List[float],
    title: str = "Metric Distribution",
    save_path: Optional[str] = None,
):
    """Plot distribution of a metric (e.g., bootstrap results)."""
    plt.figure(figsize=(10, 6))
    sns.histplot(data, kde=True, color="blue", alpha=0.3)
    plt.axvline(
        np.mean(data), color="red", linestyle="--", label=f"Mean: {np.mean(data):.2f}"
    )
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path)
    # Always render the figure (inline in notebooks); saving does not suppress it.
    plt.show()


def plot_bland_altman(
    means: np.ndarray,
    diffs: np.ndarray,
    limits: Tuple[float, float],
    title: str = "Bland-Altman Plot",
    save_path: Optional[str] = None,
):
    """Plot Bland-Altman analysis."""
    plt.figure(figsize=(8, 6))
    plt.scatter(means, diffs, alpha=0.5)
    plt.axhline(np.mean(diffs), color="black", linestyle="-")
    plt.axhline(limits[0], color="red", linestyle="--")
    plt.axhline(limits[1], color="red", linestyle="--")
    plt.title(title)
    plt.xlabel("Mean")
    plt.ylabel("Difference")
    plt.grid(True, alpha=0.3)
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path)
    # Always render the figure (inline in notebooks); saving does not suppress it.
    plt.show()


def plot_her_heatmap(
    her_scores: Dict[str, Dict[str, float]],
    title: str = "HER Heatmap",
    save_path: Optional[str] = None,
):
    """Plot HER scores as a heatmap."""
    groups = list(next(iter(her_scores.values())).keys())
    corpora = list(her_scores.keys())

    matrix = np.zeros((len(corpora), len(groups)))
    for i, corpus in enumerate(corpora):
        for j, group in enumerate(groups):
            matrix[i, j] = her_scores[corpus].get(group, 0)

    plt.figure(figsize=(10, 6))
    plt.imshow(matrix, cmap="RdYlGn", vmin=0.5, vmax=1.5, aspect="auto")
    plt.colorbar(label="HER")
    plt.yticks(range(len(corpora)), corpora)
    plt.xticks(range(len(groups)), groups, rotation=45)
    plt.title(title)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path)
    # Always render the figure (inline in notebooks); saving does not suppress it.
    plt.show()


def plot_control_chart(
    data: List[float],
    ucl: float,
    lcl: float,
    title: str = "Control Chart",
    save_path: Optional[str] = None,
):
    """Plot process control chart."""
    plt.figure(figsize=(10, 5))
    plt.plot(data, marker="o")
    plt.axhline(np.mean(data), color="green", linestyle="-")
    plt.axhline(ucl, color="red", linestyle="--")
    plt.axhline(lcl, color="red", linestyle="--")
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Metric Value")
    plt.grid(True, alpha=0.3)
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path)
    # Always render the figure (inline in notebooks); saving does not suppress it.
    plt.show()


# ========================================================================
# MANUSCRIPT VISUALIZATIONS (Figures 2-7)
# ========================================================================


def plot_figure2_reliability_dashboard(
    reliability_data: Dict[str, Any],
    save_path: Optional[str] = None,
):
    """
    Figure 2: Reliability Analysis Dashboard.

    Creates a comprehensive 4-panel dashboard showing:
    - Panel A: ICC scores across raters
    - Panel B: Cronbach's Alpha with confidence intervals
    - Panel C: Bland-Altman plot for inter-rater agreement
    - Panel D: Temporal stability over time

    Args:
        reliability_data: Dict containing:
            - 'icc_scores': Dict[str, float] - ICC by rater
            - 'cronbach_alpha': Dict with 'alpha' and 'ci_lower', 'ci_upper'
            - 'bland_altman': Dict with 'means', 'diffs', 'loa_upper', 'loa_lower'
            - 'temporal': Dict with 'timepoints' and 'reliability_scores'
        save_path: Optional path to save figure

    Reference: Manuscript Figure 2
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Reliability Analysis Dashboard", fontsize=16, fontweight="bold")

    # Panel A: ICC Scores
    ax1 = axes[0, 0]
    raters = list(reliability_data["icc_scores"].keys())
    icc_values = list(reliability_data["icc_scores"].values())
    colors = [
        "green" if v > 0.75 else "orange" if v > 0.6 else "red" for v in icc_values
    ]

    ax1.barh(raters, icc_values, color=colors, alpha=0.7)
    ax1.axvline(0.75, color="green", linestyle="--", label="Excellent (>0.75)")
    ax1.axvline(0.6, color="orange", linestyle="--", label="Good (>0.6)")
    ax1.set_xlabel("ICC Score", fontsize=12)
    ax1.set_ylabel("Rater", fontsize=12)
    ax1.set_title(
        "A. Intraclass Correlation Coefficient", fontsize=14, fontweight="bold"
    )
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="x")
    ax1.set_xlim([0, 1])

    # Panel B: Cronbach's Alpha
    ax2 = axes[0, 1]
    alpha_data = reliability_data["cronbach_alpha"]
    alpha_val = alpha_data["alpha"]
    ci_lower = alpha_data["ci_lower"]
    ci_upper = alpha_data["ci_upper"]

    ax2.barh(["Cronbach's α"], [alpha_val], color="steelblue", alpha=0.7)
    ax2.errorbar(
        [alpha_val],
        ["Cronbach's α"],
        xerr=[[alpha_val - ci_lower], [ci_upper - alpha_val]],
        fmt="o",
        color="black",
        capsize=5,
        capthick=2,
    )
    ax2.axvline(0.9, color="green", linestyle="--", label="Excellent (>0.9)")
    ax2.axvline(0.8, color="orange", linestyle="--", label="Good (>0.8)")
    ax2.axvline(0.7, color="red", linestyle="--", label="Acceptable (>0.7)")
    ax2.set_xlabel("Alpha Value", fontsize=12)
    ax2.set_title(
        "B. Internal Consistency (Cronbach's α)", fontsize=14, fontweight="bold"
    )
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="x")
    ax2.set_xlim([0, 1])

    # Panel C: Bland-Altman Plot
    ax3 = axes[1, 0]
    ba_data = reliability_data["bland_altman"]
    means = np.array(ba_data["means"])
    diffs = np.array(ba_data["diffs"])
    mean_diff = np.mean(diffs)
    loa_upper = ba_data["loa_upper"]
    loa_lower = ba_data["loa_lower"]

    ax3.scatter(means, diffs, alpha=0.5, s=30, color="steelblue")
    ax3.axhline(
        mean_diff,
        color="black",
        linestyle="-",
        linewidth=2,
        label=f"Mean: {mean_diff:.3f}",
    )
    ax3.axhline(
        loa_upper,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Upper LoA: {loa_upper:.3f}",
    )
    ax3.axhline(
        loa_lower,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Lower LoA: {loa_lower:.3f}",
    )
    ax3.fill_between(
        [means.min(), means.max()], loa_lower, loa_upper, alpha=0.1, color="red"
    )
    ax3.set_xlabel("Mean of Two Measurements", fontsize=12)
    ax3.set_ylabel("Difference Between Measurements", fontsize=12)
    ax3.set_title("C. Bland-Altman Agreement Analysis", fontsize=14, fontweight="bold")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Panel D: Temporal Stability
    ax4 = axes[1, 1]
    temporal = reliability_data["temporal"]
    timepoints = temporal["timepoints"]
    scores = temporal["reliability_scores"]

    ax4.plot(
        timepoints, scores, marker="o", linewidth=2, markersize=8, color="steelblue"
    )
    ax4.axhline(0.75, color="green", linestyle="--", label="Target (>0.75)")
    ax4.fill_between(timepoints, 0.75, 1.0, alpha=0.1, color="green")
    ax4.set_xlabel("Time Point", fontsize=12)
    ax4.set_ylabel("Reliability Score", fontsize=12)
    ax4.set_title(
        "D. Temporal Stability of Reliability", fontsize=14, fontweight="bold"
    )
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 1])

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    # Always render the figure (inline in notebooks); saving does not suppress it.
    plt.show()


def plot_figure3_corpus_comparison(
    corpus_data: Dict[str, Any],
    save_path: Optional[str] = None,
):
    """
    Figure 3: Three-Corpus Comparative Analysis.

    Creates a comprehensive 4-panel comparison across corpora:
    - Panel A: Bias-Gini coefficients by corpus
    - Panel B: Temporal fairness drift over time
    - Panel C: Clinical harm distribution (HAFG)
    - Panel D: Network stability metrics

    Args:
        corpus_data: Dict containing:
            - 'bias_gini': Dict[corpus_name, float] - Bias-Gini by corpus
            - 'temporal_drift': Dict with 'timepoints', 'peer_reviewed', 'community', 'mimic'
            - 'clinical_harm': Dict with 'peer_reviewed', 'community', 'mimic' (lists of HAFG values)
            - 'network_stability': Dict[corpus_name, Dict] with 'density', 'clustering', 'modularity'
        save_path: Optional path to save figure

    Reference: Manuscript Figure 3
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Three-Corpus Comparative Analysis", fontsize=16, fontweight="bold")

    corpora_names = ["Peer-Reviewed", "Community", "MIMIC-IV"]
    colors_corpus = {
        "Peer-Reviewed": "steelblue",
        "Community": "orange",
        "MIMIC-IV": "green",
    }

    # Panel A: Bias-Gini Coefficients
    ax1 = axes[0, 0]
    bias_gini = corpus_data["bias_gini"]
    corpora = list(bias_gini.keys())
    gini_values = list(bias_gini.values())

    bars = ax1.bar(
        corpora,
        gini_values,
        color=[colors_corpus.get(c, "gray") for c in corpora],
        alpha=0.7,
    )
    ax1.axhline(
        0.3, color="red", linestyle="--", linewidth=2, label="High Inequality (>0.3)"
    )
    ax1.set_ylabel("Bias-Gini Coefficient", fontsize=12)
    ax1.set_title("A. Bias Inequality Across Corpora", fontsize=14, fontweight="bold")
    ax1.set_ylim([0, max(gini_values) * 1.2])
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    # Panel B: Temporal Fairness Drift
    ax2 = axes[0, 1]
    temporal = corpus_data["temporal_drift"]
    timepoints = temporal["timepoints"]

    for corpus_name in corpora_names:
        key = corpus_name.lower().replace("-", "_")
        if key in temporal:
            ax2.plot(
                timepoints,
                temporal[key],
                marker="o",
                linewidth=2,
                label=corpus_name,
                color=colors_corpus[corpus_name],
            )

    ax2.axhline(
        0.1, color="red", linestyle="--", linewidth=2, label="Stability Threshold (0.1)"
    )
    ax2.set_xlabel("Time Point", fontsize=12)
    ax2.set_ylabel("Temporal Fairness Drift (TFD)", fontsize=12)
    ax2.set_title(
        "B. Temporal Fairness Drift Over Time", fontsize=14, fontweight="bold"
    )
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Panel C: Clinical Harm Distribution (HAFG)
    ax3 = axes[1, 0]
    harm_data = corpus_data["clinical_harm"]

    harm_values = []
    harm_labels = []
    for corpus_name in corpora_names:
        key = corpus_name.lower().replace("-", "_")
        if key in harm_data:
            harm_values.append(harm_data[key])
            harm_labels.append(corpus_name)

    bp = ax3.boxplot(
        harm_values, labels=harm_labels, patch_artist=True, notch=True, showmeans=True
    )

    # Color the boxes
    for patch, corpus_name in zip(bp["boxes"], harm_labels):
        patch.set_facecolor(colors_corpus[corpus_name])
        patch.set_alpha(0.7)

    ax3.axhline(
        0.15, color="red", linestyle="--", linewidth=2, label="High Harm (>0.15)"
    )
    ax3.set_ylabel("Harm-Adjusted Fairness Gap (HAFG)", fontsize=12)
    ax3.set_title(
        "C. Clinical Harm Distribution by Corpus", fontsize=14, fontweight="bold"
    )
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis="y")

    # Panel D: Network Stability Metrics
    ax4 = axes[1, 1]
    network_stability = corpus_data["network_stability"]

    metrics = ["Density", "Clustering", "Modularity"]
    x_pos = np.arange(len(corpora_names))
    width = 0.25

    for idx, metric_name in enumerate(metrics):
        metric_key = metric_name.lower()
        values = [
            network_stability[c.lower().replace("-", "_")][metric_key]
            for c in corpora_names
            if c.lower().replace("-", "_") in network_stability
        ]

        ax4.bar(x_pos + idx * width, values, width, label=metric_name, alpha=0.7)

    ax4.set_xlabel("Corpus", fontsize=12)
    ax4.set_ylabel("Metric Value", fontsize=12)
    ax4.set_title(
        "D. Network Stability Characteristics", fontsize=14, fontweight="bold"
    )
    ax4.set_xticks(x_pos + width)
    ax4.set_xticklabels(corpora_names)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis="y")
    ax4.set_ylim([0, 1])

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    # Always render the figure (inline in notebooks); saving does not suppress it.
    plt.show()


def plot_figure4_temporal_robustness(
    analysis_data: Dict[str, Any],
    save_path: Optional[str] = None,
):
    """
    Figure 4: Temporal and Robustness Analysis.

    Creates a 3-panel visualization showing:
    - Panel A: Mediation analysis diagram (indirect/direct effects)
    - Panel B: Regression analysis with bootstrap CIs
    - Panel C: Robustness over perturbations

    Args:
        analysis_data: Dict containing:
            - 'mediation': Dict with 'indirect_effect', 'direct_effect', 'ci_lower', 'ci_upper'
            - 'regression': Dict with 'predictors', 'coefficients', 'ci_lower', 'ci_upper'
            - 'robustness': Dict with 'perturbation_levels', 'rcs_scores'
        save_path: Optional path to save figure

    Reference: Manuscript Figure 4
    """
    fig = plt.figure(figsize=(18, 6))
    gs = fig.add_gridspec(1, 3, hspace=0.3, wspace=0.3)
    fig.suptitle("Temporal and Robustness Analysis", fontsize=16, fontweight="bold")

    # Panel A: Mediation Analysis
    ax1 = fig.add_subplot(gs[0, 0])
    mediation = analysis_data["mediation"]

    effects = ["Indirect\nEffect", "Direct\nEffect", "Total\nEffect"]
    values = [
        mediation["indirect_effect"],
        mediation["direct_effect"],
        mediation["indirect_effect"] + mediation["direct_effect"],
    ]
    ci_lower = [
        mediation["indirect_ci_lower"],
        mediation["direct_ci_lower"],
        mediation["total_ci_lower"],
    ]
    ci_upper = [
        mediation["indirect_ci_upper"],
        mediation["direct_ci_upper"],
        mediation["total_ci_upper"],
    ]

    colors_med = ["coral", "steelblue", "purple"]
    bars = ax1.barh(effects, values, color=colors_med, alpha=0.7)

    # Add error bars
    for i, (effect, val, lower, upper) in enumerate(
        zip(effects, values, ci_lower, ci_upper)
    ):
        ax1.errorbar(
            [val],
            [i],
            xerr=[[val - lower], [upper - val]],
            fmt="o",
            color="black",
            capsize=5,
            capthick=2,
        )

    ax1.axvline(0, color="black", linestyle="-", linewidth=1)
    ax1.set_xlabel("Effect Size (β)", fontsize=12)
    ax1.set_title(
        "A. Mediation Analysis\n(72.1% Indirect Pathway)",
        fontsize=14,
        fontweight="bold",
    )
    ax1.grid(True, alpha=0.3, axis="x")

    # Add value labels
    for bar, val in zip(bars, values):
        width = bar.get_width()
        ax1.text(
            width + 0.1,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            ha="left",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    # Panel B: Regression Coefficients with Bootstrap CIs
    ax2 = fig.add_subplot(gs[0, 1])
    regression = analysis_data["regression"]

    predictors = regression["predictors"]
    coefficients = regression["coefficients"]
    ci_lower_reg = regression["ci_lower"]
    ci_upper_reg = regression["ci_upper"]

    y_pos = np.arange(len(predictors))
    colors_reg = ["green" if c > 0 else "red" for c in coefficients]

    ax2.barh(y_pos, coefficients, color=colors_reg, alpha=0.7)
    ax2.errorbar(
        coefficients,
        y_pos,
        xerr=[
            [coefficients[i] - ci_lower_reg[i] for i in range(len(coefficients))],
            [ci_upper_reg[i] - coefficients[i] for i in range(len(coefficients))],
        ],
        fmt="o",
        color="black",
        capsize=5,
        capthick=2,
    )

    ax2.axvline(0, color="black", linestyle="-", linewidth=2)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(predictors)
    ax2.set_xlabel("Regression Coefficient (β)", fontsize=12)
    ax2.set_title(
        "B. Regression Analysis with Bootstrap CIs", fontsize=14, fontweight="bold"
    )
    ax2.grid(True, alpha=0.3, axis="x")

    # Panel C: Robustness Certification
    ax3 = fig.add_subplot(gs[0, 2])
    robustness = analysis_data["robustness"]

    perturbation_levels = robustness["perturbation_levels"]
    rcs_scores = robustness["rcs_scores"]

    ax3.plot(
        perturbation_levels,
        rcs_scores,
        marker="o",
        linewidth=3,
        markersize=8,
        color="steelblue",
        label="RCS Score",
    )
    ax3.fill_between(perturbation_levels, 0, rcs_scores, alpha=0.2, color="steelblue")
    ax3.axhline(
        0.8, color="green", linestyle="--", linewidth=2, label="Deployment Ready (>0.8)"
    )
    ax3.axhline(
        0.6, color="orange", linestyle="--", linewidth=2, label="Caution (0.6-0.8)"
    )

    ax3.set_xlabel("Perturbation Level (ε)", fontsize=12)
    ax3.set_ylabel("Robustness Certification Score (RCS)", fontsize=12)
    ax3.set_title("C. Robustness Under Perturbations", fontsize=14, fontweight="bold")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1])

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    # Always render the figure (inline in notebooks); saving does not suppress it.
    plt.show()


def plot_figure5_ethics_governance(
    ethics_data: Dict[str, Any],
    save_path: Optional[str] = None,
):
    """
    Figure 5: Ethics and Governance Dashboard.

    Creates a 3-panel visualization showing:
    - Panel A: Ethical risk index over time
    - Panel B: RAMS (Risk, Accountability, Monitoring, Safety) radar chart
    - Panel C: Fairness ecosystem interconnections

    Args:
        ethics_data: Dict containing:
            - 'ethical_risk': Dict with 'timepoints', 'eri_scores', 'violations'
            - 'rams': Dict with 'risk', 'accountability', 'monitoring', 'safety' (0-1 scores)
            - 'fairness_ecosystem': Dict with 'components', 'connections' (adjacency matrix)
        save_path: Optional path to save figure

    Reference: Manuscript Figure 5
    """
    fig = plt.figure(figsize=(18, 6))
    gs = fig.add_gridspec(1, 3, hspace=0.3, wspace=0.3)
    fig.suptitle("Ethics and Governance Dashboard", fontsize=16, fontweight="bold")

    # Panel A: Ethical Risk Index Over Time
    ax1 = fig.add_subplot(gs[0, 0])
    ethical_risk = ethics_data["ethical_risk"]

    timepoints = ethical_risk["timepoints"]
    eri_scores = ethical_risk["eri_scores"]
    violations = ethical_risk.get("violations", [0] * len(timepoints))

    ax1_twin = ax1.twinx()

    # ERI line
    line1 = ax1.plot(
        timepoints,
        eri_scores,
        marker="o",
        linewidth=2,
        markersize=8,
        color="red",
        label="ERI Score",
    )
    ax1.axhline(
        0.15,
        color="red",
        linestyle="--",
        linewidth=2,
        alpha=0.5,
        label="High Risk (>0.15)",
    )
    ax1.fill_between(timepoints, 0, eri_scores, alpha=0.2, color="red")

    # Violations bars
    line2 = ax1_twin.bar(
        timepoints, violations, alpha=0.3, color="orange", label="Violations"
    )

    ax1.set_xlabel("Time Point", fontsize=12)
    ax1.set_ylabel("Ethical Risk Index (ERI)", fontsize=12, color="red")
    ax1_twin.set_ylabel("Number of Violations", fontsize=12, color="orange")
    ax1.set_title("A. Ethical Risk Monitoring", fontsize=14, fontweight="bold")
    ax1.tick_params(axis="y", labelcolor="red")
    ax1_twin.tick_params(axis="y", labelcolor="orange")
    ax1.grid(True, alpha=0.3)

    lines = line1 + [line2]
    labels = [l.get_label() for l in line1] + ["Violations"]
    ax1.legend(lines, labels, loc="upper left")

    # Panel B: RAMS Radar Chart
    ax2 = fig.add_subplot(gs[0, 1], projection="polar")
    rams = ethics_data["rams"]

    categories = [
        "Risk\nManagement",
        "Accountability",
        "Monitoring",
        "Safety\nProtocols",
    ]
    values = [rams["risk"], rams["accountability"], rams["monitoring"], rams["safety"]]
    values += values[:1]  # Complete the circle

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    ax2.plot(angles, values, "o-", linewidth=2, color="steelblue", label="Current")
    ax2.fill(angles, values, alpha=0.25, color="steelblue")

    # Target values (all at 0.9)
    target = [0.9] * len(angles)
    ax2.plot(
        angles,
        target,
        "o--",
        linewidth=2,
        color="green",
        alpha=0.5,
        label="Target (0.9)",
    )

    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(categories, fontsize=10)
    ax2.set_ylim(0, 1)
    ax2.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax2.set_title("B. RAMS Assessment", fontsize=14, fontweight="bold", pad=20)
    ax2.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    ax2.grid(True)

    # Panel C: Fairness Ecosystem Network
    ax3 = fig.add_subplot(gs[0, 2])
    ecosystem = ethics_data["fairness_ecosystem"]

    components = ecosystem["components"]
    connections = np.array(ecosystem["connections"])

    # Create network graph
    G = nx.from_numpy_array(connections)
    pos = nx.spring_layout(G, seed=42)

    # Node colors by importance (degree)
    node_degrees = dict(G.degree())
    node_colors = [node_degrees[node] for node in G.nodes()]

    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_colors,
        node_size=800,
        cmap="YlOrRd",
        vmin=0,
        vmax=max(node_colors),
        ax=ax3,
    )
    nx.draw_networkx_labels(
        G,
        pos,
        labels={i: components[i] for i in range(len(components))},
        font_size=8,
        font_weight="bold",
        ax=ax3,
    )
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=2, edge_color="gray", ax=ax3)

    ax3.set_title(
        "C. Fairness Ecosystem Interconnections", fontsize=14, fontweight="bold"
    )
    ax3.axis("off")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    # Always render the figure (inline in notebooks); saving does not suppress it.
    plt.show()


def plot_figure6_metric_networks(
    network_data: Dict[str, Any],
    save_path: Optional[str] = None,
):
    """
    Figure 6: Metric Network Graphs.

    Creates 4-panel network visualization showing metric correlations:
    - Panel A: Peer-Reviewed corpus network
    - Panel B: Community corpus network
    - Panel C: MIMIC-IV corpus network
    - Panel D: Combined cross-corpus network

    Args:
        network_data: Dict containing:
            - 'peer_reviewed': Dict with 'adjacency_matrix', 'metric_names'
            - 'community': Dict with 'adjacency_matrix', 'metric_names'
            - 'mimic': Dict with 'adjacency_matrix', 'metric_names'
            - 'combined': Dict with 'adjacency_matrix', 'metric_names'
        save_path: Optional path to save figure

    Reference: Manuscript Figure 6
    """
    fig, axes = plt.subplots(2, 2, figsize=(18, 16))
    fig.suptitle(
        "Metric Correlation Networks Across Corpora", fontsize=16, fontweight="bold"
    )

    corpora = [
        ("peer_reviewed", "A. Peer-Reviewed Corpus", axes[0, 0]),
        ("community", "B. Community Corpus", axes[0, 1]),
        ("mimic", "C. MIMIC-IV Clinical Notes", axes[1, 0]),
        ("combined", "D. Combined Cross-Corpus Network", axes[1, 1]),
    ]

    for corpus_key, title, ax in corpora:
        if corpus_key not in network_data:
            continue

        data = network_data[corpus_key]
        adjacency = np.array(data["adjacency_matrix"])
        metric_names = data["metric_names"]

        # Create graph (use absolute values and threshold)
        threshold = 0.3
        adj_filtered = np.where(np.abs(adjacency) > threshold, adjacency, 0)
        G = nx.from_numpy_array(adj_filtered)

        # Relabel nodes with metric names
        if metric_names and len(metric_names) == G.number_of_nodes():
            mapping = {i: metric_names[i] for i in range(len(metric_names))}
            G = nx.relabel_nodes(G, mapping)

        # Calculate layout
        pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)

        # Node colors by degree centrality
        centrality = nx.degree_centrality(G)
        node_colors = [centrality[node] for node in G.nodes()]

        # Draw network
        nx.draw_networkx_nodes(
            G,
            pos,
            node_color=node_colors,
            node_size=500,
            cmap="viridis",
            vmin=0,
            vmax=max(node_colors) if node_colors else 1,
            ax=ax,
        )
        nx.draw_networkx_labels(G, pos, font_size=7, font_weight="bold", ax=ax)

        # Edge colors by weight
        edges = G.edges()
        weights = [G[u][v]["weight"] for u, v in edges]
        nx.draw_networkx_edges(
            G,
            pos,
            edge_color=weights,
            width=2,
            edge_cmap=plt.cm.RdYlGn,
            edge_vmin=-1,
            edge_vmax=1,
            alpha=0.6,
            ax=ax,
        )

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.axis("off")

        # Add network statistics
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        density = nx.density(G)
        ax.text(
            0.02,
            0.98,
            f"Nodes: {n_nodes}\nEdges: {n_edges}\nDensity: {density:.3f}",
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    # Always render the figure (inline in notebooks); saving does not suppress it.
    plt.show()


def plot_figure7_intersectional_heatmap(
    intersectional_data: Dict[str, Any],
    save_path: Optional[str] = None,
):
    """
    Figure 7: Intersectional Fairness Heatmaps.

    Creates comprehensive heatmap visualization showing:
    - Main heatmap: Fairness scores across intersectional groups
    - Pareto-optimality frontier overlay
    - Marginal distributions

    Args:
        intersectional_data: Dict containing:
            - 'fairness_matrix': 2D array of fairness scores
            - 'row_labels': List of row labels (e.g., race categories)
            - 'col_labels': List of column labels (e.g., gender categories)
            - 'pareto_optimal': Boolean mask of Pareto-optimal cells
            - 'marginal_row': Marginal fairness scores by row
            - 'marginal_col': Marginal fairness scores by column
        save_path: Optional path to save figure

    Reference: Manuscript Figure 7
    """
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(
        3,
        3,
        figure=fig,
        width_ratios=[0.2, 1, 0.2],
        height_ratios=[0.2, 1, 0.05],
        hspace=0.05,
        wspace=0.05,
    )

    fig.suptitle(
        "Intersectional Fairness Analysis with Pareto-Optimality",
        fontsize=16,
        fontweight="bold",
    )

    # Main heatmap
    ax_main = fig.add_subplot(gs[1, 1])
    fairness_matrix = np.array(intersectional_data["fairness_matrix"])
    row_labels = intersectional_data["row_labels"]
    col_labels = intersectional_data["col_labels"]
    pareto_optimal = np.array(
        intersectional_data.get(
            "pareto_optimal", np.zeros_like(fairness_matrix, dtype=bool)
        )
    )

    # Plot heatmap
    im = ax_main.imshow(fairness_matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    # Add text annotations
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            value = fairness_matrix[i, j]
            color = "white" if value < 0.5 else "black"
            text = ax_main.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                color=color,
                fontsize=10,
                fontweight="bold",
            )

            # Highlight Pareto-optimal cells
            if pareto_optimal[i, j]:
                rect = plt.Rectangle(
                    (j - 0.4, i - 0.4),
                    0.8,
                    0.8,
                    fill=False,
                    edgecolor="blue",
                    linewidth=3,
                )
                ax_main.add_patch(rect)

    ax_main.set_xticks(np.arange(len(col_labels)))
    ax_main.set_yticks(np.arange(len(row_labels)))
    ax_main.set_xticklabels(col_labels, rotation=45, ha="right")
    ax_main.set_yticklabels(row_labels)
    ax_main.set_xlabel("Gender", fontsize=12, fontweight="bold")
    ax_main.set_ylabel("Race/Ethnicity", fontsize=12, fontweight="bold")

    # Top marginal (column averages)
    ax_top = fig.add_subplot(gs[0, 1], sharex=ax_main)
    marginal_col = intersectional_data.get(
        "marginal_col", np.mean(fairness_matrix, axis=0)
    )
    ax_top.bar(np.arange(len(col_labels)), marginal_col, color="steelblue", alpha=0.7)
    ax_top.set_ylim([0, 1])
    ax_top.set_ylabel("Avg", fontsize=10)
    ax_top.tick_params(axis="x", labelbottom=False)
    ax_top.grid(True, alpha=0.3, axis="y")

    # Right marginal (row averages)
    ax_right = fig.add_subplot(gs[1, 2], sharey=ax_main)
    marginal_row = intersectional_data.get(
        "marginal_row", np.mean(fairness_matrix, axis=1)
    )
    ax_right.barh(np.arange(len(row_labels)), marginal_row, color="coral", alpha=0.7)
    ax_right.set_xlim([0, 1])
    ax_right.set_xlabel("Avg", fontsize=10)
    ax_right.tick_params(axis="y", labelleft=False)
    ax_right.grid(True, alpha=0.3, axis="x")

    # Colorbar
    ax_cbar = fig.add_subplot(gs[2, 1])
    cbar = plt.colorbar(im, cax=ax_cbar, orientation="horizontal")
    cbar.set_label("Fairness Score (0=Unfair, 1=Fair)", fontsize=12)

    # Legend for Pareto-optimal cells
    from matplotlib.patches import Rectangle

    legend_elements = [
        Rectangle(
            (0, 0),
            1,
            1,
            fill=False,
            edgecolor="blue",
            linewidth=3,
            label="Pareto-Optimal",
        )
    ]
    ax_main.legend(
        handles=legend_elements,
        loc="upper left",
        bbox_to_anchor=(1.15, 1.0),
        fontsize=10,
    )

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    # Always render the figure (inline in notebooks); saving does not suppress it.
    plt.show()
