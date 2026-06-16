"""All six manuscript figures render with the bundled sample data."""
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from equimed_dss.utils import (  # noqa: E402
    generate_figure_data,
    plot_equity_radar,
    plot_geographic_dumbbell,
    plot_figure2_reliability_dashboard,
    plot_figure3_corpus_comparison,
    plot_figure4_temporal_robustness,
    plot_figure5_ethics_governance,
    plot_figure6_metric_networks,
    plot_figure7_intersectional_heatmap,
)


def test_generate_figure_data_has_all_keys():
    figs = generate_figure_data()
    assert set(figs) == {"fig2", "fig3", "fig4", "fig5", "fig6", "fig7"}


def test_all_figures_render(tmp_path):
    figs = generate_figure_data()
    plotters = {
        "fig2": plot_figure2_reliability_dashboard,
        "fig3": plot_figure3_corpus_comparison,
        "fig4": plot_figure4_temporal_robustness,
        "fig5": plot_figure5_ethics_governance,
        "fig6": plot_figure6_metric_networks,
        "fig7": plot_figure7_intersectional_heatmap,
    }
    for name, fn in plotters.items():
        out = tmp_path / f"{name}.png"
        fn(figs[name], save_path=str(out))
        assert out.exists() and out.stat().st_size > 0


# ----------------------------------------------------------------------
# v1.5.2 additions: equity radar + geographic dumbbell
# ----------------------------------------------------------------------
def test_equity_radar_returns_figure_and_saves(tmp_path):
    scores = {
        "Reliability": 0.16, "Fairness": 0.95, "Governance": 0.80,
        "Representation": 0.33, "Robustness": 0.73,
    }
    out = tmp_path / "radar.png"
    fig = plot_equity_radar(scores, reference=0.8, save_path=str(out))
    assert isinstance(fig, plt.Figure)
    assert out.exists() and out.stat().st_size > 0
    plt.close(fig)


def test_equity_radar_validates_inputs():
    with pytest.raises(ValueError):
        plot_equity_radar({"A": 0.5, "B": 0.5})           # < 3 axes
    with pytest.raises(ValueError):
        plot_equity_radar({"A": 1.5, "B": 0.2, "C": 0.3})  # out of [0, 1]


def test_geographic_dumbbell_returns_figure_and_saves(tmp_path):
    burden = {"AMRO": 0.114, "SEARO": 0.211, "AFRO": 0.150,
              "EMRO": 0.230, "EURO": 0.195, "WPRO": 0.100}
    evidence = {"AMRO": 0.780, "SEARO": 0.0, "AFRO": 0.002,
                "EMRO": 0.037, "EURO": 0.105, "WPRO": 0.077}
    out = tmp_path / "dumbbell.png"
    fig = plot_geographic_dumbbell(burden, evidence, save_path=str(out))
    assert isinstance(fig, plt.Figure)
    assert out.exists() and out.stat().st_size > 0
    plt.close(fig)


def test_geographic_dumbbell_normalizes_counts():
    # raw counts should be normalized internally (no error, returns a figure)
    fig = plot_geographic_dumbbell({"A": 80, "B": 20}, {"A": 50, "B": 50})
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
    with pytest.raises(ValueError):
        plot_geographic_dumbbell({}, {"A": 1.0})
