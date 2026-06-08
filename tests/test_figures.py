"""All six manuscript figures render with the bundled sample data."""
import matplotlib

matplotlib.use("Agg")

from equimed_dss.utils import (  # noqa: E402
    generate_figure_data,
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
