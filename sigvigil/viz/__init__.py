"""sigvigil.viz: Publication-quality visualization modules."""
from sigvigil.viz.heatmap import plot_heatmap
from sigvigil.viz.scatter import plot_volcano, plot_ebgm_vs_ror
from sigvigil.viz.timeline import plot_amplified_signals, plot_timeline

__all__ = [
    "plot_heatmap",
    "plot_volcano",
    "plot_ebgm_vs_ror",
    "plot_amplified_signals",
    "plot_timeline",
]
