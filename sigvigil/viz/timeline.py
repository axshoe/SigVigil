"""
sigvigil.viz.timeline
=======================
Figure 3: Adolescent-amplified signals bar chart.
Figure 4: Reporting rate over time (temporal trends).
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd


DRUG_COLORS = {
    "topiramate": "#1F77B4",
    "valproate": "#FF7F0E",
    "amitriptyline": "#2CA02C",
    "propranolol": "#D62728",
    "candesartan": "#9467BD",
    "erenumab": "#0d7a7a",
    "fremanezumab": "#8C564B",
    "galcanezumab": "#E377C2",
}
DEFAULT_COLOR = "#AAAAAA"


def plot_amplified_signals(
    stratified_df: pd.DataFrame,
    output_path: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    top_n: int = 20,
) -> plt.Figure:
    """Figure 3: Bar chart of adolescent-amplified signals.

    Shows drug-event pairs where IC025 in the adolescent female cohort
    is significantly higher than in the full FAERS population on the
    same drug. The amplification_delta column (from run_stratified_comparison)
    is used as the primary sort key.

    Parameters
    ----------
    stratified_df : output of run_stratified_comparison().
    top_n : number of top-amplified pairs to display.
    """
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9,
    })

    df = stratified_df.copy()
    df = df[df["is_amplified"]].nlargest(top_n, "amplification_delta")

    if df.empty:
        fig, ax = plt.subplots(figsize=figsize or (10, 4))
        ax.text(0.5, 0.5, "No amplified signals detected", ha="center",
                va="center", transform=ax.transAxes, fontsize=12)
        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
        return fig

    # Dynamic figure height: taller when many bars, shorter when few
    n_bars = len(df)
    auto_height = max(3.5, min(n_bars * 1.1 + 2.0, 14))
    if figsize is None:
        figsize = (11, auto_height)

    labels = [
        f"{row['event'].capitalize()}\n({row['drug'].capitalize()})"
        for _, row in df.iterrows()
    ]
    bar_colors = df["drug"].map(DRUG_COLORS).fillna(DEFAULT_COLOR).tolist()

    fig, ax = plt.subplots(figsize=figsize)

    bars = ax.barh(
        y=range(len(df)),
        width=df["amplification_delta"],
        color=bar_colors,
        edgecolor="#333333",
        linewidth=0.5,
        alpha=0.85,
    )

    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(labels, fontsize=8.5, fontfamily="Times New Roman")
    ax.invert_yaxis()

    ax.axvline(0, color="#333333", linewidth=0.8)

    # Set x-limit with enough right-side padding to keep annotations inside axes
    max_delta = df["amplification_delta"].max()
    ax.set_xlim(left=0, right=max_delta * 1.22)

    # Annotate inside the axes (slightly left of right edge rather than beyond bar end)
    for i, (_, row) in enumerate(df.iterrows()):
        delta = row["amplification_delta"]
        # Place label inside bar if bar is wide enough, otherwise just outside
        x_pos = delta + max_delta * 0.025
        ax.text(
            x_pos,
            i,
            f"Δ={delta:.2f}",
            va="center",
            fontsize=8,
            fontfamily="Times New Roman",
            color="#222222",
            clip_on=True,
        )

    ax.set_xlabel(
        "IC025 amplification delta (target − background)",
        fontsize=10,
        fontfamily="Times New Roman",
    )
    ax.set_title(
        "Fig. 3. Adverse events with amplified pharmacovigilance signals "
        "in adolescent females vs. general FAERS population\n"
        "IC025 amplification delta = IC025(adolescent female cohort) − IC025(full FAERS). "
        "Only pairs with target IC025 > 0 shown.",
        fontsize=9, pad=10, fontfamily="Times New Roman",
    )

    # Drug color legend
    from matplotlib.patches import Patch
    seen = set()
    legend_patches = []
    for drug, color in DRUG_COLORS.items():
        if drug in df["drug"].values and drug not in seen:
            legend_patches.append(Patch(color=color, label=drug.capitalize()))
            seen.add(drug)
    if legend_patches:
        ax.legend(handles=legend_patches, title="Drug", loc="lower right",
                  fontsize=7.5, title_fontsize=8, framealpha=0.85)

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    return fig


def plot_timeline(
    db,  # FAERSDatabase
    drugs: Sequence[str],
    population_filter=None,
    output_path: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 6),
) -> plt.Figure:
    """Figure 4: Reporting rate over time by drug.

    Line chart of adolescent female migraine reports per year per drug.
    """
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9,
    })

    timeline_df = db.reporting_rate_by_year(drugs, population_filter)

    fig, ax = plt.subplots(figsize=figsize)

    for drug in drugs:
        subset = timeline_df[timeline_df["drug"] == drug].sort_values("report_year")
        if subset.empty:
            continue
        color = DRUG_COLORS.get(drug, DEFAULT_COLOR)
        ax.plot(
            subset["report_year"],
            subset["n_reports"],
            marker="o",
            markersize=4,
            linewidth=1.8,
            color=color,
            label=drug.capitalize(),
            alpha=0.9,
        )

    # Annotate key drug entry dates
    ENTRY_YEARS = {
        "erenumab": (2018, "Erenumab approved"),
        "fremanezumab": (2018, "Fremanezumab approved"),
        "galcanezumab": (2018, "Galcanezumab approved"),
    }
    y_max = timeline_df["n_reports"].max() * 1.05 if not timeline_df.empty else 10
    for drug, (yr, label) in ENTRY_YEARS.items():
        if drug in drugs:
            ax.axvline(yr, color="#888888", linewidth=0.7, linestyle=":",
                       alpha=0.7)
            ax.text(yr + 0.1, y_max * 0.9, label, fontsize=6.5,
                    rotation=90, va="top", color="#555555",
                    fontfamily="Times New Roman")

    ax.set_xlabel("Year", fontsize=10, fontfamily="Times New Roman")
    ax.set_ylabel("Reports (N)", fontsize=10, fontfamily="Times New Roman")
    ax.set_title(
        "Fig. 4. Annual FAERS report counts by drug: adolescent females with migraine (2004–2024)\n"
        "Dotted verticals mark CGRP monoclonal antibody FDA approval year (2018). "
        "Counts reflect deduplicated case-level data.",
        fontsize=9, pad=10, fontfamily="Times New Roman",
    )
    ax.legend(title="Drug", fontsize=8, title_fontsize=8.5,
              loc="upper left", framealpha=0.85)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.set_xlim(left=2003)

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    return fig