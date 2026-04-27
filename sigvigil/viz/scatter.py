"""
sigvigil.viz.scatter
======================
Figure 2: Volcano plot (log2 ROR vs. -log10 p-value).
Figure 5: EBGM vs. ROR scatter demonstrating Bayesian shrinkage.

Volcano plot design: one point per drug-event pair, colored by drug,
sized by N (case count), labeled where significant. Standard genomics
volcano format applied to pharmacovigilance; widely readable across
biomedical audiences.

EBGM vs. ROR scatter: shows how EBGM shrinks small-count estimates
toward null while leaving high-N estimates unchanged. Pedagogically
the clearest demonstration of why Bayesian shrinkage matters.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
    "nortriptyline": "#7F7F7F",
    "venlafaxine": "#BCBD22",
}
DEFAULT_COLOR = "#AAAAAA"


def plot_volcano(
    signals: pd.DataFrame,
    output_path: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 8),
    ror_col: str = "ror",
    pval_col: str = "p_value",
    n_col: str = "n_de",
    drug_col: str = "drug",
    event_col: str = "event",
    bh_col: str = "bh_qvalue",
    label_top_n: int = 15,
) -> plt.Figure:
    """Generate volcano plot (Figure 2).

    Parameters
    ----------
    signals : DataFrame from FAERSDatabase.analyze().
    output_path : save path if provided.
    label_top_n : label this many top significant points.

    Returns
    -------
    matplotlib Figure.
    """
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9,
    })

    df = signals.copy()
    df = df.dropna(subset=[ror_col, pval_col])
    df = df[df[ror_col] > 0]

    df["log2_ror"] = np.log2(df[ror_col].clip(lower=0.01))
    df["neg_log10_p"] = -np.log10(df[pval_col].clip(lower=1e-30))

    # Size by N (capped)
    n_vals = df.get(n_col, pd.Series(np.ones(len(df)))).fillna(1).clip(lower=1)
    sizes = np.sqrt(n_vals).clip(upper=12) * 8 + 15

    # Color by drug
    colors = df[drug_col].map(DRUG_COLORS).fillna(DEFAULT_COLOR)

    # Significance: BH q < 0.05 and ROR > 2
    if bh_col in df.columns:
        significant = (df[bh_col] <= 0.05) & (df["log2_ror"] > 1.0)
    else:
        significant = (df[pval_col] <= 0.05) & (df["log2_ror"] > 1.0)

    fig, ax = plt.subplots(figsize=figsize)

    # Background points
    ax.scatter(
        df.loc[~significant, "log2_ror"],
        df.loc[~significant, "neg_log10_p"],
        s=sizes[~significant],
        c=colors[~significant],
        alpha=0.35,
        linewidths=0,
        rasterized=True,
    )
    # Significant points
    ax.scatter(
        df.loc[significant, "log2_ror"],
        df.loc[significant, "neg_log10_p"],
        s=sizes[significant],
        c=colors[significant],
        alpha=0.85,
        linewidths=0.5,
        edgecolors="black",
        rasterized=False,
    )

    # Reference lines
    ax.axvline(1.0, color="#333333", linewidth=0.8, linestyle="--", alpha=0.7,
               label="ROR = 2 (log₂ = 1)")
    pval_threshold = -np.log10(0.05)
    ax.axhline(pval_threshold, color="#888888", linewidth=0.8, linestyle=":",
               alpha=0.7, label="p = 0.05")

    # Label top significant points
    if significant.any():
        top = df[significant].nlargest(label_top_n, "neg_log10_p")
        for _, row in top.iterrows():
            ax.annotate(
                row[event_col].capitalize(),
                xy=(row["log2_ror"], row["neg_log10_p"]),
                xytext=(5, 3),
                textcoords="offset points",
                fontsize=6.5,
                fontfamily="Times New Roman",
                color="#222222",
                arrowprops=dict(arrowstyle="-", color="#888888", lw=0.5),
            )

    # Legend for drugs
    drugs_in_data = df[drug_col].unique()
    legend_patches = []
    for d in drugs_in_data:
        c = DRUG_COLORS.get(d, DEFAULT_COLOR)
        legend_patches.append(
            mpatches.Patch(color=c, label=d.capitalize())
        )
    drug_legend = ax.legend(
        handles=legend_patches,
        title="Drug",
        loc="upper left",
        fontsize=7.5,
        title_fontsize=8,
        framealpha=0.85,
    )
    ax.add_artist(drug_legend)
    ax.legend(loc="upper right", fontsize=7.5, framealpha=0.85)

    ax.set_xlabel("log₂(ROR)", fontsize=10, fontfamily="Times New Roman")
    ax.set_ylabel("−log₁₀(p-value)", fontsize=10, fontfamily="Times New Roman")
    ax.set_title(
        "Fig. 2. Pharmacovigilance signal volcano plot: migraine preventives "
        "in adolescent females (FAERS 2004–2024)\n"
        "Point size ∝ √N; horizontal line: p = 0.05; vertical line: ROR = 2; "
        "labeled points: top significant drug-event pairs",
        fontsize=9, pad=10, fontfamily="Times New Roman",
    )

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    return fig


def plot_ebgm_vs_ror(
    signals: pd.DataFrame,
    drug: str = "topiramate",
    output_path: Optional[str] = None,
    figsize: Tuple[float, float] = (9, 7),
    ebgm_col: str = "ebgm",
    ror_col: str = "ror",
    n_col: str = "n_de",
    event_col: str = "event",
) -> plt.Figure:
    """Generate EBGM vs ROR scatter (Figure 5).

    Demonstrates Bayesian shrinkage: for high-N pairs, EBGM ≈ ROR.
    For low-N pairs, EBGM < ROR (shrunk toward null).
    """
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9,
    })

    df = signals.copy()
    df = df.dropna(subset=[ebgm_col, ror_col])
    df = df[df[ror_col] > 0]

    n_vals = df.get(n_col, pd.Series(np.ones(len(df)))).fillna(1).clip(lower=1)
    sizes = np.sqrt(n_vals) * 12 + 20

    cmap = plt.cm.get_cmap("YlOrRd")
    log_n = np.log1p(n_vals)
    colors = cmap((log_n - log_n.min()) / (log_n.max() - log_n.min() + 1e-6))

    fig, ax = plt.subplots(figsize=figsize)

    scatter = ax.scatter(
        df[ror_col],
        df[ebgm_col],
        s=sizes,
        c=log_n,
        cmap="YlOrRd",
        alpha=0.75,
        linewidths=0.5,
        edgecolors="#444444",
    )

    # Identity line: y = x means no shrinkage
    max_val = max(df[ror_col].max(), df[ebgm_col].max())
    line_vals = np.linspace(0, max_val * 1.05, 200)
    ax.plot(line_vals, line_vals, color="#333333", linewidth=1.0,
            linestyle="--", alpha=0.7, label="EBGM = ROR (no shrinkage)")

    # Reference lines at null
    ax.axhline(1.0, color="#888888", linewidth=0.6, linestyle=":", alpha=0.5)
    ax.axvline(1.0, color="#888888", linewidth=0.6, linestyle=":", alpha=0.5)

    # Signal threshold line at EBGM = 2 (EB05 >= 2 criterion, approximate)
    ax.axhline(2.0, color="#0d7a7a", linewidth=0.8, linestyle="--",
               alpha=0.7, label="EBGM = 2 (signal threshold)")

    # Label points — avoid overlap by only labeling well-separated points
    if len(df) > 0:
        # Sort by ROR descending to label most extreme first
        label_df = df.sort_values(ror_col, ascending=False)
        labeled_positions = []  # (x, y) already labeled
        min_dist = 0.8  # minimum Euclidean distance between labels

        for _, row in label_df.iterrows():
            x = row[ror_col]
            y = row[ebgm_col]
            name = str(row[event_col]).capitalize()

            # Check if too close to already-labeled points
            too_close = any(
                ((x - lx) ** 2 + (y - ly) ** 2) ** 0.5 < min_dist
                for lx, ly in labeled_positions
            )
            if too_close:
                continue

            labeled_positions.append((x, y))
            ax.annotate(
                name,
                xy=(x, y),
                xytext=(8, 4),
                textcoords="offset points",
                fontsize=7,
                fontfamily="Times New Roman",
                color="#222222",
                arrowprops=dict(arrowstyle="-", color="#cccccc", lw=0.5),
            )

    cbar = fig.colorbar(scatter, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("log(1 + N) — case count", fontsize=8, fontfamily="Times New Roman")
    cbar.ax.tick_params(labelsize=7)

    ax.set_xlabel("Reporting Odds Ratio (ROR)", fontsize=10, fontfamily="Times New Roman")
    ax.set_ylabel("Empirical Bayes Geometric Mean (EBGM)", fontsize=10, fontfamily="Times New Roman")
    ax.set_title(
        f"Fig. 5. EBGM vs. ROR for {drug.capitalize()}: demonstrating Bayesian shrinkage\n"
        "Each point = one adverse event. Points below identity line = shrunk toward null "
        "(low case count).\nPoint size and color scale with case count (N).",
        fontsize=9, pad=10, fontfamily="Times New Roman",
    )
    ax.legend(fontsize=7.5, framealpha=0.85)

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    return fig