"""
sigvigil.viz.heatmap
======================
Figure 1: Drug x adverse event signal landscape heatmap.

Rows = adverse event categories (metabolic, neurological, psychiatric,
       reproductive, dermatological, sleep).
Columns = drugs (topiramate, valproate, amitriptyline, propranolol,
          candesartan, erenumab, fremanezumab, galcanezumab).
Color = IC025 value (blue = below null, white = near null, red = strong signal).
Cell annotations = ROR where signal is significant.

Design standards: Times New Roman, teal/blue accent (#0d7a7a), journal-publishable.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd


# Canonical adverse event categories for migraine preventives
AE_CATEGORIES: Dict[str, List[str]] = {
    "Metabolic": [
        "weight decreased", "weight increased", "decreased appetite",
        "anorexia", "dehydration",
    ],
    "Neurological": [
        "paraesthesia", "memory impairment", "cognitive disorder",
        "confusional state", "somnolence", "dizziness", "headache",
    ],
    "Psychiatric": [
        "depression", "anxiety", "suicidal ideation", "insomnia",
        "mood altered", "irritability",
    ],
    "Reproductive": [
        "amenorrhoea", "dysmenorrhoea", "menstrual disorder",
        "polycystic ovaries", "irregular menstruation",
    ],
    "Dermatological": [
        "alopecia", "rash", "hyperhidrosis", "urticaria", "pruritus",
    ],
    "Sleep": [
        "sleep disorder", "insomnia", "middle insomnia", "hypersomnia",
        "sleep disturbance",
    ],
}

DRUG_ORDER = [
    "topiramate", "valproate", "amitriptyline",
    "propranolol", "candesartan", "erenumab",
    "fremanezumab", "galcanezumab",
]

DRUG_LABELS = {
    "topiramate": "Topiramate",
    "valproate": "Valproate",
    "amitriptyline": "Amitriptyline",
    "propranolol": "Propranolol",
    "candesartan": "Candesartan",
    "erenumab": "Erenumab",
    "fremanezumab": "Fremanezumab",
    "galcanezumab": "Galcanezumab",
}


def plot_heatmap(
    signals: pd.DataFrame,
    output_path: Optional[str] = None,
    figsize: Tuple[float, float] = (14, 10),
    ic025_col: str = "ic025",
    ror_col: str = "ror",
    signal_col: str = "ror_signal",
    vmin: float = -1.5,
    vmax: float = 3.0,
) -> plt.Figure:
    """Generate the signal landscape heatmap (Figure 1).

    Parameters
    ----------
    signals : DataFrame from FAERSDatabase.analyze() with columns:
              drug, event, ic025, ror, ror_signal.
    output_path : if provided, save figure to this path.
    figsize : figure dimensions in inches.
    ic025_col, ror_col, signal_col : column names in signals DataFrame.
    vmin, vmax : IC025 color range.

    Returns
    -------
    matplotlib Figure.
    """
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
    })

    # Build event list in category order
    all_events = []
    category_boundaries = {}  # event_name -> category label
    for cat, events in AE_CATEGORIES.items():
        for e in events:
            if e not in all_events:
                all_events.append(e)
                category_boundaries[e] = cat

    # Filter to events present in signals
    present_events = set(signals["event"].str.lower())
    all_events = [e for e in all_events if e in present_events]

    # Filter drugs to those present
    present_drugs = set(signals["drug"].str.lower())
    drugs = [d for d in DRUG_ORDER if d in present_drugs]

    if not all_events or not drugs:
        # Fallback: use whatever is in the DataFrame
        all_events = sorted(signals["event"].unique())
        drugs = sorted(signals["drug"].unique())

    # Build matrix: rows = events, cols = drugs
    ic025_matrix = np.full((len(all_events), len(drugs)), np.nan)
    ror_matrix = np.full((len(all_events), len(drugs)), np.nan)
    sig_matrix = np.zeros((len(all_events), len(drugs)), dtype=bool)

    event_idx = {e: i for i, e in enumerate(all_events)}
    drug_idx = {d: i for i, d in enumerate(drugs)}

    for _, row in signals.iterrows():
        e = str(row["event"]).lower()
        d = str(row["drug"]).lower()
        if e in event_idx and d in drug_idx:
            ei, di = event_idx[e], drug_idx[d]
            ic025_matrix[ei, di] = row.get(ic025_col, np.nan)
            ror_matrix[ei, di] = row.get(ror_col, np.nan)
            sig_matrix[ei, di] = bool(row.get(signal_col, False))

    # Colormap: blue-white-red, centered at 0
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "sigvigil",
        ["#2166ac", "#92c5de", "#f7f7f7", "#f4a582", "#d6604d", "#b2182b"],
        N=256,
    )
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(
        ic025_matrix,
        cmap=cmap,
        norm=norm,
        aspect="auto",
        interpolation="nearest",
    )

    # Annotate cells with ROR where significant
    for ei in range(len(all_events)):
        for di in range(len(drugs)):
            if sig_matrix[ei, di] and not np.isnan(ror_matrix[ei, di]):
                ror_val = ror_matrix[ei, di]
                ic_val = ic025_matrix[ei, di]
                text_color = "white" if abs(ic_val) > 1.0 else "black"
                ax.text(
                    di, ei, f"{ror_val:.1f}",
                    ha="center", va="center",
                    fontsize=7, color=text_color, fontweight="bold",
                    fontfamily="Times New Roman",
                )

    # Axes
    drug_labels = [DRUG_LABELS.get(d, d.capitalize()) for d in drugs]
    ax.set_xticks(range(len(drugs)))
    ax.set_xticklabels(drug_labels, rotation=35, ha="right", fontsize=9)

    ax.set_yticks(range(len(all_events)))
    ax.set_yticklabels([e.capitalize() for e in all_events], fontsize=8)

    # Category separators
    drawn_cats = set()
    for i, evt in enumerate(all_events):
        cat = category_boundaries.get(evt, "")
        if cat not in drawn_cats and i > 0:
            ax.axhline(i - 0.5, color="#333333", linewidth=0.8, linestyle="--", alpha=0.5)
            drawn_cats.add(cat)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("IC025 (lower 95% credible bound)", fontsize=9, fontfamily="Times New Roman")
    cbar.ax.tick_params(labelsize=8)

    ax.set_title(
        "Fig. 1. Pharmacovigilance signal landscape: migraine preventives "
        "in adolescent females (FAERS 2004–2024)\n"
        "Color = IC025; cell annotation = ROR (shown where ROR lower 95% CI > 1, n ≥ 3)",
        fontsize=9, pad=12, fontfamily="Times New Roman",
    )

    ax.set_xlabel("Drug", fontsize=10, fontfamily="Times New Roman")
    ax.set_ylabel("Adverse event", fontsize=10, fontfamily="Times New Roman")

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")

    return fig
