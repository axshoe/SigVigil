"""
sigvigil.analysis.stratified
==============================
Stratified analysis: compare signal strength in a target population
(adolescent females with migraine) versus the full FAERS population
on the same drugs.

The novel contribution of SigVigil is not just identifying signals in
the adolescent female cohort in isolation, but identifying signals that
are *amplified* in this population relative to the general population.
This module computes that delta.

Population-amplified signal definition:
    IC025_target - IC025_background > 0
    AND IC025_target > 0 (signal exists in target population)
    AND n_de_target >= 3

Output includes an 'amplification_delta' column for each drug-event pair:
    amplification_delta = IC025_target - IC025_background
    Positive = signal stronger in target population than in background.
    Negative = signal weaker (or absent) in target vs background.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence

import pandas as pd
import numpy as np

from sigvigil.stats.contingency import build_contingency
from sigvigil.stats.ror import compute_ror
from sigvigil.stats.ic import compute_ic
from sigvigil.stats.prr import compute_prr

logger = logging.getLogger(__name__)


def run_stratified_comparison(
    db,  # FAERSDatabase
    drugs: Sequence[str],
    events: Sequence[str],
    target_filter,  # Filter (adolescent female migraine)
    comparison_filter=None,  # Filter for background (None = full FAERS on same drugs)
) -> pd.DataFrame:
    """Compare disproportionality signals between target and background populations.

    Parameters
    ----------
    db : FAERSDatabase instance.
    drugs : drug names to analyze.
    events : MedDRA preferred terms to analyze.
    target_filter : Filter for the target population (e.g., adolescent females).
    comparison_filter : Filter for background. If None, uses all FAERS cases
                        for the same drugs (no demographic restriction).

    Returns
    -------
    DataFrame with one row per (drug, event) and columns:
        drug, event,
        n_de_target, ror_target, ic_target, ic025_target,
        n_de_background, ror_background, ic_background, ic025_background,
        amplification_delta, is_amplified.
    """
    from sigvigil.core import Filter

    target_db = db.apply_filter(target_filter)
    bg_db = db.apply_filter(comparison_filter) if comparison_filter else db

    N_target = len(target_db.cases)
    N_background = len(bg_db.cases)

    logger.info(
        "Stratified comparison: target N=%d, background N=%d",
        N_target,
        N_background,
    )

    rows = []
    for drug in drugs:
        for event in events:
            ct_t = build_contingency(
                drug, event,
                target_db.drugs_df, target_db.reactions_df, N_target
            )
            ct_b = build_contingency(
                drug, event,
                bg_db.drugs_df, bg_db.reactions_df, N_background
            )

            ror_t = compute_ror(ct_t["n_de"], ct_t["n_d_not_e"], ct_t["n_not_d_e"], ct_t["n_not_d_not_e"])
            ror_b = compute_ror(ct_b["n_de"], ct_b["n_d_not_e"], ct_b["n_not_d_e"], ct_b["n_not_d_not_e"])

            ic_t = compute_ic(ct_t["n_de"], ct_t["n_d"], ct_t["n_e"], N_target)
            ic_b = compute_ic(ct_b["n_de"], ct_b["n_d"], ct_b["n_e"], N_background)

            prr_t = compute_prr(ct_t["n_de"], ct_t["n_d"], ct_t["n_not_d_e"], N_target - ct_t["n_d"])

            delta = ic_t["ic025"] - ic_b["ic025"]

            rows.append({
                "drug": drug,
                "event": event,
                # Target population
                "n_de_target": ct_t["n_de"],
                "ror_target": ror_t["ror"],
                "ror_lo95_target": ror_t["lo95"],
                "ic_target": ic_t["ic"],
                "ic025_target": ic_t["ic025"],
                "prr_target": prr_t["prr"],
                "signal_target": ror_t["signal"] or ic_t["signal"],
                # Background population
                "n_de_background": ct_b["n_de"],
                "ror_background": ror_b["ror"],
                "ic_background": ic_b["ic"],
                "ic025_background": ic_b["ic025"],
                "signal_background": ror_b["signal"] or ic_b["signal"],
                # Amplification
                "amplification_delta": delta,
                "is_amplified": (delta > 0) and ic_t["signal"],
            })

    result = pd.DataFrame(rows)
    result = result.sort_values("amplification_delta", ascending=False)
    logger.info(
        "Stratified comparison complete: %d amplified signals out of %d pairs",
        result["is_amplified"].sum(),
        len(result),
    )
    return result
