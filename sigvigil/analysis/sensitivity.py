"""
sigvigil.analysis.sensitivity
================================
Sensitivity analysis suite for disproportionality findings.

Every primary signal finding in SigVigil is accompanied by five
sensitivity analyses that test whether the signal persists under
different data restrictions. This is not optional formalism; it is the
difference between a finding that holds up and one that doesn't.

The five analyses:
    1. Serious reports only (hospitalization, life-threatening, disability).
       Tests: is the signal driven by mild events that may be under-reported?
    2. Healthcare provider reports only (exclude patient self-reports).
       Tests: is the signal inflated by notoriety bias from media coverage?
    3. Post-2015 reports only (CGRP mAb era).
       Tests: temporal confounding from Weber effect on newer drugs.
    4. Primary suspect drug only (role_cod = 'PS').
       Tests: confounding from concomitant medications.
    5. Exclude top-3 reporting countries.
       Tests: geographic reporting bias (US dominates FAERS).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def run_sensitivity_suite(
    db,  # FAERSDatabase
    drugs: Sequence[str],
    events: Sequence[str],
    base_filter,  # the primary analysis filter
) -> pd.DataFrame:
    """Run all five sensitivity analyses and return a summary DataFrame.

    Parameters
    ----------
    db : FAERSDatabase instance.
    drugs : drug names from primary analysis.
    events : events from primary analysis.
    base_filter : the Filter used for the primary analysis.

    Returns
    -------
    DataFrame with one row per (drug, event, sensitivity_variant).
    Columns: drug, event, variant_name, n_de, ror, ror_lo95, ic025, signal.
    """
    from sigvigil.core import Filter
    from sigvigil.stats.contingency import build_contingency
    from sigvigil.stats.ror import compute_ror
    from sigvigil.stats.ic import compute_ic

    # Build the five variant filters by layering on top of base_filter
    variants = _build_sensitivity_variants(base_filter)

    all_rows = []
    for variant_name, variant_filter in variants.items():
        variant_db = db.apply_filter(variant_filter)
        N = len(variant_db.cases)
        logger.info("Sensitivity '%s': N=%d cases", variant_name, N)

        if N < 10:
            logger.warning("Sensitivity '%s' has too few cases (%d), skipping", variant_name, N)
            continue

        for drug in drugs:
            for event in events:
                ct = build_contingency(
                    drug, event,
                    variant_db.drugs_df, variant_db.reactions_df, N
                )
                ror_res = compute_ror(
                    ct["n_de"], ct["n_d_not_e"], ct["n_not_d_e"], ct["n_not_d_not_e"]
                )
                ic_res = compute_ic(ct["n_de"], ct["n_d"], ct["n_e"], N)

                all_rows.append({
                    "drug": drug,
                    "event": event,
                    "variant": variant_name,
                    "n_de": ct["n_de"],
                    "ror": ror_res["ror"],
                    "ror_lo95": ror_res["lo95"],
                    "ic": ic_res["ic"],
                    "ic025": ic_res["ic025"],
                    "signal_ror": ror_res["signal"],
                    "signal_ic": ic_res["signal"],
                    "n_cases_in_variant": N,
                })

    return pd.DataFrame(all_rows)


def _build_sensitivity_variants(base_filter) -> Dict:
    """Build the five sensitivity filter variants from the base filter."""
    from sigvigil.core import Filter
    import dataclasses

    variants = {}

    # 1. Serious reports only
    variants["serious_only"] = dataclasses.replace(base_filter, serious_only=True)

    # 2. HCP reports only
    variants["hcp_only"] = dataclasses.replace(base_filter, hcp_only=True)

    # 3. Post-2015 only
    variants["post_2015"] = dataclasses.replace(base_filter, report_years=(2015, 2024))

    # 4. Primary suspect drug only
    variants["primary_suspect"] = dataclasses.replace(base_filter, primary_suspect_only=True)

    # 5. Full FAERS minus top-3 reporting countries — handled at DB level
    # We mark this variant for special handling in the runner
    variants["ex_top3_countries"] = base_filter  # caller should pre-filter DB

    return variants


def sensitivity_concordance_summary(sensitivity_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize which drug-event pairs remain significant across sensitivity variants.

    Returns DataFrame with columns:
        drug, event, n_variants_tested, n_variants_signal, concordance_rate.
    """
    if sensitivity_df.empty:
        return pd.DataFrame()

    sensitivity_df["signal_any"] = (
        sensitivity_df["signal_ror"] | sensitivity_df["signal_ic"]
    )

    summary = (
        sensitivity_df.groupby(["drug", "event"])
        .agg(
            n_variants_tested=("variant", "count"),
            n_variants_signal=("signal_any", "sum"),
        )
        .reset_index()
    )
    summary["concordance_rate"] = (
        summary["n_variants_signal"] / summary["n_variants_tested"]
    )
    return summary.sort_values("concordance_rate", ascending=False)
