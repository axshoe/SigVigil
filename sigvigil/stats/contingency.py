"""
sigvigil.stats.contingency
============================
Build the 2x2 contingency table required for disproportionality analysis.

For a given drug d and adverse event e:

             event e   not event e
drug d       n_de      n_d_not_e
not drug d   n_not_d_e n_not_d_not_e

All cell counts are derived by direct set intersection from the case-level
data, so the numbers are exact (no approximation).
"""

from __future__ import annotations

from typing import Dict
import pandas as pd


def build_contingency(
    drug: str,
    event: str,
    drugs_df: pd.DataFrame,
    reactions_df: pd.DataFrame,
    n_total_cases: int,
) -> Dict[str, int]:
    """Compute the four cells of the 2x2 table for (drug, event).

    Parameters
    ----------
    drug : canonical drug name (must match drugs_df['drug_name_normalized']).
    event : MedDRA preferred term (must match reactions_df['pt_name']).
    drugs_df : one row per (caseid, drug_name_normalized) link.
    reactions_df : one row per (caseid, pt_name) link.
    n_total_cases : total number of unique cases in the analysis population.

    Returns
    -------
    dict with keys: n_de, n_d_not_e, n_not_d_e, n_not_d_not_e, n_d, n_e.
    """
    drug_cases: set = set(
        drugs_df.loc[drugs_df["drug_name_normalized"] == drug, "caseid"]
    )
    event_cases: set = set(
        reactions_df.loc[reactions_df["pt_name"] == event, "caseid"]
    )

    n_de = len(drug_cases & event_cases)
    n_d = len(drug_cases)
    n_e = len(event_cases)

    n_d_not_e = n_d - n_de
    n_not_d_e = n_e - n_de
    n_not_d_not_e = n_total_cases - n_d - n_e + n_de

    # Sanity check
    assert n_d_not_e >= 0, f"Negative n_d_not_e for ({drug}, {event}): {n_d_not_e}"
    assert n_not_d_e >= 0, f"Negative n_not_d_e for ({drug}, {event}): {n_not_d_e}"

    return {
        "n_de": n_de,
        "n_d_not_e": max(n_d_not_e, 0),
        "n_not_d_e": max(n_not_d_e, 0),
        "n_not_d_not_e": max(n_not_d_not_e, 0),
        "n_d": n_d,
        "n_e": n_e,
    }
