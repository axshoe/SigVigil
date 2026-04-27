"""
sigvigil.stats.prr
====================
Proportional Reporting Ratio (PRR) with Evans 2001 signal criteria.

Source: Evans SJW, Waller PC, Davis S. (2001). Use of proportional reporting
ratios (PRRs) for signal generation from spontaneous adverse drug reaction
reports. Pharmacoepidemiol Drug Saf. 10(6):483-6.

Mathematical specification
--------------------------
For drug d and event e:
    n_de        = cases with drug d and event e
    n_d         = total cases with drug d (n_de + n_d_not_e)
    n_not_d_e   = cases with event e but not drug d
    n_not_d     = total cases not reporting drug d

PRR = (n_de / n_d) / (n_not_d_e / n_not_d)

Expected count E = n_d * (n_not_d_e / n_not_d)

Chi-squared statistic (1 df, Yates uncorrected):
    chi2 = (n_de - E)^2 / E

Signal threshold (Evans 2001): PRR >= 2 AND chi2 >= 4 AND n_de >= 3
"""

from __future__ import annotations

import math
from typing import Dict

import numpy as np


def compute_prr(
    n_de: int,
    n_d: int,
    n_not_d_e: int,
    n_not_d: int,
) -> Dict[str, float]:
    """Compute PRR, chi-squared statistic, and Evans 2001 signal flag.

    Parameters
    ----------
    n_de : cases with drug and event.
    n_d : total cases with drug.
    n_not_d_e : cases with event but not drug.
    n_not_d : total cases not reporting drug.

    Returns
    -------
    dict with keys: prr, chi2, signal.
    """
    if n_d == 0 or n_not_d == 0 or n_not_d_e == 0:
        return {"prr": np.nan, "chi2": np.nan, "signal": False}

    drug_proportion = n_de / n_d
    background_proportion = n_not_d_e / n_not_d

    if background_proportion == 0:
        return {"prr": np.nan, "chi2": np.nan, "signal": False}

    prr = drug_proportion / background_proportion

    # Chi-squared with expected count
    e = n_d * background_proportion
    if e == 0:
        return {"prr": prr, "chi2": np.nan, "signal": False}

    chi2 = ((n_de - e) ** 2) / e

    # Evans 2001 signal criteria
    signal = (prr >= 2.0) and (chi2 >= 4.0) and (n_de >= 3)

    return {
        "prr": prr,
        "chi2": chi2,
        "signal": bool(signal),
    }
