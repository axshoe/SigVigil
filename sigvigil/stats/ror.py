"""
sigvigil.stats.ror
====================
Reporting Odds Ratio (ROR) with 95% confidence interval.

Source: Evans SJW, Waller PC, Davis S. (2001). Use of proportional reporting
ratios (PRRs) for signal generation from spontaneous adverse drug reaction
reports. Pharmacoepidemiol Drug Saf. 10(6):483-6.

Also: van Puijenbroek EP, Bate A, Leufkens HGM, et al. (2002). A comparison
of measures of disproportionality for signal detection in spontaneous
reporting systems for adverse drug reactions. Pharmacoepidemiol Drug Saf.
11(1):3-10.

Mathematical specification
--------------------------
For drug d and adverse event e, let:
    n_de        = cases reporting drug d AND event e
    n_d_not_e   = cases reporting drug d but NOT event e
    n_not_d_e   = cases NOT reporting drug d but reporting event e
    n_not_d_not_e = cases reporting NEITHER

Haldane-Anscombe correction: if any cell = 0, add 0.5 to all four cells.

ROR = (n_de * n_not_d_not_e) / (n_d_not_e * n_not_d_e)

log_ROR = ln(ROR)
SE(log_ROR) = sqrt(1/n_de + 1/n_d_not_e + 1/n_not_d_e + 1/n_not_d_not_e)

95% CI: [exp(log_ROR - 1.96*SE), exp(log_ROR + 1.96*SE)]

Signal threshold: lower 95% CI > 1 AND n_de >= 3
"""

from __future__ import annotations

import math
from typing import Dict

import numpy as np
from scipy import stats as scipy_stats


def compute_ror(
    n_de: int,
    n_d_not_e: int,
    n_not_d_e: int,
    n_not_d_not_e: int,
) -> Dict[str, float]:
    """Compute ROR, 95% CI, and signal flag.

    Parameters
    ----------
    n_de : cases with both drug and event.
    n_d_not_e : cases with drug but not event.
    n_not_d_e : cases with event but not drug.
    n_not_d_not_e : cases with neither.

    Returns
    -------
    dict with keys: ror, lo95, hi95, log_ror, se, signal, p_value.
    """
    a, b, c, d = float(n_de), float(n_d_not_e), float(n_not_d_e), float(n_not_d_not_e)

    # Haldane-Anscombe correction for zero cells
    if min(a, b, c, d) == 0:
        a += 0.5
        b += 0.5
        c += 0.5
        d += 0.5

    numerator = a * d
    denominator = b * c

    if denominator == 0:
        return {
            "ror": np.nan,
            "lo95": np.nan,
            "hi95": np.nan,
            "log_ror": np.nan,
            "se": np.nan,
            "signal": False,
            "p_value": np.nan,
        }

    ror = numerator / denominator
    log_ror = math.log(ror)
    se = math.sqrt(1.0 / a + 1.0 / b + 1.0 / c + 1.0 / d)

    lo95 = math.exp(log_ror - 1.96 * se)
    hi95 = math.exp(log_ror + 1.96 * se)

    # Two-sided p-value from the normal approximation to log-ROR
    z = log_ror / se
    p_value = 2.0 * (1.0 - scipy_stats.norm.cdf(abs(z)))

    # Signal: lower CI > 1 AND original (uncorrected) n_de >= 3
    signal = (lo95 > 1.0) and (n_de >= 3)

    return {
        "ror": ror,
        "lo95": lo95,
        "hi95": hi95,
        "log_ror": log_ror,
        "se": se,
        "signal": bool(signal),
        "p_value": p_value,
    }
