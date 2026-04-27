"""
sigvigil.stats.ic
===================
Information Component (IC) and IC025 lower credible bound.

Source: Norén GN, Orre R, Bate A, Edwards IR. (2006). Duplicate detection
in adverse drug reaction surveillance. Drug Saf. 29(2):131-41.

Also: Bate A, Lindquist M, Edwards IR, et al. (1998). A Bayesian neural
network method for adverse drug reaction signal generation. Eur J Clin
Pharmacol. 54(4):315-21.

Mathematical specification
--------------------------
Let:
    N     = total cases in analysis population
    N_d   = cases reporting drug d
    N_e   = cases reporting event e
    N_de  = cases reporting both d and e

Expected count under independence:
    E_de = (N_d * N_e) / N

Information Component:
    IC = log2((N_de + 0.5) / (E_de + 0.5))

Variance approximation (shrinkage toward 0):
    var_IC025 = 1 / ((N_de + 0.5) * ln(2)^2) * (1 - (N_de + 0.5) / (N + 1))

Lower 95% credible bound:
    IC025 = IC - 1.96 * sqrt(var_IC025)

Signal threshold: IC025 > 0 AND N_de >= 3

Interpretation: IC > 0 means the observed drug-event count exceeds
expectation under independence. IC = 1 means observed count is twice
expected. IC025 > 0 means even the lower credible bound exceeds parity.
"""

from __future__ import annotations

import math
from typing import Dict

import numpy as np


def compute_ic(
    n_de: int,
    n_d: int,
    n_e: int,
    n_total: int,
) -> Dict[str, float]:
    """Compute IC and IC025 for a single drug-event pair.

    Parameters
    ----------
    n_de : cases with both drug and event.
    n_d : cases with drug (regardless of event).
    n_e : cases with event (regardless of drug).
    n_total : total cases in analysis population.

    Returns
    -------
    dict with keys: ic, ic025, e_de, signal.
    """
    if n_total == 0:
        return {"ic": np.nan, "ic025": np.nan, "e_de": np.nan, "signal": False}

    e_de = (n_d * n_e) / n_total

    # IC with shrinkage additive (0.5 to numerator and denominator)
    ic = math.log2((n_de + 0.5) / (e_de + 0.5))

    # Variance approximation from Norén 2006
    # var(IC025) = 1 / ((N_de + 0.5) * ln(2)^2) * (1 - (N_de + 0.5)/(N+1))
    ln2_sq = math.log(2) ** 2
    shrinkage_factor = 1.0 - (n_de + 0.5) / (n_total + 1.0)
    var_ic025 = max(shrinkage_factor / ((n_de + 0.5) * ln2_sq), 0.0)

    ic025 = ic - 1.96 * math.sqrt(var_ic025)

    signal = (ic025 > 0.0) and (n_de >= 3)

    return {
        "ic": ic,
        "ic025": ic025,
        "e_de": e_de,
        "signal": bool(signal),
    }
