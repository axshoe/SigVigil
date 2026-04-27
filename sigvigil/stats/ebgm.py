"""
sigvigil.stats.ebgm
=====================
Empirical Bayes Geometric Mean (EBGM) — DuMouchel 1999 MGPS algorithm.

Source: DuMouchel W. (1999). Bayesian data mining in large frequency tables,
with an application to the FDA Spontaneous Reporting System. Am Stat.
53(3):177-190.

This is the most mathematically demanding component of sigvigil. The EBGM
shrinks small-count estimates toward the null more aggressively than IC,
making it the most conservative signal detection method and the one preferred
by FDA in their MGPS tool.

Mathematical specification
--------------------------
MODEL:
    N_de | mu ~ Poisson(mu * E_de)
    mu ~ mixture of two Gamma distributions:
        mu ~ p * Gamma(a1, b1) + (1-p) * Gamma(a2, b2)

    where b1, b2 are rate parameters (not scale). Prior parameters
    theta = (a1, b1, a2, b2, p) are estimated by MLE across all
    drug-event cells in the database.

POSTERIOR (closed form for Gamma-Poisson conjugate):
    Given N_de = n and E_de = e, the posterior for mu given mixture
    component k is Gamma(ak + n, bk + e).

    Posterior mixing weight for component 1:
        Q1 = p * f(n; a1, b1, e) / [p * f(n; a1, b1, e) + (1-p) * f(n; a2, b2, e)]
        where f(n; a, b, e) = NegBinom(n; a, b/(b+e)) (marginal likelihood)

    Geometric mean of posterior (EBGM):
        log(EBGM) = Q1 * [digamma(a1 + n) - log(b1 + e)]
                  + (1 - Q1) * [digamma(a2 + n) - log(b2 + e)]
        EBGM = exp(log(EBGM))

    For EB05 (5th percentile), we numerically invert the CDF of the
    mixture-of-Gammas posterior.

PRIOR FITTING:
    Maximize sum over all drug-event cells of log P(N_de | theta)
    where P(N_de | theta) is the marginal likelihood (NegBinom mixture).

Signal threshold: EB05 >= 2

Interpretation: EBGM >= 2 means the geometric mean posterior estimate
of the true reporting ratio is at least 2x the null. EB05 >= 2 is the
conservative criterion requiring the lower 5th percentile to exceed 2x.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Sequence

import numpy as np
from scipy import optimize, special, stats


# ---------------------------------------------------------------------------
# Marginal likelihood helpers
# ---------------------------------------------------------------------------

def _negbinom_logpmf(n: float, alpha: float, beta: float, e: float) -> float:
    """Log PMF of NegBinom under Gamma(alpha, beta) prior and Poisson(mu*e) likelihood.

    This is the marginal P(N_de = n | alpha, beta, E_de = e).

    P(n) = C(n + alpha - 1, n) * (beta/(beta+e))^alpha * (e/(beta+e))^n

    Using log-gamma for numerical stability.
    """
    if alpha <= 0 or beta <= 0 or e < 0:
        return -np.inf
    p = beta / (beta + e)
    log_p = (
        special.gammaln(alpha + n)
        - special.gammaln(alpha)
        - special.gammaln(n + 1)
        + alpha * np.log(p)
        + n * np.log(1.0 - p + 1e-300)
    )
    return float(log_p)


def _mixture_logpmf(
    n: float, e: float, a1: float, b1: float, a2: float, b2: float, p: float
) -> float:
    """Log marginal PMF under two-component Gamma mixture prior."""
    lp1 = np.log(p + 1e-300) + _negbinom_logpmf(n, a1, b1, e)
    lp2 = np.log(1.0 - p + 1e-300) + _negbinom_logpmf(n, a2, b2, e)
    # log-sum-exp for numerical stability
    log_max = max(lp1, lp2)
    return log_max + np.log(np.exp(lp1 - log_max) + np.exp(lp2 - log_max))


def _neg_log_likelihood(
    params: np.ndarray, cells: List[Dict]
) -> float:
    """Negative log-likelihood of mixture prior given observed drug-event cells."""
    a1, log_b1, a2, log_b2, logit_p = params
    b1 = np.exp(log_b1)
    b2 = np.exp(log_b2)
    p = 1.0 / (1.0 + np.exp(-logit_p))

    # Enforce identifiability: a1 < a2
    if a1 > a2:
        return 1e10

    total = 0.0
    for cell in cells:
        n = cell["n_de"]
        e = cell["e_de"]
        if e <= 0:
            continue
        total += _mixture_logpmf(n, e, a1, b1, a2, b2, p)

    return -total


# ---------------------------------------------------------------------------
# Prior fitting
# ---------------------------------------------------------------------------

def fit_ebgm_prior(cells: List[Dict]) -> Dict[str, float]:
    """Fit the two-component Gamma mixture prior by MLE.

    Parameters
    ----------
    cells : list of dicts with 'n_de' (int) and 'e_de' (float),
            one per drug-event pair in the background database.

    Returns
    -------
    dict with keys: a1, b1, a2, b2, p (fitted prior parameters).
    """
    # Initial values from DuMouchel 1999 paper
    x0 = np.array([0.2, np.log(0.06), 1.4, np.log(1.8), 0.0])

    bounds = [
        (0.01, 20.0),   # a1
        (-5.0, 5.0),    # log_b1
        (0.01, 20.0),   # a2
        (-5.0, 5.0),    # log_b2
        (-10.0, 10.0),  # logit_p
    ]

    result = optimize.minimize(
        _neg_log_likelihood,
        x0,
        args=(cells,),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 1000, "ftol": 1e-9},
    )

    if not result.success:
        warnings.warn(
            f"EBGM prior fitting did not fully converge: {result.message}. "
            "Results may be less accurate."
        )

    a1, log_b1, a2, log_b2, logit_p = result.x
    b1 = np.exp(log_b1)
    b2 = np.exp(log_b2)
    p = 1.0 / (1.0 + np.exp(-logit_p))

    return {"a1": a1, "b1": b1, "a2": a2, "b2": b2, "p": p}


# ---------------------------------------------------------------------------
# Posterior computation for a single drug-event pair
# ---------------------------------------------------------------------------

def compute_ebgm_posterior(
    n_de: int, e_de: float, prior: Dict[str, float]
) -> Dict[str, float]:
    """Compute EBGM and EB05 for a single (drug, event) pair.

    Parameters
    ----------
    n_de : observed drug-event co-count.
    e_de : expected count under independence.
    prior : dict from fit_ebgm_prior with keys a1, b1, a2, b2, p.

    Returns
    -------
    dict with keys: ebgm, eb05, posterior_weight_1.
    """
    if e_de <= 0:
        return {"ebgm": np.nan, "eb05": np.nan, "posterior_weight_1": np.nan}

    a1, b1 = prior["a1"], prior["b1"]
    a2, b2 = prior["a2"], prior["b2"]
    p = prior["p"]

    # Posterior mixing weight for component 1
    lp1 = np.log(p + 1e-300) + _negbinom_logpmf(n_de, a1, b1, e_de)
    lp2 = np.log(1.0 - p + 1e-300) + _negbinom_logpmf(n_de, a2, b2, e_de)

    log_max = max(lp1, lp2)
    q1_unnorm = np.exp(lp1 - log_max)
    q2_unnorm = np.exp(lp2 - log_max)
    total = q1_unnorm + q2_unnorm
    q1 = q1_unnorm / total
    q2 = q2_unnorm / total

    # Posterior parameters
    a1_post = a1 + n_de
    b1_post = b1 + e_de
    a2_post = a2 + n_de
    b2_post = b2 + e_de

    # EBGM: geometric mean of mixture-of-Gammas posterior
    # E[log(mu)] = q1 * (digamma(a1_post) - log(b1_post)) + q2 * (...)
    log_ebgm = (
        q1 * (special.digamma(a1_post) - np.log(b1_post))
        + q2 * (special.digamma(a2_post) - np.log(b2_post))
    )
    ebgm = np.exp(log_ebgm)

    # EB05: 5th percentile via numerical CDF inversion
    def mixture_cdf(x: float) -> float:
        cdf1 = stats.gamma.cdf(x, a=a1_post, scale=1.0 / b1_post)
        cdf2 = stats.gamma.cdf(x, a=a2_post, scale=1.0 / b2_post)
        return float(q1 * cdf1 + q2 * cdf2)

    try:
        eb05 = optimize.brentq(
            lambda x: mixture_cdf(x) - 0.05,
            a=1e-6,
            b=max(ebgm * 10.0, 50.0),
            xtol=1e-4,
            maxiter=100,
        )
    except (ValueError, RuntimeError):
        eb05 = np.nan

    return {
        "ebgm": float(ebgm),
        "eb05": float(eb05) if not np.isnan(eb05) else np.nan,
        "posterior_weight_1": float(q1),
    }
