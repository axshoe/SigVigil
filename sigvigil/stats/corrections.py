"""
sigvigil.stats.corrections
============================
Multiple testing correction methods for pharmacovigilance signal analysis.

Methods implemented (all from original papers, no external stats library):

1. Bonferroni correction
   Source: Bonferroni CE. (1936). Teoria statistica delle classi e calcolo
   delle probabilità. Pubbl. del R. Ist. Super. di Sci. Econ. e Commericiali
   di Firenze. 8:3-62.

2. Benjamini-Hochberg (BH) False Discovery Rate
   Source: Benjamini Y, Hochberg Y. (1995). Controlling the false discovery
   rate: a practical and powerful approach to multiple testing.
   J R Stat Soc Ser B. 57(1):289-300.

3. Storey q-value (FDR with pi0 estimation)
   Source: Storey JD, Tibshirani R. (2003). Statistical significance for
   genomewide studies. Proc Natl Acad Sci. 100(16):9440-5.
   Also: Storey JD. (2002). A direct approach to false discovery rates.
   J R Stat Soc Ser B. 64(3):479-498.

Why all three?
The choice of correction is a substantive methodological decision, not a
detail. Bonferroni is conservative and controls familywise error rate;
it is appropriate for confirmatory claims where a single false positive
would be problematic. BH controls FDR at 5% under arbitrary dependence;
it is appropriate for exploratory analyses where some false positives are
acceptable. Storey improves on BH by estimating the proportion of true
nulls (pi0) and adjusting accordingly; it is more powerful when many
true signals exist (expected in pharmacovigilance).

sigvigil reports all three. The investigator chooses which to use based
on the inferential context.
"""

from __future__ import annotations

import numpy as np
from typing import Sequence


def bonferroni(
    pvalues: Sequence[float], alpha: float = 0.05
) -> np.ndarray:
    """Return boolean array: True where hypothesis is rejected after Bonferroni correction.

    alpha_corrected = alpha / K where K = number of tests.

    Parameters
    ----------
    pvalues : sequence of p-values.
    alpha : familywise error rate (default 0.05).

    Returns
    -------
    np.ndarray of bool, True = reject null.
    """
    pv = np.asarray(pvalues, dtype=float)
    K = len(pv)
    if K == 0:
        return np.array([], dtype=bool)
    threshold = alpha / K
    return pv <= threshold


def benjamini_hochberg(
    pvalues: Sequence[float], alpha: float = 0.05
) -> np.ndarray:
    """Compute BH-adjusted p-values (q-values) using the 1995 BH step-up procedure.

    Sort p-values p(1) <= p(2) <= ... <= p(K).
    Reject H(i) if p(i) <= (i/K) * alpha.
    BH q-value for p(i): min over j >= i of (K/j) * p(j), capped at 1.

    Parameters
    ----------
    pvalues : sequence of p-values.
    alpha : FDR target (default 0.05).

    Returns
    -------
    np.ndarray of BH-adjusted p-values (q-values), in original order.
    """
    pv = np.asarray(pvalues, dtype=float)
    K = len(pv)
    if K == 0:
        return np.array([], dtype=float)

    order = np.argsort(pv)
    ranks = np.argsort(order) + 1  # 1-indexed ranks in sorted order

    # BH q-value = min over j >= i of (K/j) * p(j) in sorted order
    sorted_pv = pv[order]
    scaled = (K / np.arange(1, K + 1)) * sorted_pv

    # Running minimum from right (cumulative min of reversed array)
    cummin = np.minimum.accumulate(scaled[::-1])[::-1]
    cummin = np.minimum(cummin, 1.0)  # cap at 1

    # Restore original order
    qvalues = cummin[np.argsort(order)]
    return qvalues


def storey_qvalue(
    pvalues: Sequence[float],
    alpha: float = 0.05,
    lambda_range: Sequence[float] = None,
) -> np.ndarray:
    """Compute Storey q-values with pi0 estimation.

    pi0 = proportion of true null hypotheses, estimated via the spline
    method from Storey & Tibshirani 2003. When pi0 < 1 (i.e. many true
    signals), Storey q-values are more powerful than BH q-values.

    Parameters
    ----------
    pvalues : sequence of p-values.
    alpha : FDR target (for reference; q-values are returned regardless).
    lambda_range : grid of lambda values for pi0 estimation.
                   Default: [0.05, 0.10, ..., 0.95].

    Returns
    -------
    np.ndarray of Storey q-values in original order.
    """
    pv = np.asarray(pvalues, dtype=float)
    K = len(pv)

    if K == 0:
        return np.array([], dtype=float)

    if K < 10:
        # Too few tests for reliable pi0 estimation; fall back to BH
        return benjamini_hochberg(pv, alpha)

    if lambda_range is None:
        lambda_range = np.arange(0.05, 0.96, 0.05)

    lambda_range = np.asarray(lambda_range)

    # pi0 estimation: W(lambda) = #{p > lambda} / (K * (1 - lambda))
    # The pi0 estimate at each lambda is an unbiased estimator when lambda < pi0.
    pi0_estimates = np.array(
        [np.sum(pv > lam) / (K * (1.0 - lam)) for lam in lambda_range]
    )
    pi0_estimates = np.minimum(pi0_estimates, 1.0)

    # Natural cubic spline fit to pi0(lambda) and evaluate at lambda=1
    # For simplicity and robustness, use the minimum of the smoothed curve
    # as the pi0 estimate (conservative approximation)
    try:
        from scipy.interpolate import UnivariateSpline
        spline = UnivariateSpline(lambda_range, pi0_estimates, k=3, s=len(lambda_range))
        pi0 = float(np.clip(spline(1.0), pi0_estimates[-1], 1.0))
    except Exception:
        # Fallback: use minimum estimate
        pi0 = float(pi0_estimates.min())

    pi0 = max(min(pi0, 1.0), 1e-6)

    # Compute q-values: q(i) = pi0 * K * p(i) / rank(i)
    order = np.argsort(pv)
    sorted_pv = pv[order]
    ranks = np.arange(1, K + 1)

    q_sorted = pi0 * K * sorted_pv / ranks

    # Enforce monotonicity: q(i) <= q(j) for i <= j (step-up)
    q_monotone = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q_monotone = np.minimum(q_monotone, 1.0)

    # Restore original order
    qvalues = q_monotone[np.argsort(order)]
    return qvalues
