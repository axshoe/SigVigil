"""sigvigil.stats: Statistical methods for disproportionality analysis."""
from sigvigil.stats.ror import compute_ror
from sigvigil.stats.ic import compute_ic
from sigvigil.stats.prr import compute_prr
from sigvigil.stats.ebgm import fit_ebgm_prior, compute_ebgm_posterior
from sigvigil.stats.corrections import bonferroni, benjamini_hochberg, storey_qvalue
from sigvigil.stats.contingency import build_contingency

__all__ = [
    "compute_ror", "compute_ic", "compute_prr",
    "fit_ebgm_prior", "compute_ebgm_posterior",
    "bonferroni", "benjamini_hochberg", "storey_qvalue",
    "build_contingency",
]
