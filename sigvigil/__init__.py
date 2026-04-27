"""
sigvigil: Open-source pharmacovigilance signal analysis for FAERS.

Implements ROR, IC, PRR, EBGM from first principles, stratified analysis,
multiple testing correction, and publication-quality visualization.

Developed by Angie Xiu | The Xiu Lab | thexiulab.org
GitHub: axshoe/sigvigil
"""

from sigvigil.core import FAERSDatabase, Filter, SignalResult
from sigvigil.data.downloader import FAERSDownloader
from sigvigil.stats.ror import compute_ror
from sigvigil.stats.ic import compute_ic
from sigvigil.stats.prr import compute_prr
from sigvigil.stats.ebgm import fit_ebgm_prior, compute_ebgm_posterior

__version__ = "1.0.0"
__author__ = "Angie Xiu"
__email__ = "thexiulab.org"

__all__ = [
    "FAERSDatabase",
    "Filter",
    "SignalResult",
    "FAERSDownloader",
    "compute_ror",
    "compute_ic",
    "compute_prr",
    "fit_ebgm_prior", "compute_ebgm_posterior",
]
