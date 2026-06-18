"""
flyspanker: tools for quickly examining fluxes and metadata in astronomical images.
"""

from .spanker import Spanker
from .headers import summarize_fits_headers

__all__ = ["Spanker", "summarize_fits_headers"]
__version__ = "0.1.0"
