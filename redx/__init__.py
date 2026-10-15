"""RedX: template-based redshift and classification for SPHEREx spectra.

    import redx
    r = redx.classify('spectrum.tbl')
"""
from .core import classify
from .reader import read_spectrum, Spectrum
from .templates import build_library

__all__ = ['classify', 'read_spectrum', 'Spectrum', 'build_library']
