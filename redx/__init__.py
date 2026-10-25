"""RedX: template-based redshift and classification for SPHEREx spectra.

    import redx
    r = redx.classify('spectrum.tbl')
"""
from importlib.metadata import version as _version, PackageNotFoundError
from .core import classify
from .reader import read_spectrum, Spectrum
from .templates import build_library

try:
    __version__ = _version('redx')
except PackageNotFoundError:
    __version__ = 'unknown'

__all__ = ['__version__', 'classify', 'read_spectrum', 'Spectrum', 'build_library']
