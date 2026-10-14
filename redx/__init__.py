"""RedX -- SPHEREx spectrum redshift + classification (QSO / galaxy / star), template-based, no ML.

    from redx import classify
    r = classify("spectrum.tbl")
    print(r['class'], r['z_RedX'], r['verdict'])
"""
from .core import classify
from .reader import read_spectrum, Spectrum
from .templates import build_library

__all__ = ['classify', 'read_spectrum', 'Spectrum', 'build_library']
