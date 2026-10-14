"""Read a SPHEREx spectrum into a clean, monotonic, inverse-variance-binned form.

Accepts either
  - an IPAC `.tbl` of forced photometry as served by IRSA (columns lambda [um], flux [uJy], flux_err [uJy],
    flags, fit_ql), binned here, or
  - an already binned spectrum, as CSV or as a FITS table, with columns wavelength_angstrom,
    flux_ujy_weighted_mean, flux_ujy_weighted_sem and optionally ignored (rows with ignored != 0 are dropped).

Fluxes are f_nu in micro-Jansky, wavelengths observed-frame. Binning of a `.tbl`: keep all finite, positive-error
points -- the SPHEREx `flags` column is dominated by an informational bit, so cutting on flags==0 throws away
almost all the data -- then combine repeat detections in log-lambda bins by inverse-variance weight. Per-channel
S/N is ~2, so the continuum and lines only emerge after this binning.
"""
import numpy as np
from astropy.table import Table
from .lines_data import SPHEREX_MIN_UM, SPHEREX_MAX_UM


class Spectrum:
    """A cleaned SPHEREx spectrum. lam_um ascending; flux/sigma in micro-Jansky (f_nu)."""

    def __init__(self, lam_um, flux_ujy, sigma_ujy, meta=None):
        o = np.argsort(lam_um)
        self.lam_um = np.asarray(lam_um, float)[o]
        self.flux = np.asarray(flux_ujy, float)[o]
        self.sigma = np.asarray(sigma_ujy, float)[o]
        self.meta = meta or {}

    @property
    def n(self):
        return len(self.lam_um)

    @property
    def median_snr(self):
        m = self.sigma > 0
        return float(np.median(self.flux[m] / self.sigma[m])) if m.any() else np.nan


def _bin_inverse_variance(lam, fl, fe, nbin=44):
    """Inverse-variance mean of repeat detections in `nbin` log-lambda bins spanning the data."""
    if len(lam) < 2 or not lam.max() > lam.min():
        return np.array([]), np.array([]), np.array([])
    idx = np.clip(np.digitize(lam, np.geomspace(lam.min(), lam.max(), nbin + 1)), 1, nbin) - 1
    w = fe ** -2.0
    sw = np.bincount(idx, w, nbin)
    m = sw > 0
    return (np.bincount(idx, w * lam, nbin)[m] / sw[m], np.bincount(idx, w * fl, nbin)[m] / sw[m],
            sw[m] ** -0.5)


def _col(t, name):
    return np.asarray(t[name], float)


def _read_tbl(path, nbin=44):
    t = Table.read(path, format='ipac')
    lam, fl, fe = _col(t, 'lambda'), _col(t, 'flux'), _col(t, 'flux_err')       # um, uJy, uJy
    good = np.isfinite(fl) & np.isfinite(fe) & (fe > 0) & (lam > SPHEREX_MIN_UM) & (lam < SPHEREX_MAX_UM)
    if 'fit_ql' in t.colnames:                     # drop obviously failed forced-photometry fits
        good &= (_col(t, 'fit_ql') > 0) & (_col(t, 'fit_ql') < 5)
    meta = {'source': path, 'n_raw': int(good.sum())}
    if 'ra' in t.colnames and len(t):
        meta.update(ra=float(np.nanmedian(_col(t, 'ra'))), dec=float(np.nanmedian(_col(t, 'dec'))))
    return Spectrum(*_bin_inverse_variance(lam[good], fl[good], fe[good], nbin=nbin), meta)


def _from_binned(t, meta):
    lam = _col(t, 'wavelength_angstrom') / 1e4
    fl, fe = _col(t, 'flux_ujy_weighted_mean'), _col(t, 'flux_ujy_weighted_sem')
    keep = np.isfinite(lam) & np.isfinite(fl) & np.isfinite(fe) & (fe > 0)
    if 'ignored' in t.colnames:
        keep &= np.asarray(t['ignored'], int) == 0
    return Spectrum(lam[keep], fl[keep], fe[keep], meta)


def read_spectrum(path, nbin=44):
    """`.tbl`: IPAC forced photometry, binned here to `nbin` bins; `.csv` or `.fits`: an already binned
    spectrum (for FITS, scalar header keywords are copied into meta, lower-cased)."""
    p = str(path).lower()
    if p.endswith('.csv'):
        return _from_binned(Table.read(path, format='csv'), {'source': path})
    if p.endswith(('.fits', '.fit', '.fits.gz')):
        t = Table.read(path)
        return _from_binned(t, {'source': path, **{k.lower(): v for k, v in t.meta.items()
                                                   if isinstance(v, (str, int, float))}})
    return _read_tbl(path, nbin=nbin)
