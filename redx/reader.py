"""SPHEREx spectra: IRSA forced-photometry .tbl (binned here) or an already binned .csv / .fits; f_nu in uJy.
The single exposures behind a .tbl or a SPIFF binned_spectrum.csv (from results.csv in the same folder) are kept in
meta['exposures'] for the template fit. Of the .tbl flags only FIT_ERROR is used, for the exposures (the column is
dominated by an informational bit)."""
import os
import csv
import numpy as np
from astropy.table import Table
from .lines_data import SPHEREX_MIN_UM, SPHEREX_MAX_UM

SPIFF_BINNED = 'binned_spectrum.csv'
SYS_FRAC = 0.03       # repeatability of a single measurement (fraction of f), added to its error (Expl. Suppl. Note 1)
CLIP, CLIP_FRAC, CLIP_WINDOW = 10.0, 0.5, 0.02   # single-exposure outliers: > CLIP errors and > CLIP_FRAC of the
                                                 # median of the exposures within CLIP_WINDOW in ln lambda
FIT_ERROR_BIT = 33


class Spectrum:
    """Wavelength [um, ascending], flux and error [uJy]."""

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
    """Inverse-variance means in nbin log-lambda bins."""
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


def _clip(lam, f, e):
    ln, bad = np.log(lam), np.zeros(len(lam), bool)
    for i in range(len(lam)):
        m = np.abs(ln - ln[i]) < CLIP_WINDOW
        m[i] = False
        if m.sum() >= 3:
            med = np.median(f[m])
            d = abs(f[i] - med)
            bad[i] = d > CLIP * e[i] and d > CLIP_FRAC * abs(med)
    return bad


def _exposures(lam, f, e):
    """Spectrum of single exposures: outliers clipped, errors sqrt(e^2 + (SYS_FRAC f)^2); meta n_clip."""
    bad = _clip(lam, f, e)
    s = Spectrum(lam[~bad], f[~bad], np.sqrt(e[~bad] ** 2 + (SYS_FRAC * f[~bad]) ** 2))
    s.meta['n_clip'] = int(bad.sum())
    return s


def _spiff_exposures(binned_path):
    """Exposures of a SPIFF run from results.csv next to its binned_spectrum.csv, selected as SPIFF bins them
    (status ok, >= 3 pixels in the fit, S/N >= 0.1 unless the run is --forced); None if not available."""
    rp = os.path.join(os.path.dirname(os.path.abspath(binned_path)), 'results.csv')
    if not os.path.exists(rp):
        return None
    need = ('psf_scipy_wv_um', 'psf_scipy_flux_uJy', 'psf_scipy_flux_uJy_err', 'psf_scipy_snr',
            'fits_analysis_status', 'n_pix_used_in_fit')
    with open(rp, newline='', encoding='utf-8') as fh:
        rd = csv.DictReader(fh)
        if not set(need) <= set(rd.fieldnames or ()):
            return None
        rows = list(rd)

    def num(c):
        out = np.full(len(rows), np.nan)
        for i, r in enumerate(rows):
            try:
                out[i] = float(r[c])
            except (TypeError, ValueError):
                pass
        return out
    lam, f, e = num('psf_scipy_wv_um'), num('psf_scipy_flux_uJy'), num('psf_scipy_flux_uJy_err')
    ok = (np.isfinite(lam) & np.isfinite(f) & np.isfinite(e) & (e > 0) & (lam > SPHEREX_MIN_UM)
          & (lam < SPHEREX_MAX_UM) & np.array([r['fits_analysis_status'] == 'ok' for r in rows], bool)
          & (num('n_pix_used_in_fit') >= 3))
    forced = any(r.get('psf_scipy_method_used') == 'forced-linear' for r in rows)
    if not forced:
        ok &= num('psf_scipy_snr') >= 0.1
    s = _exposures(lam[ok], f[ok], e[ok])
    s.meta['spiff_forced'] = bool(forced)
    return s


def _read_tbl(path, nbin=44):
    t = Table.read(path, format='ipac')
    lam, fl, fe = _col(t, 'lambda'), _col(t, 'flux'), _col(t, 'flux_err')
    good = np.isfinite(fl) & np.isfinite(fe) & (fe > 0) & (lam > SPHEREX_MIN_UM) & (lam < SPHEREX_MAX_UM)
    if 'fit_ql' in t.colnames:
        good &= (_col(t, 'fit_ql') > 0) & (_col(t, 'fit_ql') < 5)
    meta = {'source': path, 'n_raw': int(good.sum())}
    if 'ra' in t.colnames and len(t):
        meta.update(ra=float(np.nanmedian(_col(t, 'ra'))), dec=float(np.nanmedian(_col(t, 'dec'))))
    ex = good.copy()
    if 'flags' in t.colnames:
        ex &= (np.asarray(t['flags'], np.int64) >> FIT_ERROR_BIT) & 1 == 0
    meta['exposures'] = _exposures(lam[ex], fl[ex], fe[ex])
    return Spectrum(*_bin_inverse_variance(lam[good], fl[good], fe[good], nbin=nbin), meta)


def _from_binned(t, meta):
    lam = _col(t, 'wavelength_angstrom') / 1e4
    fl, fe = _col(t, 'flux_ujy_weighted_mean'), _col(t, 'flux_ujy_weighted_sem')
    keep = np.isfinite(lam) & np.isfinite(fl) & np.isfinite(fe) & (fe > 0)
    if 'ignored' in t.colnames:
        keep &= np.asarray(t['ignored'], int) == 0
    return Spectrum(lam[keep], fl[keep], fe[keep], meta)


def read_spectrum(path, nbin=44):
    """Read a .tbl (binned to nbin bins), .csv or .fits spectrum; FITS header keywords go to meta."""
    p = str(path).lower()
    if p.endswith('.csv'):
        meta = {'source': path}
        if os.path.basename(p) == SPIFF_BINNED:
            meta['exposures'] = _spiff_exposures(path)
        return _from_binned(Table.read(path, format='csv'), meta)
    if p.endswith(('.fits', '.fit', '.fits.gz')):
        t = Table.read(path)
        return _from_binned(t, {'source': path, **{k.lower(): v for k, v in t.meta.items()
                                                   if isinstance(v, (str, int, float))}})
    return _read_tbl(path, nbin=nbin)
