"""Emission lines from the S/N residual above a smooth continuum, independent of the templates."""
import numpy as np
from .lines_data import STRONG_LINES, SPHEREX_MIN_UM, SPHEREX_MAX_UM

_REST_A = np.array([lam for _, lam, _ in STRONG_LINES])
_WEIGHT = np.array([ew / lam for _, lam, ew in STRONG_LINES])


def _continuum(lam_um, flux):
    """Cubic in log-log space, refitted without points more than 2 sigma above it."""
    m = flux > 0
    if m.sum() < 6:
        return np.full_like(flux, np.nanmedian(flux[flux > 0]) if (flux > 0).any() else 0.0)
    x = np.log10(lam_um)
    lx, ly = x[m], np.log10(flux[m])
    keep = np.ones(len(lx), bool)
    for _ in range(4):
        c = np.polyfit(lx[keep], ly[keep], min(3, keep.sum() - 1))
        resid = ly - np.polyval(c, lx)
        s = np.std(resid[keep]) or 1e-6
        new = resid < 2.0 * s
        if new.sum() == keep.sum() or new.sum() < 5:
            break
        keep = new
    return np.power(10.0, np.polyval(c, x))


def residuals(spec):
    """S/N residual above the continuum."""
    return np.where(spec.sigma > 0, (spec.flux - _continuum(spec.lam_um, spec.flux)) / spec.sigma, 0.0)


def line_features(spec, resid):
    """Largest residual and d log f_nu / d log lambda over 2.3-5 um."""
    lam, fl = spec.lam_um, spec.flux
    red = lam >= 2.3
    slope = np.nan
    if red.sum() >= 4 and np.all(fl[red] > 0):
        slope = np.polyfit(np.log10(lam[red]), np.log10(fl[red]), 1)[0]
    return float(np.max(resid)), float(slope)


def _line_snr(spec, resid, z):
    """Residual at each strong line for each z, shape (nz, nlines); nan outside SPHEREx."""
    lo = np.multiply.outer(1.0 + np.atleast_1d(z), _REST_A) / 1e4
    v = np.interp(lo, spec.lam_um, resid, left=0.0, right=0.0)
    return np.where((lo >= SPHEREX_MIN_UM) & (lo <= SPHEREX_MAX_UM), v, np.nan)


def line_support(spec, resid, z, snr=3.0):
    """Number of strong lines with residual > snr at z."""
    return int(np.sum(_line_snr(spec, resid, z)[0] > snr))


def _matched(v, min_lines):
    """sum w r / sqrt(sum w^2) over lines inside SPHEREx, w = W / lambda; -inf with fewer than min_lines."""
    inside = np.isfinite(v)
    norm = np.sqrt(np.sum(np.where(inside, _WEIGHT ** 2, 0.0), -1))
    return np.where(inside.sum(-1) >= min_lines, np.nansum(_WEIGHT * v, -1) / np.where(norm > 0, norm, 1.0), -np.inf)


def line_signif(spec, resid, z):
    """Matched-filter significance of the strong lines at z."""
    s = float(_matched(_line_snr(spec, resid, z), 1)[0])
    return s if np.isfinite(s) else np.nan


def line_redshift(spec, resid, dz=0.002, zmax=7.0):
    """z maximising the matched filter over at least two lines; returns (z, lines above 2 sigma, score, zgrid)."""
    zgrid = np.arange(0.0, zmax + 1e-9, dz)
    v = _line_snr(spec, resid, zgrid)
    score = _matched(v, 2)
    j = int(np.argmax(score))
    return zgrid[j], int(np.sum(v[j] > 2.0)), score, zgrid
