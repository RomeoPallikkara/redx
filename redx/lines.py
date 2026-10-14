"""Independent emission-line redshift cross-check.

Separate from the template fit: estimate a smooth continuum, take the S/N residual, and for each trial redshift
match the residual at the observed positions of the strong QSO lines inside the SPHEREx window against their
expected contrast. The best match is z_lines, a consistency check on the template redshift.
"""
import numpy as np
from .lines_data import STRONG_LINES, SPHEREX_MIN_UM, SPHEREX_MAX_UM

_REST_A = np.array([lam for _, lam, _ in STRONG_LINES])
_WEIGHT = np.array([ew / lam for _, lam, ew in STRONG_LINES])     # expected line contrast per unit ln(lambda)


def _continuum(lam_um, flux):
    """Emission-line-free continuum: sigma-clipped low-order polynomial in log-log space. Iteratively drop
    positive outliers (emission lines) so the continuum tracks the underlying shape, not the line peaks."""
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
        new = resid < 2.0 * s                      # drop positive (emission) outliers
        if new.sum() == keep.sum() or new.sum() < 5:
            break
        keep = new
    return np.power(10.0, np.polyval(c, x))


def residuals(spec):
    """S/N residual of the spectrum above its emission-line-free continuum."""
    return np.where(spec.sigma > 0, (spec.flux - _continuum(spec.lam_um, spec.flux)) / spec.sigma, 0.0)


def line_features(spec, resid):
    """Sharp-emission-line diagnostics that separate a quasar from a (line-free) star or galaxy.

    Returns max_line_snr (strongest positive residual above the line-free continuum) and red_slope
    (d log f_nu / d log lambda over 2.3-5 um: >0 rises to the red like a QSO's hot dust; <0 turns over
    like the 1.6um stellar/galaxy bump)."""
    lam, fl = spec.lam_um, spec.flux
    red = lam >= 2.3
    slope = np.nan
    if red.sum() >= 4 and np.all(fl[red] > 0):
        slope = np.polyfit(np.log10(lam[red]), np.log10(fl[red]), 1)[0]
    return float(np.max(resid)), float(slope)


def _line_snr(spec, resid, z):
    """Residual S/N at each strong line for each redshift in z (shape (nz, nlines)); nan outside SPHEREx."""
    lo = np.multiply.outer(1.0 + np.atleast_1d(z), _REST_A) / 1e4
    v = np.interp(lo, spec.lam_um, resid, left=0.0, right=0.0)
    return np.where((lo >= SPHEREX_MIN_UM) & (lo <= SPHEREX_MAX_UM), v, np.nan)


def line_support(spec, resid, z, snr=3.0):
    """Number of strong QSO lines inside SPHEREx with residual S/N > `snr` at redshift z. >=2: the redshift is
    pinned by multiple lines (secure); 1: a single-line solution that can be aliased (an Hb taken for Ha)."""
    return int(np.sum(_line_snr(spec, resid, z)[0] > snr))


def _matched(v, min_lines):
    """Matched-filter significance sum_l w_l r_l / sqrt(sum_l w_l^2) over the lines inside SPHEREx (r_l: residual
    S/N, w_l = W_l / lambda_l); -inf with fewer than `min_lines` lines inside. A line on no excess counts against."""
    inside = np.isfinite(v)
    norm = np.sqrt(np.sum(np.where(inside, _WEIGHT ** 2, 0.0), -1))
    return np.where(inside.sum(-1) >= min_lines, np.nansum(_WEIGHT * v, -1) / np.where(norm > 0, norm, 1.0), -np.inf)


def line_signif(spec, resid, z):
    """Combined significance of the strong lines at redshift z (nan if none falls inside SPHEREx)."""
    s = float(_matched(_line_snr(spec, resid, z), 1)[0])
    return s if np.isfinite(s) else np.nan


def line_redshift(spec, resid, dz=0.002, zmax=7.0):
    """Line-only redshift: the z maximising the matched-filter significance of at least two strong lines.
    Returns (z_lines, n_lines, score, zgrid); n_lines = lines with residual S/N > 2 at z_lines."""
    zgrid = np.arange(0.0, zmax + 1e-9, dz)
    v = _line_snr(spec, resid, zgrid)
    score = _matched(v, 2)
    j = int(np.argmax(score))
    return zgrid[j], int(np.sum(v[j] > 2.0)), score, zgrid
