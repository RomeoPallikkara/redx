"""Classification and redshift of one SPHEREx spectrum."""
import numpy as np
from .reader import read_spectrum
from .templates import build_library, library_id, typing_library
from .fit import fit_spectrum, peak_interval, binned_library, model, chi2_sys
from .lines import residuals, line_redshift, line_features, line_support, line_signif, line_scale
from .startype import type_star

_LIBRARY = None
_LIBRARY_ID = ''
_TYPING = None
SIGMA_Z_SYS = 0.0033    # systematic error / (1+z) added to the 68% interval, calibrated on spectroscopic QSOs
LINE_SIGNIF = 3.0       # line significance at z required for a QSO
MISFIT = (20.0, 12.0)   # a single-line QSO with reduced chi2 above [0] and line_signif below [1] is inconclusive
SINGLE_LINE_SIGNIF = 6.0  # a single-line QSO needs line_signif above this (weaker single lines alias)
GALAXY_OR_QSO = (1.5, 5.0)  # a galaxy is inconclusive if chi2(QSO) / chi2(galaxy) < [0] and lines at the QSO z > [1]
POOR_FIT = (0.10, 5.0)  # template systematic (fraction of the model); above this reduced chi2 no class fits: inconclusive
LINES_ABSENT = (0.3, 5.0)  # a QSO whose lines are below [0] of the template's prediction at > [1] sigma is inconclusive


def _library():
    global _LIBRARY, _LIBRARY_ID
    if _LIBRARY is None:
        _LIBRARY = build_library()
        _LIBRARY_ID = library_id(_LIBRARY)
    return _LIBRARY


def _typing():
    global _TYPING
    if _TYPING is None:
        _TYPING = typing_library()
    return _TYPING


def _dealias(spec, resid, fit):
    """Of the P(z) peaks above 0.2 of the maximum, the one supported by most lines (then the more probable,
    then the lower z). Returns (z, n_line_support)."""
    pz, zg, z0 = fit['pz'], fit['zgrid'], fit['z']
    if pz is None:
        return z0, line_support(spec, resid, z0)
    i = np.flatnonzero((pz[1:-1] >= pz[:-2]) & (pz[1:-1] > pz[2:]) & (pz[1:-1] >= 0.2 * pz.max())) + 1
    cand = [(line_support(spec, resid, z), p, -z) for z, p in zip(zg[i], pz[i])]
    cand.append((line_support(spec, resid, z0), np.interp(z0, zg, pz), -z0))
    ns, _, mz = max(cand)
    return -mz, ns


def classify(path_or_spec, dz=0.002):
    """Classify a spectrum (path or reader.Spectrum); returns a flat dict of results."""
    spec = path_or_spec if hasattr(path_or_spec, 'lam_um') else read_spectrum(path_or_spec)
    snr = spec.median_snr
    lib = _library()
    out = {'library': _LIBRARY_ID, 'class': 'inconclusive', 'z_RedX': np.nan, 'z_lo': np.nan, 'z_hi': np.nan,
           'z_lines': np.nan, 'n_lines': 0, 'n_line_support': 0, 'line_signif': np.nan, 'max_line_snr': np.nan,
           'red_slope': np.nan, 'signif': np.nan, 'redchi2': np.nan, 'chi2_sys': np.nan, 'line_scale': np.nan,
           'n_bins': spec.n, 'n_exp': 0, 'n_clip': 0, 'median_snr': snr, 'template': '', 'flags': 'too_few_bins',
           'verdict': 'inconclusive: too few good bins', 'secure': False, 'star_type': '', 'star_type_range': '',
           'star_lum': '', 'spec': spec}
    if spec.n < 6:
        return out

    fit = fit_spectrum(spec, lib, dz=dz)
    if fit['n_exp']:
        out.update(n_exp=fit['n_exp'], n_clip=spec.meta['exposures'].meta.get('n_clip', 0))
    resid = residuals(spec)
    zl, n_lines, _, _ = line_redshift(spec, resid, dz=dz)
    max_line_snr, red_slope = line_features(spec, resid)

    n_good = int(np.sum((spec.sigma > 0) & (spec.flux > spec.sigma)))
    low_quality = snr < 2.0 or spec.n < 10 or n_good < 6

    flags, n_support, cls, z = [], 0, fit['class'], np.nan
    pc = fit['per_class']
    if low_quality:
        cls, verdict = 'inconclusive', 'inconclusive: too noisy / no usable signal'
    elif cls == 'star':
        verdict = 'STAR'
    elif cls == 'galaxy':
        if (pc['qso']['redchi2'] < GALAXY_OR_QSO[0] * pc['galaxy']['redchi2']
                and line_signif(spec, resid, pc['qso']['z']) > GALAXY_OR_QSO[1]):
            cls, verdict = 'inconclusive', 'inconclusive: galaxy or QSO'
            flags.append('galaxy_or_qso')
        else:
            z = fit['z']
            verdict = f'GALAXY z={z:.3f}'
    if cls == 'qso' and not line_signif(spec, resid, fit['z']) > LINE_SIGNIF:
        cls, verdict = 'inconclusive', 'inconclusive: no identifiable emission line'
    if cls == 'qso':
        z, n_support = _dealias(spec, resid, fit)
        out['line_signif'] = line_signif(spec, resid, z)
        if not abs(zl - z) / (1 + z) < 0.05:
            cls, verdict = 'inconclusive', 'inconclusive: template and line redshifts disagree'
            flags.append('z_lines_disagree')
        elif fit['redchi2'] > MISFIT[0] and n_support < 2 and out['line_signif'] < MISFIT[1]:
            cls, verdict = 'inconclusive', 'inconclusive: poor fit with one weak line'
            flags.append('misfit_single_line')
        elif n_support < 2 and not out['line_signif'] >= SINGLE_LINE_SIGNIF:
            cls, verdict = 'inconclusive', 'inconclusive: one weak line'
            flags.append('weak_single_line')
        else:
            if n_support < 2:
                flags.append('single_line_alias_risk')
            out['secure'] = n_support >= 2
            verdict = f'QSO z={z:.3f} ({n_support} lines, ' + ('secure)' if out['secure'] else 'check the redshift)')
        if cls != 'qso':
            z, n_support = np.nan, 0
    if cls != 'inconclusive':
        t = next(t for t in binned_library(lib, fit['dln']) if t.name == fit['template'])
        m = model(spec, t, z if cls != 'star' else 0.0)
        out['chi2_sys'] = chi2_sys(spec, m, POOR_FIT[0])
        if cls == 'qso':
            out['line_scale'], deficit = line_scale(spec, m, z)
            if out['line_scale'] < LINES_ABSENT[0] and deficit > LINES_ABSENT[1]:
                cls, verdict = 'inconclusive', 'inconclusive: the emission lines of the fit are absent'
                flags.append(f"lines_absent(scale={out['line_scale']:.2f})")
        if cls != 'inconclusive' and out['chi2_sys'] > POOR_FIT[1]:
            cls, verdict = 'inconclusive', 'inconclusive: no class fits well'
            flags.append(f"poor_fit(chi2_sys={out['chi2_sys']:.1f})")
        if cls == 'inconclusive':
            z, n_support, out['secure'] = np.nan, 0, False
    if cls == 'star':
        out.update(type_star(spec, _typing(), fit['dln']))
    if fit['dchi2_class'] < 2.0:
        flags.append(f"class_ambiguous(vs_{fit['runner_up']})")
    if 2.0 <= snr < 4.0:
        flags.append('marginal_snr')
    if np.isfinite(z) and fit['pz'] is not None:
        lo, hi = peak_interval(fit['zgrid'], fit['pz'], z)
        s = SIGMA_Z_SYS * (1 + z)
        out.update(z_lo=z - np.hypot(z - lo, s), z_hi=z + np.hypot(hi - z, s))
    out.update({'class': cls, 'z_RedX': z, 'z_lines': zl, 'n_lines': n_lines,
                'n_line_support': n_support, 'max_line_snr': max_line_snr, 'red_slope': red_slope,
                'signif': fit['signif'], 'redchi2': fit['redchi2'], 'template': fit['template'],
                'flags': ';'.join(flags) or 'ok', 'verdict': verdict, '_fit': fit})
    return out
