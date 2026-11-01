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
MIN_SNR = 2.0           # median S/N per bin below which a spectrum is too noisy to classify
SIGMA_Z_SYS = 0.0033   # systematic error / (1+z) added to the 68% interval, calibrated on spectroscopic QSOs
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


def _low_quality(spec, fit):
    """Too noisy for a star or galaxy class (the continuum shape does not fix the class): median S/N per bin below
    MIN_SNR, fewer than 10 bins or fewer than 6 bins with f > sigma. A QSO must still pass the line tests."""
    n_good = int(np.sum((spec.sigma > 0) & (spec.flux > spec.sigma)))
    return spec.median_snr < MIN_SNR or spec.n < 10 or n_good < 6


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


QUALITY = {
    'secure': 'no warning: QSO with two or more emission lines, or star with a clear class and S/N >= 4',
    'probable': 'classified with a warning (single line, ambiguous class or S/N 2-4), or galaxy',
    'low': 'QSO from a low_snr spectrum; its emission lines pass every test',
}
REASONS = {
    'no_data': 'fewer than 6 usable bins',
    'too_noisy': 'spectrum too noisy for a star or galaxy class (see low_snr) and no QSO lines',
    'galaxy_or_qso': 'galaxy fit, but a QSO fit is nearly as good and shows emission lines',
    'no_lines': 'QSO fit without an identifiable emission line',
    'z_disagree': 'template and emission-line redshifts disagree',
    'weak_line': 'a single emission line, too weak to fix the redshift',
    'lines_absent': 'the emission lines of the QSO fit are absent from the spectrum',
    'poor_fit': 'no template fits (reduced chi2 > 5 with a 10% template error)',
}
WARNINGS = {
    'single_line': 'QSO redshift from one emission line: possible alias',
    'class_ambiguous': 'the second-best class is within delta chi2 = 2',
    'low_snr': 'median S/N per bin < 2, fewer than 10 bins, or fewer than 6 bins with flux > 1 sigma',
    'marginal_snr': 'median S/N per bin 2-4',
}


def classify(path_or_spec, dz=0.002):
    """Classify a spectrum (path or reader.Spectrum). Returns a dict: class, z_RedX, z_lo, z_hi, quality (classified
    sources: QUALITY), reason (inconclusive: one of REASONS), warnings (';'-joined WARNINGS), star_type,
    star_type_range, star_lum, and diagnostics."""
    spec = path_or_spec if hasattr(path_or_spec, 'lam_um') else read_spectrum(path_or_spec)
    snr = spec.median_snr
    lib = _library()
    out = {'class': 'inconclusive', 'z_RedX': np.nan, 'z_lo': np.nan, 'z_hi': np.nan, 'quality': '', 'reason': 'no_data',
           'warnings': '', 'star_type': '', 'star_type_range': '', 'star_lum': '', 'median_snr': snr, 'n_bins': spec.n,
           'n_exp': 0, 'n_clip': 0, 'n_line_support': 0, 'line_signif': np.nan, 'z_lines': np.nan, 'n_lines': 0,
           'redchi2': np.nan, 'chi2_sys': np.nan, 'line_scale': np.nan, 'signif': np.nan, 'max_line_snr': np.nan,
           'red_slope': np.nan, 'template': '', 'library': _LIBRARY_ID, 'spec': spec}
    if spec.n < 6:
        return out

    fit = fit_spectrum(spec, lib, dz=dz)
    if fit['n_exp']:
        out.update(n_exp=fit['n_exp'], n_clip=spec.meta['exposures'].meta.get('n_clip', 0))
    resid = residuals(spec)
    zl, n_lines, _, _ = line_redshift(spec, resid, dz=dz)
    max_line_snr, red_slope = line_features(spec, resid)
    low_quality = _low_quality(spec, fit)

    reason, n_support, cls, z = '', 0, fit['class'], np.nan
    pc = fit['per_class']
    if low_quality and cls != 'qso':
        reason = 'too_noisy'
    elif cls == 'galaxy':
        if (pc['qso']['redchi2'] < GALAXY_OR_QSO[0] * pc['galaxy']['redchi2']
                and line_signif(spec, resid, pc['qso']['z']) > GALAXY_OR_QSO[1]):
            reason = 'galaxy_or_qso'
        else:
            z = fit['z']
    elif cls == 'qso':
        if not line_signif(spec, resid, fit['z']) > LINE_SIGNIF:
            reason = 'no_lines'
        else:
            z, n_support = _dealias(spec, resid, fit)
            out['line_signif'] = line_signif(spec, resid, z)
            if not abs(zl - z) / (1 + z) < 0.05:
                reason = 'z_disagree'
            elif n_support < 2 and (not out['line_signif'] >= SINGLE_LINE_SIGNIF
                                    or (fit['redchi2'] > MISFIT[0] and out['line_signif'] < MISFIT[1])):
                reason = 'weak_line'
    if not reason:
        t = next(t for t in binned_library(lib, fit['dln']) if t.name == fit['template'])
        m = model(spec, t, z if cls != 'star' else 0.0)
        out['chi2_sys'] = chi2_sys(spec, m, POOR_FIT[0])
        if cls == 'qso':
            out['line_scale'], deficit = line_scale(spec, m, z)
            if out['line_scale'] < LINES_ABSENT[0] and deficit > LINES_ABSENT[1]:
                reason = 'lines_absent'
        if not reason and out['chi2_sys'] > POOR_FIT[1]:
            reason = 'poor_fit'

    warnings, quality = [], ''
    if reason:
        cls, z, n_support = 'inconclusive', np.nan, 0
    else:
        if cls == 'star':
            out.update(type_star(spec, _typing(), fit['dln']))
        if cls == 'qso' and n_support < 2:
            warnings.append('single_line')
        if fit['dchi2_class'] < 2.0:
            warnings.append('class_ambiguous')
        if low_quality:
            warnings.append('low_snr')
        elif snr < 4.0:
            warnings.append('marginal_snr')
        quality = 'low' if low_quality else 'probable' if warnings or cls == 'galaxy' else 'secure'   # galaxy: stars are confused with low-z galaxies
    if np.isfinite(z) and fit['pz'] is not None:
        lo, hi = peak_interval(fit['zgrid'], fit['pz'], z)
        s = SIGMA_Z_SYS * (1 + z)
        out.update(z_lo=z - np.hypot(z - lo, s), z_hi=z + np.hypot(hi - z, s))
    out.update({'class': cls, 'z_RedX': z, 'quality': quality, 'reason': reason, 'warnings': ';'.join(warnings),
                'z_lines': zl, 'n_lines': n_lines, 'n_line_support': n_support, 'max_line_snr': max_line_snr,
                'red_slope': red_slope, 'signif': fit['signif'], 'redchi2': fit['redchi2'],
                'template': fit['template'], 'library': _LIBRARY_ID, '_fit': fit})
    return out


def _f(v, nd):
    return f'{v:.{nd}f}' if isinstance(v, (float, np.floating)) else str(v)


def summary(r):
    """A classify() result as readable text."""
    out = [f"class    : {r['class']}"]
    if r['class'] in ('qso', 'galaxy'):
        out.append(f"z_RedX   : {_f(r['z_RedX'], 3)}   (68%: {_f(r['z_lo'], 3)} - {_f(r['z_hi'], 3)})")
    if r['quality']:
        out.append(f"quality  : {r['quality']}   ({QUALITY[r['quality']]})")
    if r['reason']:
        out.append(f"reason   : {r['reason']}   ({REASONS[r['reason']]})")
    for w in filter(None, r['warnings'].split(';')):
        out.append(f"warning  : {w}   ({WARNINGS[w]})")
    if r.get('star_type'):
        out.append(f"star     : {r['star_type']}   (range {r['star_type_range']}, {r['star_lum']})")
    out.append(f"S/N      : {_f(r['median_snr'], 2)} median per bin, {r['n_bins']} bins, {r['n_exp']} exposures")
    return '\n'.join(out)
