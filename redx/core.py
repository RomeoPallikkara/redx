"""Classification and redshift of one SPHEREx spectrum."""
import numpy as np
from .reader import read_spectrum
from .templates import build_library
from .fit import fit_spectrum, peak_interval
from .lines import residuals, line_redshift, line_features, line_support, line_signif

_LIBRARY = None
BLEND_DCHI2 = 49.0      # scaled chi2 gain for a star/galaxy fit to become a blended QSO (inf: off)
SIGMA_Z_SYS = 0.0036    # systematic error / (1+z) added to the 68% interval, calibrated on spectroscopic QSOs
LINE_SIGNIF = 3.0       # line significance at z required for a QSO


def _library():
    global _LIBRARY
    if _LIBRARY is None:
        _LIBRARY = build_library()
    return _LIBRARY


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
    out = {'class': 'inconclusive', 'z_RedX': np.nan, 'z_lo': np.nan, 'z_hi': np.nan, 'z_lines': np.nan,
           'n_lines': 0, 'n_line_support': 0, 'line_signif': np.nan, 'max_line_snr': np.nan, 'red_slope': np.nan,
           'signif': np.nan, 'redchi2': np.nan, 'n_bins': spec.n, 'median_snr': snr, 'template': '',
           'flags': 'too_few_bins', 'verdict': 'inconclusive: too few good bins', 'spec': spec}
    if spec.n < 6:
        return out

    fit = fit_spectrum(spec, _library(), dz=dz)
    resid = residuals(spec)
    zl, n_lines, _, _ = line_redshift(spec, resid, dz=dz)
    max_line_snr, red_slope = line_features(spec, resid)

    n_good = int(np.sum((spec.sigma > 0) & (spec.flux > spec.sigma)))
    low_quality = snr < 2.0 or spec.n < 10 or n_good < 6
    red_rising = np.isfinite(red_slope) and red_slope > 0.0
    strong_pref = 9.0 < fit['dchi2_class'] < np.inf

    flags, n_support, cls, z = [], 0, fit['class'], np.nan
    blend = fit.get('blend')
    if low_quality:
        cls, verdict = 'inconclusive', 'inconclusive: too noisy / no usable signal'
    elif blend and blend['dchi2'] > BLEND_DCHI2 and line_signif(spec, resid, blend['z']) > LINE_SIGNIF:
        fit = {**fit, **blend}
        cls = 'qso'
        flags.append(f"blend({blend['contaminant']},qso_frac={blend['qso_frac']:.2f})")
    elif cls == 'star':
        verdict = 'STAR'
    elif cls == 'galaxy':
        z = fit['z']
        verdict = f'GALAXY z={z:.3f}'
    lines_at_z = cls == 'qso' and line_signif(spec, resid, fit['z']) > LINE_SIGNIF
    if cls == 'qso' and not (lines_at_z or (red_rising and strong_pref)):
        cls, verdict = 'inconclusive', 'inconclusive: no identifiable emission line'
    if cls == 'qso':
        z, n_support = _dealias(spec, resid, fit)
        out['line_signif'] = line_signif(spec, resid, z)
        agree = abs(zl - z) / (1 + z) < 0.05
        flags += [f for f, bad in (('qso_no_line', not lines_at_z),
                                   ('single_line_alias_risk', lines_at_z and n_support < 2),
                                   ('z_lines_disagree', not agree)) if bad]
        verdict = f'QSO z={z:.3f} ({n_support} lines, ' + \
            ('secure)' if lines_at_z and n_support >= 2 and agree else 'check the redshift)')
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
