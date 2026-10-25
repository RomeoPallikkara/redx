"""Spectral type of a star: the typing templates (SPIFF standards, IRTF Spectral Library if in the cache) are each
compared over the bins they measure, provided these hold >= MIN_SIGNAL of the spectrum's (S/N)^2, with the amplitude
of an IRTF star no smaller than its physical minimum (templates.D_MAX_KPC)."""
import re
import numpy as np
from .fit import binned_library

_SEQ = 'OBAFGKMLTY'
MIN_COVER = 0.6          # minimum fraction of the bins a template must measure
MIN_SIGNAL = 0.85        # minimum fraction of sum (f/sigma)^2 in those bins
TYPE_FLOOR = 1.0         # the type range is at least +- this many subtypes


def type_code(sp):
    """10 * index in OBAFGKMLTY + subtype (M0 = 60, L0 = 70); nan if none."""
    m = re.search(r'([OBAFGKMLTY])\s*(\d+(?:\.\d+)?)', sp or '')
    return 10.0 * _SEQ.index(m.group(1)) + float(m.group(2)) if m else np.nan


def code_label(c):
    if not np.isfinite(c):
        return ''
    i = int(c // 10)
    return f'{_SEQ[i]}{c - 10 * i:g}'


def lum_class(sp):
    """'subdwarf', 'giant' (I-III), 'dwarf' (IV-V, or an M, L, T, Y type without a class), '' if unknown."""
    s = (sp or '').replace(' ', '')
    if re.match(r'^(d/)?e?sd', s):
        return 'subdwarf'
    m = re.search(r'[OBAFGKMLTY]\d(?:\.\d+)?[^IV]*(IV|V|I)', s)
    if m:
        return 'dwarf' if m.group(1) in ('IV', 'V') else 'giant'
    return 'dwarf' if s[:1] and s[:1] in 'MLTY' else ''


def is_young(sp):
    """Supergiant or bright giant (luminosity class I or II, also II-III): a young disc star."""
    s = (sp or '').replace(' ', '')
    return lum_class(s) == 'giant' and re.search(r'(?<![IV])I{1,2}(?![IV])', s) is not None


def type_star(spec, typing_lib, dln):
    """Best-fitting typing template: star_type, star_type_range (types of the templates with reduced chi2 within
    2 sqrt(2/dof) max(1, best reduced chi2) of the best, widened to +-TYPE_FLOOR) and star_lum."""
    out = {'star_type': '', 'star_type_range': '', 'star_lum': ''}
    lam, f = spec.lam_um, spec.flux
    w = np.where(spec.sigma > 0, spec.sigma ** -2.0, 0.0)
    s2 = w * np.clip(f, 0.0, None) ** 2
    x = np.log(lam)
    rows = []
    for t in binned_library(typing_lib, dln):
        lo, hi = np.searchsorted(t.measured, x - dln / 2), np.searchsorted(t.measured, x + dln / 2)
        m = (hi > lo) & (x >= t.measured[0] - dln / 2) & (x <= t.measured[-1] + dln / 2) & (w > 0)
        if m.sum() < max(6, MIN_COVER * spec.n) or s2[m].sum() < MIN_SIGNAL * s2.sum():
            continue
        M = t.flux_obs(lam, 0.0)
        smm, sdm = np.sum(w[m] * M[m] ** 2), np.sum(w[m] * f[m] * M[m])
        if smm > 0 and sdm > 0:
            a = max(sdm / smm, t.amp_min)          # no farther than the star can be
            rows.append((np.sum(w[m] * (f[m] - a * M[m]) ** 2) / (m.sum() - 1), m.sum(), t))
    if not rows:
        return out
    rows.sort(key=lambda r: r[0])
    rc, dof, best = rows[0]
    ok = [t for c, _, t in rows if c <= rc + 2.0 * np.sqrt(2.0 / dof) * max(1.0, rc)]
    code = type_code(best.spectral_type)
    codes = [c for c in map(type_code, (t.spectral_type for t in ok)) if np.isfinite(c)]
    if np.isfinite(code):
        codes += [code - TYPE_FLOOR, code + TYPE_FLOOR]
    lums = {lum_class(t.spectral_type) for t in ok}
    out['star_type'] = best.spectral_type
    if codes:
        lo, hi = code_label(max(min(codes), 0.0)), code_label(min(max(codes), 99.0))
        out['star_type_range'] = lo if lo == hi else f'{lo}-{hi}'
    out['star_lum'] = lums.pop() if len(lums) == 1 else 'ambiguous'
    return out
