"""Template chi2 fit over a redshift grid: one non-negative amplitude per template and z. Fitted to the single
exposures (spec.meta['exposures'], templates.Exposed) when there are at least MIN_EXPOSURES, else to the binned
spectrum (templates.Binned). P(z) = exp(-chi2 / 2s), s = max(1, reduced chi2)."""
import numpy as np
from .templates import Binned, Exposed

_trapezoid = getattr(np, 'trapezoid', None) or np.trapz
_BINNED = {}
_EXPOSED = None
MIN_EXPOSURES = 20


def bin_width(spec):
    """Median spacing of the spectrum in ln(lambda), rounded to 1e-3."""
    return round(float(np.median(np.diff(np.log(spec.lam_um)))), 3) if spec.n > 1 else 0.0


def binned_library(library, dln):
    """The library binned to width dln, cached."""
    key = (id(library), dln)
    if key not in _BINNED or _BINNED[key][0] is not library:
        _BINNED[key] = (library, [Binned(t, dln) for t in library])
    return _BINNED[key][1]


def exposed_library(library):
    """The library as measured by single exposures, cached."""
    global _EXPOSED
    if _EXPOSED is None or _EXPOSED[0] is not library:
        _EXPOSED = (library, [Exposed(t) for t in library])
    return _EXPOSED[1]


def chi2_amp(d, m, w, amp_min=0.0):
    """Chi2 and best amplitude >= amp_min (>= 0) of each model m[..., :] for data d, weights w."""
    smm, sdm, sdd = np.sum(w * m * m, -1), np.sum(w * d * m, -1), np.sum(w * d * d)
    a = np.where(smm > 0, np.maximum(sdm / np.where(smm > 0, smm, 1.0), max(amp_min, 0.0)), 0.0)
    return sdd - 2.0 * a * sdm + a * a * smm, a


def fit_spectrum(spec, library, dz=0.002, zmax_global=7.0):
    """Best fit per class and P(z) of the winning class; dln: bin width of the binned spectrum."""
    dln = bin_width(spec)
    ex = spec.meta.get('exposures')
    if ex is not None and ex.n >= MIN_EXPOSURES:
        data, lib = ex, exposed_library(library)
    else:
        data, lib = spec, binned_library(library, dln)
    w = np.where(data.sigma > 0, 1.0 / data.sigma ** 2, 0.0)
    zgrid = np.arange(0.0, zmax_global + 1e-9, dz)
    per_class = {}
    for t in lib:
        zs = zgrid[zgrid <= t.zmax] if t.z_free else np.zeros(1)
        c, _ = chi2_amp(data.flux, t.flux_obs(data.lam_um, zs[:, None]), w, t.amp_min)
        cur = per_class.setdefault(t.cls, {'chi2': np.inf, 'chi2_z': None})
        if t.z_free:
            c_z = np.concatenate([c, np.full(len(zgrid) - len(zs), np.inf)])
            cur['chi2_z'] = c_z if cur['chi2_z'] is None else np.minimum(cur['chi2_z'], c_z)
        i = int(np.argmin(c))
        if c[i] < cur['chi2']:
            cur.update(chi2=c[i], z=zs[i], template=t.name)

    ranked = sorted(per_class, key=lambda k: per_class[k]['chi2'])
    win = per_class[ranked[0]]
    dof = max(data.n - 2, 1)
    res = {'class': ranked[0], 'z': win['z'], 'redchi2': win['chi2'] / dof, 'template': win['template'],
           'n_bins': spec.n, 'n_exp': data.n if data is ex else 0, 'zgrid': zgrid, 'dln': dln,
           'per_class': {k: {'redchi2': v['chi2'] / dof, 'z': v['z'], 'template': v['template']}
                         for k, v in per_class.items()},
           'runner_up': ranked[1] if len(ranked) > 1 else None,
           'dchi2_class': per_class[ranked[1]]['chi2'] - win['chi2'] if len(ranked) > 1 else np.inf}
    res.update(_posterior(win['chi2_z'], zgrid, res['redchi2']))
    return res


def model(spec, t, z, lam=None):
    """Binned template t at z fitted to spec with amplitude >= amp_min (as in the fit), on lam (default: the bins)."""
    w = np.where(spec.sigma > 0, spec.sigma ** -2.0, 0.0)
    _, a = chi2_amp(spec.flux, t.flux_obs(spec.lam_um, z), w, t.amp_min)
    return a * t.flux_obs(spec.lam_um if lam is None else lam, z)


def chi2_sys(spec, m, eps, npar=2):
    """Reduced chi2 of model m with a template systematic eps * m added to the errors."""
    ok = spec.sigma > 0
    return float(np.sum((spec.flux - m)[ok] ** 2 / (spec.sigma[ok] ** 2 + (eps * m[ok]) ** 2)) / max(ok.sum() - npar, 1))


def _posterior(chi2_z, zgrid, redchi2):
    """P(z) and detection significance from chi2(z)."""
    if chi2_z is None or not np.isfinite(chi2_z).any():
        return {'pz': None, 'signif': np.nan}
    chi2_z = chi2_z / max(1.0, redchi2)
    m = np.isfinite(chi2_z)
    pz = np.exp(-0.5 * (chi2_z - chi2_z[m].min()))
    pz /= _trapezoid(pz[m], zgrid[m])
    return {'pz': pz, 'signif': _detection_signif(chi2_z, zgrid)}


def peak_interval(zgrid, pz, z, frac=0.68):
    """Highest-density interval holding `frac` of the P(z) peak (between its minima) that contains z."""
    lo = hi = int(np.searchsorted(zgrid, z))
    while lo > 0 and 0 < pz[lo - 1] <= pz[lo]:
        lo -= 1
    while hi < len(pz) - 1 and 0 < pz[hi + 1] <= pz[hi]:
        hi += 1
    p = pz[lo:hi + 1]
    order = np.argsort(p)[::-1]
    keep = order[:np.searchsorted(np.cumsum(p[order]) / p.sum(), frac) + 1] + lo
    return float(zgrid[keep].min()), float(zgrid[keep].max())


def _detection_signif(chi2_z, zgrid, half_width=0.05):
    """sqrt(delta chi2) to the best solution more than half_width away in z."""
    i = int(np.argmin(chi2_z))
    far = np.isfinite(chi2_z) & (np.abs(zgrid - zgrid[i]) > half_width)
    return float(np.sqrt(max(chi2_z[far].min() - chi2_z[i], 0.0))) if far.any() else np.nan
