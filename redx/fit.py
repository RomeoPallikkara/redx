"""Template chi2 fit over a redshift grid: one non-negative amplitude per template and z, templates binned as the
spectrum (templates.Binned). P(z) = exp(-chi2 / 2s), s = max(1, reduced chi2)."""
import numpy as np
from .templates import Binned

_trapezoid = getattr(np, 'trapezoid', None) or np.trapz
_BINNED = {}


def bin_width(spec):
    """Median spacing of the spectrum in ln(lambda), rounded to 1e-3."""
    return round(float(np.median(np.diff(np.log(spec.lam_um)))), 3) if spec.n > 1 else 0.0


def binned_library(library, dln):
    """The library binned to width dln, cached."""
    key = (id(library), dln)
    if key not in _BINNED or _BINNED[key][0] is not library:
        _BINNED[key] = (library, [Binned(t, dln) for t in library])
    return _BINNED[key][1]


def chi2_amp(d, m, w):
    """Chi2 and best non-negative amplitude of each model m[..., :] for data d, weights w."""
    smm, sdm, sdd = np.sum(w * m * m, -1), np.sum(w * d * m, -1), np.sum(w * d * d)
    ok = (smm > 0) & (sdm > 0)
    smm = np.where(ok, smm, 1.0)
    return np.where(ok, sdd - sdm * sdm / smm, sdd), np.where(ok, sdm / smm, 0.0)


def fit_spectrum(spec, library, dz=0.002, zmax_global=7.0):
    """Best fit per class, P(z) of the winning class and, if a star or galaxy wins, the QSO blend fit."""
    w = np.where(spec.sigma > 0, 1.0 / spec.sigma ** 2, 0.0)
    zgrid = np.arange(0.0, zmax_global + 1e-9, dz)
    dln = bin_width(spec)
    per_class = {}
    for t in binned_library(library, dln):
        zs = zgrid[zgrid <= t.zmax] if t.z_free else np.zeros(1)
        c, _ = chi2_amp(spec.flux, t.flux_obs(spec.lam_um, zs[:, None]), w)
        cur = per_class.setdefault(t.cls, {'chi2': np.inf, 'chi2_z': None})
        if t.z_free:
            c_z = np.concatenate([c, np.full(len(zgrid) - len(zs), np.inf)])
            cur['chi2_z'] = c_z if cur['chi2_z'] is None else np.minimum(cur['chi2_z'], c_z)
        i = int(np.argmin(c))
        if c[i] < cur['chi2']:
            cur.update(chi2=c[i], z=zs[i], template=t.name)

    ranked = sorted(per_class, key=lambda k: per_class[k]['chi2'])
    win = per_class[ranked[0]]
    dof = max(spec.n - 2, 1)
    res = {'class': ranked[0], 'z': win['z'], 'redchi2': win['chi2'] / dof, 'template': win['template'],
           'n_bins': spec.n, 'zgrid': zgrid, 'dln': dln,
           'per_class': {k: {'redchi2': v['chi2'] / dof, 'z': v['z'], 'template': v['template']}
                         for k, v in per_class.items()},
           'runner_up': ranked[1] if len(ranked) > 1 else None,
           'dchi2_class': per_class[ranked[1]]['chi2'] - win['chi2'] if len(ranked) > 1 else np.inf}
    res.update(_posterior(win['chi2_z'], zgrid, res['redchi2']))
    if 'qso' in per_class and ranked[0] != 'qso':
        b = _blend(spec, w, binned_library(library, dln), zgrid)
        b.update(_posterior(b['chi2_z'], zgrid, b['chi2'] / max(spec.n - 3, 1)), zgrid=zgrid)
        b['dchi2'] = (win['chi2'] - b['chi2']) / max(1.0, b['chi2'] / max(spec.n - 3, 1))
        res['blend'] = b
    return res


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


def _blend(spec, w, lib, zgrid):
    """QSO template at z plus a z = 0 star or galaxy template (a neighbour in the same pixel), both amplitudes
    positive: chi2(z), the best pair and the QSO share of the model flux."""
    d = spec.flux
    other = [t for t in lib if t.cls != 'qso']
    S = np.array([t.flux_obs(spec.lam_um, 0.0) for t in other])
    sss, sds, sdd = S ** 2 @ w, S @ (w * d), np.sum(w * d * d)
    out = {'chi2': np.inf, 'chi2_z': np.full(len(zgrid), np.inf)}
    for t in (t for t in lib if t.cls == 'qso'):
        zs = zgrid[zgrid <= t.zmax]
        Q = t.flux_obs(spec.lam_um, zs[:, None])
        sqq, sdq, sqs = Q ** 2 @ w, Q @ (w * d), (Q * w) @ S.T
        det = sqq[:, None] * sss - sqs ** 2
        with np.errstate(divide='ignore', invalid='ignore'):
            a = (sdq[:, None] * sss - sds * sqs) / det
            b = (sds * sqq[:, None] - sdq[:, None] * sqs) / det
        c = np.where((det > 0) & (a > 0) & (b > 0), sdd - a * sdq[:, None] - b * sds, np.inf)
        out['chi2_z'][:len(zs)] = np.minimum(out['chi2_z'][:len(zs)], c.min(1))
        i, k = np.unravel_index(np.argmin(c), c.shape)
        if c[i, k] < out['chi2']:
            fq, fs = a[i, k] * Q[i].sum(), b[i, k] * S[k].sum()
            out.update(chi2=c[i, k], z=zs[i], template=t.name, contaminant=other[k].name, qso_frac=fq / (fq + fs))
    return out


def _detection_signif(chi2_z, zgrid, half_width=0.05):
    """sqrt(delta chi2) to the best solution more than half_width away in z."""
    i = int(np.argmin(chi2_z))
    far = np.isfinite(chi2_z) & (np.abs(zgrid - zgrid[i]) > half_width)
    return float(np.sqrt(max(chi2_z[far].min() - chi2_z[i], 0.0))) if far.any() else np.nan
