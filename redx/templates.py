"""Template library (f_nu on rest wavelength [A]): QSO (Selsing, z 0-7), galaxy (SWIRE, z 0-3), star (SUDA, SPIFF
standards, BT-Settl and IRTF if in the cache; z = 0). See README and data/folder_description.txt."""
import os
import csv
import functools
import numpy as np
from . import sources

_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
_HC_K = 1.438776877e8            # h c / k [A K]
SPHEREX_R = (40.0, 120.0)        # resolving power below / above 3.82 um
SPHEREX_R_SPLIT_UM = 3.82
BAND_EDGES_UM = (1.10, 1.64, 2.42, 3.82, 4.42)     # SPHEREx bands 1-6
BAND_R = (39.5, 41.3, 40.7, 35.0, 111.5, 127.7)    # lambda / bandwidth of single exposures (IRSA and SPIFF medians;
                                                   # Expl. Suppl. v2.0 Tables 29-34)
_LOG_STEP = 2e-4                 # dex
IRTF_RED_MIN_UM = 4.0            # IRTF spectra used for classification must reach this wavelength
D_MAX_KPC = {'young': 30.0, 'old': 300.0}   # farthest distance of supergiants and bright giants (I, II: Galactic
                                            # disc) and of other stars (stellar halo)
PLX_NSIGMA = 2.0                 # template distance lower bound: 1 / (parallax + PLX_NSIGMA * error)
_UJY = 1e32 / 2.99792458e14      # f_lambda [W m-2 um-1] * lambda^2 [um^2] -> f_nu [uJy]


class Template:
    """spectral_type and measured (sorted ln wavelength [um] of the measured points) are set for typing templates.
    amp_min: smallest physical amplitude (fnu in uJy at a known distance: the source is no farther than allowed)."""

    def __init__(self, name, cls, lam_rest_A, fnu, z_free=True, zmax=7.0, spherex_native=False, spectral_type=None,
                 measured=None, amp_min=0.0):
        o = np.argsort(lam_rest_A)
        self.name, self.cls, self.z_free, self.zmax = name, cls, z_free, zmax
        self.spherex_native, self.spectral_type, self.measured = spherex_native, spectral_type, measured
        self.amp_min = amp_min
        self._lx = np.log10(np.asarray(lam_rest_A, float)[o])
        self._fy = np.asarray(fnu, float)[o]

    def flux_obs(self, lam_obs_um, z):
        """f_nu at observed wavelengths for redshift z, held flat past the template edges."""
        x = np.log10(np.asarray(lam_obs_um, float) * 1e4 / (1.0 + z))
        return np.interp(x, self._lx, self._fy, left=self._fy[0], right=self._fy[-1])


def _smooth(fy, dln, R=None):
    """Convolve with a top-hat of width dln and, if R is given, a Gaussian of FWHM 1/R, both in ln(lambda)."""
    step = _LOG_STEP * np.log(10.0)
    sig = 1.0 / (2.3548 * R) / step if R else 0.0
    w = max(int(round(dln / step)), 1)
    half = int(np.ceil(w / 2 + 4 * sig)) + 1
    u = np.arange(-half, half + 1)
    k = np.exp(-0.5 * (u / sig) ** 2) if sig > 0 else (u == 0).astype(float)
    k = np.convolve(k, np.ones(w), mode='same')
    return np.convolve(np.pad(fy, half, mode='edge'), k / k.sum(), mode='valid')


class _Smoothed:
    """Template on a regular log grid, smoothed once per resolving power in rs (smoothing in ln(lambda) commutes
    with redshift; SPHEREx-native templates only by the top-hat dln)."""

    def __init__(self, tmpl, dln, rs):
        self.name, self.cls, self.z_free, self.zmax = tmpl.name, tmpl.cls, tmpl.z_free, tmpl.zmax
        self.spectral_type, self.measured, self.amp_min = tmpl.spectral_type, tmpl.measured, tmpl.amp_min
        self._lx = np.arange(tmpl._lx[0], tmpl._lx[-1] + _LOG_STEP / 2, _LOG_STEP)
        fy = np.interp(self._lx, tmpl._lx, tmpl._fy)
        self._fy = [_smooth(fy, dln, None if tmpl.spherex_native else R) for R in rs]


class Binned(_Smoothed):
    """Template averaged over a SPHEREx channel and a log-lambda bin of width dln."""

    def __init__(self, tmpl, dln):
        super().__init__(tmpl, dln, SPHEREX_R)

    def flux_obs(self, lam_obs_um, z):
        lam = np.asarray(lam_obs_um, float)
        x = np.log10(lam * 1e4 / (1.0 + z))
        blue, red = (np.interp(x, self._lx, f, left=f[0], right=f[-1]) for f in self._fy)
        return np.where(lam < SPHEREX_R_SPLIT_UM, blue, red)


class Exposed(_Smoothed):
    """Template as measured by a single exposure, f = int t S dlambda / int t dlambda (Expl. Suppl. v2.0 Eq. 22):
    convolved with a Gaussian of FWHM 1/R of its band and evaluated at the exposure's central wavelength."""

    def __init__(self, tmpl):
        super().__init__(tmpl, 0.0, BAND_R)

    def flux_obs(self, lam_obs_um, z):
        lam = np.asarray(lam_obs_um, float)
        x = np.log10(lam * 1e4 / (1.0 + z))
        band = np.searchsorted(BAND_EDGES_UM, lam)
        out = np.zeros(np.broadcast(x, band).shape)
        for b in np.unique(band):
            f = self._fy[b]
            out = np.where(band == b, np.interp(x, self._lx, f, left=f[0], right=f[-1]), out)
        return out


def template_origin(name):
    """Short source of a template, for figure labels."""
    for key, origin in (('dust', 'Selsing+2016 QSO composite + hot dust'),
                        ('selsing', 'Selsing+2016 QSO composite + SMC'), ('GAL_', 'SWIRE galaxy (Polletta+2007)'),
                        ('SUDA', 'SUDA SPHEREx dwarf atlas (Tu+2026)'), ('SPIFF', 'SPIFF SPHEREx standard (Gagne+2026)'),
                        ('IRTF', 'IRTF Spectral Library (Rayner+2009)'),
                        ('btsettl', 'BT-Settl model star (Allard+2012)')):
        if key in (name or ''):
            return origin
    return name or ''


def _igm_break(lam_rest_A):
    """Transmission 0.1 at 912-1216 A, 0 below 912 A."""
    return np.where(lam_rest_A < 912.0, 0.0, np.where(lam_rest_A < 1215.67, 0.10, 1.0))


def _flam_file(path, wave_scale=1.0):
    """(wavelength [A], f_nu) from an f_lambda file, duplicate rows dropped."""
    d = np.unique(np.loadtxt(path, comments='#'), axis=0)
    lam, flam = d[:, 0] * wave_scale, d[:, 1]
    m = (lam > 0) & np.isfinite(flam)
    return lam[m], np.clip(flam[m], 0.0, None) * lam[m] ** 2


def _spiff():
    """SPIFF SPHEREx standards (f_lambda, 1.007-4.967 um) extended below 1.007 um with the SUDA template in
    data/spiff/index.csv, scaled to the standard over 1.0-1.15 um: [(name, type, lam [A], f_nu, measured ln um)]."""
    out = []
    with open(os.path.join(_DATA, 'spiff', 'index.csv'), encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            a = np.genfromtxt(os.path.join(_DATA, 'spiff', row['file']), delimiter=',', skip_header=1)
            ok = np.isfinite(a[:, 1]) & (a[:, 1] > 0)
            lam, fnu = a[ok, 0], a[ok, 1] * a[ok, 0] ** 2
            sl, sf = _flam_file(os.path.join(_DATA, 'suda', row['blue_extension'] + '.txt'), wave_scale=1e4)
            ov = lam < 1.15
            k = np.median(fnu[ov] / np.interp(lam[ov] * 1e4, sl, sf))
            blue = np.arange(0.74, lam[0] - 1e-6, 0.005)
            out.append((f"STAR_SPIFF_{row['file'][:-4]}", row['spectral_type'],
                        np.concatenate([blue, lam]) * 1e4, np.concatenate([k * np.interp(blue * 1e4, sl, sf), fnu]),
                        np.log(lam)))
    return out


def _amp_min(sp, plx, plx_err):
    """Smallest amplitude of an IRTF star (f_nu in uJy as observed) that puts the source within D_MAX_KPC: the star
    is at least 1 / (plx + PLX_NSIGMA plx_err) kpc away. 0 for dwarfs without parallax, None (not usable) for other
    stars without parallax."""
    from .startype import lum_class, is_young
    if not (plx > 0 and plx_err > 0):
        return 0.0 if lum_class(sp) in ('dwarf', 'subdwarf') else None
    d_min = 1.0 / (plx + PLX_NSIGMA * plx_err)
    return (d_min / D_MAX_KPC['young' if is_young(sp) else 'old']) ** 2


@functools.lru_cache(maxsize=1)
def _irtf():
    """IRTF Spectral Library stars in the cache (not the planets): [(name, type, lam [A], f_nu [uJy as observed],
    measured ln um, amp_min)]; telluric gaps and missing points dropped; stars whose physical distance cannot be
    checked are left out."""
    from astropy.io import fits
    d = sources.path('irtf', fetch=False)
    with open(os.path.join(_DATA, 'irtf_parallax.csv'), encoding='utf-8') as fh:
        plx = {r['file']: (float(r['plx_mas'] or 'nan'), float(r['plx_err_mas'] or 'nan')) for r in csv.DictReader(fh)}
    out = []
    for fn in sorted(f for f in os.listdir(d) if f.endswith('.fits') and not f.startswith('plnt_')) if d else ():
        with fits.open(os.path.join(d, fn)) as h:
            a, sp = np.asarray(h[0].data, float), ' '.join(str(h[0].header['SPTYPE']).split())
        amin = _amp_min(sp, *plx.get(fn[:-5], (np.nan, np.nan)))
        if amin is None:
            continue
        ok = np.isfinite(a[0]) & np.isfinite(a[1]) & (a[1] > 0)
        lam, flam = a[0][ok], a[1][ok]
        o = np.argsort(lam)
        out.append((f'STAR_IRTF_{fn[:-5]}', sp, lam[o] * 1e4, flam[o] * lam[o] ** 2 * _UJY, np.log(lam[o]), amin))
    return out


def _rj_tail(lam_A, fnu, to_A=52000.0):
    """Extend past the red end as f_nu ~ lambda^-2 from the median of the last 0.1 um."""
    red = lam_A > lam_A[-1] - 1000.0
    ext = np.arange(lam_A[-1] + 100.0, to_A, 100.0)
    return np.concatenate([lam_A, ext]), np.concatenate([fnu, np.median(fnu[red]) * (ext / np.median(lam_A[red])) ** -2.0])


def typing_library():
    """Templates for the spectral type of a star: SPIFF standards and every IRTF spectrum in the cache."""
    return ([Template(n, 'star', lam, fnu, z_free=False, zmax=0.0, spherex_native=True, spectral_type=sp, measured=m)
             for n, sp, lam, fnu, m in _spiff()] +
            [Template(n, 'star', *_rj_tail(lam, fnu), z_free=False, zmax=0.0, spectral_type=sp, measured=m, amp_min=a)
             for n, sp, lam, fnu, m, a in _irtf()])


def build_library():
    """8 QSO, 4 galaxy and 103 star templates (SUDA, SPIFF), plus the BT-Settl and IRTF stars if in the cache. Every
    star is an observed or model stellar spectrum (no black bodies: they have no luminosity to check)."""
    lib = []
    sel, swire, bts = sources.path('selsing'), sources.path('swire'), sources.path('btsettl', fetch=False)
    for ebv in sources.SELSING_EBV:
        lam, fnu = _flam_file(os.path.join(sel, f'selsing16_smc_ebv{round(ebv * 100):03d}.sed'))
        lib.append(Template(f'QSO_selsing_ebv{ebv:.2f}', 'qso', lam, fnu * _igm_break(lam)))
    lam, fnu = _flam_file(os.path.join(sel, 'selsing16_smc_ebv000.sed'))
    dust = _blackbody(lam, 1243.6)   # hot dust, Temple et al. (2021)
    dust *= np.interp(2e4, lam, fnu) / np.interp(2e4, lam, dust)
    for r in (1, 3):
        lib.append(Template(f'QSO_selsing_ebv0.00_dust{r}', 'qso', lam, (fnu + r * dust) * _igm_break(lam)))
    for g in sources.SWIRE_NAMES:
        lam, fnu = _flam_file(os.path.join(swire, f'{g}_template_norm.sed'))
        lib.append(Template(f'GAL_{g}', 'galaxy', lam, fnu, zmax=3.0))
    for fn in sorted(os.listdir(os.path.join(_DATA, 'suda'))):      # SUDA normalized_flux is f_lambda
        lam, fnu = _flam_file(os.path.join(_DATA, 'suda', fn), wave_scale=1e4)
        lib.append(Template(f'STAR_SUDA_{fn[:-4]}', 'star', lam, fnu, z_free=False, zmax=0.0, spherex_native=True))
    for fn in sorted(f for f in os.listdir(bts) if f.endswith('.txt')) if bts else ():
        lam, fnu = _flam_file(os.path.join(bts, fn))
        lib.append(Template(f'STAR_{fn[:-4]}', 'star', lam, fnu, z_free=False, zmax=0.0))
    for t in typing_library():
        if t.spherex_native or np.exp(t.measured[-1]) >= IRTF_RED_MIN_UM:
            lib.append(t)
    return lib


def library_id(lib):
    """Short description of the template set, e.g. 'qso8-gal4-star322(btsettl41,irtf178)'."""
    n = {c: sum(t.cls == c for t in lib) for c in ('qso', 'galaxy', 'star')}
    opt = [f'{k}{m}' for k, m in (('btsettl', sum('btsettl' in t.name for t in lib)),
                                  ('irtf', sum('_IRTF_' in t.name for t in lib))) if m]
    return f"qso{n['qso']}-gal{n['galaxy']}-star{n['star']}" + (f"({','.join(opt)})" if opt else '')


def _blackbody(lam_A, T):
    """B_nu up to a constant."""
    return lam_A ** -3.0 / np.expm1(_HC_K / (lam_A * T))
