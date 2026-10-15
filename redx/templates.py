"""Template library (f_nu on rest wavelength [A]): QSO (Selsing, z 0-7), galaxy (SWIRE, z 0-3), star (SUDA,
BT-Settl if downloaded, black bodies; z = 0). See README and data/folder_description.txt."""
import os
import numpy as np
from . import sources

_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
_HC_K = 1.438776877e8            # h c / k [A K]
SPHEREX_R = (40.0, 120.0)        # resolving power below / above 3.82 um
SPHEREX_R_SPLIT_UM = 3.82
_LOG_STEP = 2e-4                 # dex


class Template:
    def __init__(self, name, cls, lam_rest_A, fnu, z_free=True, zmax=7.0, spherex_native=False):
        o = np.argsort(lam_rest_A)
        self.name, self.cls, self.z_free, self.zmax = name, cls, z_free, zmax
        self.spherex_native = spherex_native
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


class Binned:
    """Template averaged over a SPHEREx channel (not for SPHEREx-native templates) and a log-lambda bin of width
    dln. Smoothing in ln(lambda) commutes with redshift, so it is done once."""

    def __init__(self, tmpl, dln):
        self.name, self.cls, self.z_free, self.zmax = tmpl.name, tmpl.cls, tmpl.z_free, tmpl.zmax
        self._lx = np.arange(tmpl._lx[0], tmpl._lx[-1] + _LOG_STEP / 2, _LOG_STEP)
        fy = np.interp(self._lx, tmpl._lx, tmpl._fy)
        self._fy = [_smooth(fy, dln, None if tmpl.spherex_native else R) for R in SPHEREX_R]

    def flux_obs(self, lam_obs_um, z):
        lam = np.asarray(lam_obs_um, float)
        x = np.log10(lam * 1e4 / (1.0 + z))
        blue, red = (np.interp(x, self._lx, f, left=f[0], right=f[-1]) for f in self._fy)
        return np.where(lam < SPHEREX_R_SPLIT_UM, blue, red)


def template_origin(name):
    """Short source of a template, for figure labels."""
    for key, origin in (('dust', 'Selsing+2016 QSO composite + hot dust'),
                        ('selsing', 'Selsing+2016 QSO composite + SMC'), ('GAL_', 'SWIRE galaxy (Polletta+2007)'),
                        ('SUDA', 'SUDA SPHEREx dwarf atlas (Tu+2026)'),
                        ('btsettl', 'BT-Settl model star (Allard+2012)'), ('STAR_bb', 'black body')):
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


def build_library():
    """8 QSO, 4 galaxy and 61 star templates, plus 9 BT-Settl stars if downloaded."""
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
    for fn in sorted(os.listdir(bts)) if bts else ():
        lam, fnu = _flam_file(os.path.join(bts, fn))
        lib.append(Template(f'STAR_{fn[:-4]}', 'star', lam, fnu, z_free=False, zmax=0.0))
    lam = np.geomspace(3000.0, 60000.0, 1500)
    for T in (3500, 5000, 8000, 12000):
        lib.append(Template(f'STAR_bb{T}K', 'star', lam, _blackbody(lam, T), z_free=False, zmax=0.0))
    return lib


def _blackbody(lam_A, T):
    """B_nu up to a constant."""
    return lam_A ** -3.0 / np.expm1(_HC_K / (lam_A * T))
