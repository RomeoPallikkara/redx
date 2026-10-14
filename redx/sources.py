"""Templates RedX may not redistribute: downloaded from their providers on first use, built as described in
data/folder_description.txt and cached in REDX_CACHE (default ~/.cache/redx).

  selsing : Selsing et al. (2016) composite, GitHub jselsing/QuasarComposite (automatic)
  swire   : SWIRE library, Polletta et al. (2007) (automatic)
  btsettl : BT-Settl CIFIST stars, Allard et al. (2012), SVO theory server, ~540 MB (optional)

    python -m redx.sources [--btsettl]      # download and build now
"""
import gzip
import hashlib
import io
import os
import sys
import tarfile
import urllib.request
import numpy as np

CACHE = os.environ.get('REDX_CACHE', os.path.join(os.path.expanduser('~'), '.cache', 'redx'))
SELSING = ('https://raw.githubusercontent.com/jselsing/QuasarComposite/master/Selsing2015.dat',
           'dce055acdd378444e35acad292ccf2fb00c2dc381216e5cc1d81083753f11c12')
SWIRE = ('http://www.iasf-milano.inaf.it/~polletta/templates/data/swire_library.tar.gz',
         '34cce03a16fde1b3f0da0a39b84ac38ba5c1d526697200d72beed0fed3f9b48c')
SWIRE_NAMES = ('Ell5', 'Sc', 'M82', 'Arp220')
BTSETTL = 'http://svo2.cab.inta-csic.es/theory/newov2/ssap.php?model=bt-settl-cifist&fid={}'
BTSETTL_FID = {3200: 179, 3500: 200, 3800: 221, 4200: 249, 4600: 277, 5000: 305, 5500: 340, 6000: 375, 7000: 445}
SELSING_EBV = (0.0, 0.05, 0.10, 0.20, 0.30, 0.50)
# Pei (1992) Table 4, SMC: (lambda_i [um], a_i, b_i, n_i); R_V = 2.93 (Table 2)
_PEI_SMC = ((0.042, 185.0, 90.0, 2.0), (0.08, 27.0, 5.5, 4.0), (0.22, 0.005, -1.95, 2.0),
            (9.7, 0.010, -1.95, 2.0), (18.0, 0.012, -1.80, 2.0), (25.0, 0.030, 0.0, 2.0))


def _get(url, sha256=None):
    print(f'redx: downloading {url}', file=sys.stderr)
    with urllib.request.urlopen(urllib.request.Request(url, headers={'Accept-Encoding': 'gzip'}), timeout=600) as r:
        data = r.read()
        if r.headers.get('Content-Encoding') == 'gzip':
            data = gzip.decompress(data)
    if sha256 and hashlib.sha256(data).hexdigest() != sha256:
        raise RuntimeError(f'{url}: checksum differs from the one RedX was validated with ({sha256})')
    return data


def _selsing(out):
    """Selsing composite + f_lambda ~ lambda^-1.5 tail (matched to the mean over 11150-11350 A) on 2200 log points,
    1000-250000 A, SMC-reddened (Pei 1992) at each E(B-V)."""
    d = np.loadtxt(io.BytesIO(_get(*SELSING)), comments='#')
    lam, f = d[:, 0], d[:, 1]
    ok = np.isfinite(f) & (f > 0)
    lam, f = lam[ok], f[ok]
    anchor = (lam >= 11150.0) & (lam <= 11350.0)
    grid = np.geomspace(1000.0, 250000.0, 2200)
    flux = np.where(grid <= lam.max(), np.interp(grid, lam, f), f[anchor].mean() * (grid / 11250.0) ** -1.5)
    x = grid / 1e4
    xi = sum(a / ((x / li) ** n + (li / x) ** n + b) for li, a, b, n in _PEI_SMC)
    for ebv in SELSING_EBV:
        red = flux * 10.0 ** (-0.4 * xi * (2.93 + 1.0) * ebv) if ebv > 0 else flux
        np.savetxt(os.path.join(out, f'selsing16_smc_ebv{round(ebv * 100):03d}.sed'), np.c_[grid, red],
                   fmt=('%12.3f', '%14.6e'), comments='# ',
                   header=f'Selsing+2016 composite + IR power-law tail (f_lambda ~ lambda^-1.5), '
                          f'SMC reddening (Pei 1992), E(B-V) = {ebv:.2f}')


def _swire(out):
    with tarfile.open(fileobj=io.BytesIO(_get(*SWIRE))) as tar:
        for m in tar.getmembers():
            name = os.path.basename(m.name)
            if name in {f'{g}_template_norm.sed' for g in SWIRE_NAMES}:
                with open(os.path.join(out, name), 'wb') as fh:
                    fh.write(tar.extractfile(m).read())


def _btsettl(out, R=500.0):
    """BT-Settl spectra smoothed with a Gaussian of FWHM lambda/R, 3 points per FWHM, 0.5-6 um."""
    from astropy.io.votable import parse_single_table
    step = 1.0 / R / 3.0
    grid = np.exp(np.arange(np.log(5000.0), np.log(60000.0), step))
    fine = np.exp(np.arange(np.log(4750.0), np.log(63000.0), step / 5))
    sig = 5.0 / (2.3548 * R * step)
    k = np.exp(-0.5 * (np.arange(-int(5 * sig), int(5 * sig) + 1) / sig) ** 2)
    for T, fid in BTSETTL_FID.items():
        t = parse_single_table(io.BytesIO(_get(BTSETTL.format(fid)))).to_table()
        lam, flam = np.asarray(t.columns[0], float), np.asarray(t.columns[1], float)
        m = (lam > 0.9 * fine[0]) & (lam < 1.1 * fine[-1])
        o = np.argsort(lam[m])
        f = np.convolve(np.interp(np.log(fine), np.log(lam[m][o]), flam[m][o]), k / k.sum(), mode='same')
        np.savetxt(os.path.join(out, f'btsettl_cifist_T{T}_g5.0.txt'),
                   np.c_[grid, np.interp(np.log(grid), np.log(fine), f)], fmt=('%.2f', '%.6e'),
                   header=f'BT-Settl CIFIST2011 (SVO fid {fid}); FWHM lambda/{R:.0f}\n'
                          'wavelength [A]  f_lambda [erg/s/cm2/A]')


_BUILD = {'selsing': (_selsing, len(SELSING_EBV)), 'swire': (_swire, len(SWIRE_NAMES)),
          'btsettl': (_btsettl, len(BTSETTL_FID))}


def path(name, fetch=True):
    """Directory holding the built `name` templates; built on first use (fetch=False: None if absent)."""
    d = os.path.join(CACHE, name)
    build, n = _BUILD[name]
    if not (os.path.isdir(d) and len(os.listdir(d)) == n):
        if not fetch:
            return None
        os.makedirs(d, exist_ok=True)
        try:
            build(d)
        except Exception as e:
            for f in os.listdir(d):
                os.remove(os.path.join(d, f))
            raise RuntimeError(f'redx: could not build the {name} templates ({e}). They are downloaded once from '
                               f'their providers; check the connection, or set REDX_CACHE to a directory '
                               f'that already holds them.') from e
    return d


if __name__ == '__main__':
    for name in ('selsing', 'swire') + (('btsettl',) if '--btsettl' in sys.argv else ()):
        print(name, '->', path(name))
