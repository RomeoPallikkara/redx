"""RedX tests on synthetic spectra: templates sampled through SPHEREx channels, Gaussian noise, binned as a .tbl.
The first run downloads the Selsing and SWIRE templates (redx.sources).

    pip install -e .[test] && pytest
"""
import numpy as np
import pytest
from astropy.table import Table

import redx
from redx.cli import main
from redx.plot import diagnostic
from redx.reader import Spectrum, _bin_inverse_variance
from redx.templates import Binned, build_library

LIB = {t.name: t for t in build_library()}


def raw(name, z, snr=20.0, seed=1, n=260):
    """Raw points of template `name` at redshift z, each averaged over one SPHEREx channel, with noise."""
    rng = np.random.default_rng(seed)
    lam = np.sort(rng.uniform(0.76, 5.0, n))
    f = Binned(LIB[name], 0.0).flux_obs(lam, z)
    f = 100.0 * f / np.median(f)
    s = np.full(n, np.median(f) / snr)
    return lam, f + rng.normal(0.0, s), s


def spectrum(name, z, **kw):
    return Spectrum(*_bin_inverse_variance(*raw(name, z, **kw)))


def test_library():
    classes = [t.cls for t in LIB.values()]
    n_bts = sum('btsettl' in n for n in LIB)
    assert n_bts in (0, 9)
    assert (classes.count('qso'), classes.count('galaxy'), classes.count('star')) == (8, 4, 61 + n_bts)


@pytest.mark.parametrize('name,z', [('QSO_selsing_ebv0.00', 1.4), ('QSO_selsing_ebv0.10', 3.2),
                                    ('QSO_selsing_ebv0.00_dust1', 4.6)])
def test_qso_redshift(name, z):
    r = redx.classify(spectrum(name, z))
    assert r['class'] == 'qso'
    assert abs(r['z_RedX'] - z) / (1 + z) < 0.01
    assert r['z_lo'] < r['z_RedX'] < r['z_hi']
    assert r['line_signif'] > 3 and 'secure' in r['verdict']
    assert abs(r['z_lines'] - z) / (1 + z) < 0.05


def test_galaxy_redshift():
    r = redx.classify(spectrum('GAL_Sc', 0.8))
    assert r['class'] == 'galaxy' and abs(r['z_RedX'] - 0.8) / 1.8 < 0.01


@pytest.mark.parametrize('name', ['STAR_SUDA_teff1500_logg5.0', 'STAR_SUDA_teff2800_logg5.0',
                                  'STAR_btsettl_cifist_T4200_g5.0', 'STAR_bb8000K'])
def test_star(name):
    if name not in LIB:
        pytest.skip('optional BT-Settl templates not downloaded')
    r = redx.classify(spectrum(name, 0.0))
    assert r['class'] == 'star' and np.isnan(r['z_RedX'])


def test_too_few_bins():
    r = redx.classify(Spectrum(np.linspace(1, 2, 4), np.ones(4), np.ones(4)))
    assert r['class'] == 'inconclusive' and r['flags'] == 'too_few_bins'


def test_readers_agree(tmp_path):
    lam, f, s = raw('QSO_selsing_ebv0.00', 3.2)
    n = len(lam)
    Table({'lambda': lam, 'flux': f, 'flux_err': s, 'fit_ql': np.ones(n), 'flags': np.zeros(n, int),
           'ra': np.full(n, 150.0), 'dec': np.full(n, 2.0)}).write(tmp_path / 'a_spherex_spectra.tbl', format='ipac')
    xb, yb, eb = _bin_inverse_variance(lam, f, s)
    tb = Table({'wavelength_angstrom': xb * 1e4, 'flux_ujy_weighted_mean': yb, 'flux_ujy_weighted_sem': eb,
                'ignored': np.zeros(len(xb), int)})
    tb.write(tmp_path / 'b.csv')
    tb.meta['OBJID'] = 'b'
    tb.write(tmp_path / 'c.fits')
    z = [redx.classify(tmp_path / p)['z_RedX'] for p in ('a_spherex_spectra.tbl', 'b.csv', 'c.fits')]
    assert z[0] == z[1] == z[2]
    assert redx.read_spectrum(tmp_path / 'c.fits').meta['objid'] == 'b'
    assert redx.read_spectrum(tmp_path / 'a_spherex_spectra.tbl').meta['ra'] == 150.0


def test_cli_batch_and_figure(tmp_path):
    for i, (name, z) in enumerate([('QSO_selsing_ebv0.00', 2.5), ('STAR_SUDA_teff2000_logg5.0', 0.0)]):
        lam, f, s = raw(name, z, seed=i)
        Table({"lambda": lam, "flux": f, "flux_err": s}).write(tmp_path / f'obj{i}_spherex_spectra.tbl', format='ipac')
    assert main(['--dir', str(tmp_path), '--out', str(tmp_path / 'res.csv'), '--plots', str(tmp_path / 'plots')]) == 0
    res = Table.read(tmp_path / 'res.csv', format='csv')
    assert list(res['id']) == ['obj0', 'obj1'] and list(res['class']) == ['qso', 'star']
    assert sorted(p.name for p in (tmp_path / 'plots').iterdir()) == ['obj0.png']
    r = redx.classify(tmp_path / 'obj0_spherex_spectra.tbl')
    diagnostic(r, out=tmp_path / 'clean.png', info=False, lines=False)
    assert (tmp_path / 'clean.png').stat().st_size > 0
