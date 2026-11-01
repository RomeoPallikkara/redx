"""Tests on synthetic spectra (templates through SPHEREx channels, Gaussian noise, binned as a .tbl)."""
import os
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
    """Noisy channel-averaged points of template `name` at redshift z."""
    rng = np.random.default_rng(seed)
    lam = np.sort(rng.uniform(0.76, 5.0, n))
    f = Binned(LIB[name], 0.0).flux_obs(lam, z)
    f = 100.0 * f / np.median(f)
    s = np.full(n, np.median(f) / snr)
    return lam, f + rng.normal(0.0, s), s


def spectrum(name, z, **kw):
    return Spectrum(*_bin_inverse_variance(*raw(name, z, **kw)))


def noisy(lam, f, snr, rng):
    """Binned spectrum of f with Gaussian noise of median(f) / snr."""
    s = np.full(len(lam), np.median(f) / snr)
    return Spectrum(*_bin_inverse_variance(lam, f + rng.normal(0, s), s))


def write_tbl(path, lam, f, s, **cols):
    Table({'lambda': lam, 'flux': f, 'flux_err': s, **cols}).write(path, format='ipac')


def write_binned(path, lam, f, s, nbin=44):
    xb, yb, eb = _bin_inverse_variance(lam, f, s, nbin=nbin)
    t = Table({'wavelength_angstrom': xb * 1e4, 'flux_ujy_weighted_mean': yb, 'flux_ujy_weighted_sem': eb,
               'ignored': np.zeros(len(xb), int)})
    t.write(path, format='csv')
    return t


def test_library():
    classes = [t.cls for t in LIB.values()]
    n_opt = sum('btsettl' in n or '_IRTF_' in n for n in LIB)
    assert (classes.count('qso'), classes.count('galaxy'), classes.count('star')) == (8, 4, 103 + n_opt)


@pytest.mark.parametrize('name,z', [('QSO_selsing_ebv0.00', 1.4), ('QSO_selsing_ebv0.10', 3.2),
                                    ('QSO_selsing_ebv0.00_dust1', 4.6)])
def test_qso_redshift(name, z):
    r = redx.classify(spectrum(name, z))
    assert r['class'] == 'qso'
    assert abs(r['z_RedX'] - z) / (1 + z) < 0.01
    assert r['z_lo'] < r['z_RedX'] < r['z_hi']
    assert r['line_signif'] > 3 and r['quality'] == 'secure' and r['reason'] == ''
    assert abs(r['z_lines'] - z) / (1 + z) < 0.05


def test_galaxy_redshift():
    r = redx.classify(spectrum('GAL_Sc', 0.8))
    assert r['class'] == 'galaxy' and abs(r['z_RedX'] - 0.8) / 1.8 < 0.01 and r['quality'] == 'probable'


@pytest.mark.parametrize('name', ['STAR_SUDA_teff1500_logg5.0', 'STAR_SUDA_teff2800_logg5.0',
                                  'STAR_btsettl_cifist_T4200_g5.0', 'STAR_SPIFF_L5'])
def test_star(name):
    if name not in LIB:
        pytest.skip('optional BT-Settl templates not downloaded')
    r = redx.classify(spectrum(name, 0.0))
    assert r['class'] == 'star' and np.isnan(r['z_RedX'])


def test_too_few_bins():
    r = redx.classify(Spectrum(np.linspace(1, 2, 4), np.ones(4), np.ones(4)))
    assert r['class'] == 'inconclusive' and r['reason'] == 'no_data'


def test_readers_agree(tmp_path):
    lam, f, s = raw('QSO_selsing_ebv0.00', 3.2)
    n = len(lam)
    write_tbl(tmp_path / 'a_spherex_spectra.tbl', lam, f, s, fit_ql=np.ones(n), flags=np.zeros(n, int),
              ra=np.full(n, 150.0), dec=np.full(n, 2.0))
    tb = write_binned(tmp_path / 'b.csv', lam, f, s)
    tb.meta['OBJID'] = 'b'
    tb.write(tmp_path / 'c.fits')
    z = [redx.classify(tmp_path / p)['z_RedX'] for p in ('a_spherex_spectra.tbl', 'b.csv', 'c.fits')]
    assert z[1] == z[2] and abs(z[0] - z[1]) / (1 + z[1]) < 0.01      # .tbl: fitted to the exposures
    assert redx.read_spectrum(tmp_path / 'c.fits').meta['objid'] == 'b'
    assert redx.read_spectrum(tmp_path / 'a_spherex_spectra.tbl').meta['ra'] == 150.0


def test_cli_batch_and_figure(tmp_path):
    for i, (name, z) in enumerate([('QSO_selsing_ebv0.00', 2.5), ('STAR_SUDA_teff2000_logg5.0', 0.0)]):
        write_tbl(tmp_path / f'obj{i}_spherex_spectra.tbl', *raw(name, z, seed=i))
    assert main(['--dir', str(tmp_path), '--out', str(tmp_path / 'res.csv'), '--plots', str(tmp_path / 'plots')]) == 0
    res = Table.read(tmp_path / 'res.csv', format='csv')
    assert list(res['id']) == ['obj0', 'obj1'] and list(res['class']) == ['qso', 'star']
    assert sorted(p.name for p in (tmp_path / 'plots').iterdir()) == ['obj0.png', 'obj1.png']
    r = redx.classify(tmp_path / 'obj0_spherex_spectra.tbl')
    diagnostic(r, out=tmp_path / 'clean.png', info=False, lines=False)
    assert (tmp_path / 'clean.png').stat().st_size > 0


def test_cli_recursive_spiff(tmp_path):
    run = tmp_path / 'runs' / 'obj+1_RA10.000000_DEC-5.000000'
    (run / 'figs' / 'exp1').mkdir(parents=True)
    write_binned(run / 'binned_spectrum.csv', *raw('QSO_selsing_ebv0.00', 2.5), nbin=100)
    for aux in (run / 'results.csv', run / 'figs' / 'exp1' / 'result.csv'):
        aux.write_text('a,b\n1,2\n')
    out = tmp_path / 'res.csv'
    assert main(['--dir', str(tmp_path / 'runs'), '--out', str(out)]) == 1
    assert main(['--dir', str(tmp_path / 'runs'), '--recursive', '--out', str(out)]) == 0
    res = Table.read(out, format='csv')
    assert list(res['id']) == ['obj+1'] and list(res['class']) == ['qso']


def _spiff_run(path, lam, f, s, forced, f_exp=None):
    """SPIFF run folder: binned_spectrum.csv of f and results.csv of the exposures f_exp (default f)."""
    path.mkdir(parents=True)
    write_binned(path / 'binned_spectrum.csv', lam, f, s, nbin=100)
    f = f if f_exp is None else f_exp
    n = len(lam)
    status = np.where(np.arange(n) == 0, 'failed', 'ok')
    Table({'fits_analysis_status': status, 'n_pix_used_in_fit': np.full(n, 9), 'psf_scipy_wv_um': lam,
           'psf_scipy_flux_uJy': f, 'psf_scipy_flux_uJy_err': s, 'psf_scipy_snr': f / s,
           'psf_scipy_method_used': np.full(n, 'forced-linear' if forced else 'scipy')}).write(path / 'results.csv')
    return path / 'binned_spectrum.csv'


@pytest.mark.parametrize('forced', [False, True])
def test_spiff_exposures(tmp_path, forced):
    from redx.reader import SYS_FRAC
    lam, f, s = raw('QSO_selsing_ebv0.00', 3.2)
    fe = f.copy()
    fe[1], fe[130] = -f[1], 1e4                # an S/N < 0.1 exposure and an outlier
    r = redx.classify(_spiff_run(tmp_path / 'run', lam, f, s, forced, fe))
    ex = r['spec'].meta['exposures']
    assert ex.meta['spiff_forced'] == forced and r['n_clip'] == 1 == ex.meta['n_clip']
    assert r['n_exp'] == ex.n == len(lam) - 2 - (not forced)        # status not ok, clipped, S/N cut unless forced
    assert np.allclose(ex.sigma, np.hypot(np.interp(ex.lam_um, lam, s), SYS_FRAC * ex.flux))
    assert r['class'] == 'qso' and abs(r['z_RedX'] - 3.2) / 4.2 < 0.01


def test_exposures_fallback(tmp_path):
    lam, f, s = raw('QSO_selsing_ebv0.00', 3.2)
    b = _spiff_run(tmp_path / 'run', lam, f, s, True)
    (tmp_path / 'run' / 'results.csv').unlink()
    r = redx.classify(b)
    assert r['spec'].meta['exposures'] is None and r['n_exp'] == 0 and r['class'] == 'qso'
    n = len(lam)
    flags = np.where(np.arange(n) < 5, 1 << 33, 1 << 32)               # FIT_ERROR dropped, CONTAINS_BAD_PIXEL kept
    write_tbl(tmp_path / 'a.tbl', lam, f, s, flags=flags)
    assert redx.classify(tmp_path / 'a.tbl')['n_exp'] == n - 5
    write_tbl(tmp_path / 'few.tbl', lam[:15], f[:15], s[:15])
    assert redx.classify(tmp_path / 'few.tbl')['n_exp'] == 0          # fewer than MIN_EXPOSURES: binned fit


def test_figure_and_summary_show_the_result():
    r = redx.classify(spectrum('QSO_selsing_ebv0.00', 2.5, snr=30))
    box = ' '.join(t.get_text() for t in diagnostic(r).axes[0].texts)
    assert f"quality  = {r['quality']}" in box and 'reason' not in box
    assert any(line.startswith('quality') for line in redx.summary(r).splitlines())


@pytest.mark.parametrize('name,z,snr', [('QSO_selsing_ebv0.00', 2.5, 30), ('QSO_selsing_ebv0.00', 2.5, 0.8),
                                        ('GAL_Sc', 0.8, 30), ('STAR_SPIFF_L5', 0.0, 30), ('STAR_SPIFF_L5', 0.0, 0.8)])
def test_every_result_is_explained(name, z, snr):
    r = redx.classify(spectrum(name, z, snr=snr))
    warnings = [w for w in r['warnings'].split(';') if w]
    if r['class'] == 'inconclusive':
        assert r['reason'] in redx.REASONS and r['quality'] == '' and not warnings and np.isnan(r['z_RedX'])
    else:
        assert r['reason'] == '' and r['quality'] in redx.QUALITY and all(w in redx.WARNINGS for w in warnings)
        assert (r['quality'] == 'secure') == (not warnings and r['class'] != 'galaxy')


def test_import_btsettl(tmp_path, monkeypatch):
    from redx import sources
    src = tmp_path / 'src'
    src.mkdir()
    lam = np.geomspace(3500.0, 250000.0, 50)
    for fn in ('btsettl_T2900_g5.0_eazy.txt', 'btsettl_T3000_g5.0_eazy.txt', 'btsettl_T3100_g4.5_eazy.txt',
               'btsettl_T3100_g4.0_eazy.txt', 'other.txt'):
        np.savetxt(src / fn, np.c_[lam, np.ones(50)])
    monkeypatch.setattr(sources, 'CACHE', str(tmp_path / 'cache'))
    assert sources.import_btsettl(str(src)) == 2
    d = sources.path('btsettl', fetch=False)
    assert sorted(f for f in os.listdir(d) if f.endswith('.txt')) == ['btsettl_T3000_g5.0_eazy.txt',
                                                                      'btsettl_T3100_g4.5_eazy.txt']
    with pytest.raises(RuntimeError):
        sources.import_btsettl(str(tmp_path))


def test_library_id():
    from redx.templates import library_id
    assert library_id(list(LIB.values())).startswith('qso8-gal4-star')
    assert redx.classify(spectrum('QSO_selsing_ebv0.00', 2.5))['library'].startswith('qso8-gal4-star')


def test_figure_without_redshift(tmp_path):
    noisy = redx.classify(spectrum('QSO_selsing_ebv0.00', 2.5, snr=0.5))
    few = redx.classify(Spectrum(np.linspace(1, 4, 4), np.ones(4), np.ones(4)))
    assert noisy['class'] == 'inconclusive' and few['reason'] == 'no_data'
    for i, r in enumerate((noisy, few)):
        diagnostic(r, out=tmp_path / f'f{i}.png')
        assert (tmp_path / f'f{i}.png').stat().st_size > 0


def test_cli_same_stem_and_plot_failure(tmp_path, monkeypatch):
    lam, f, s = raw('QSO_selsing_ebv0.00', 2.5)
    (tmp_path / 'in').mkdir()
    write_tbl(tmp_path / 'in' / 'a.tbl', lam, f, s)
    write_binned(tmp_path / 'in' / 'a.csv', lam, f, s)
    args = ['--dir', str(tmp_path / 'in'), '--out', str(tmp_path / 'res.csv'), '--plots', str(tmp_path / 'plots')]
    assert main(args) == 0
    assert sorted(p.name for p in (tmp_path / 'plots').iterdir()) == ['a.png', 'a_2.png']

    def boom(*a, **k):
        raise RuntimeError('plot')
    monkeypatch.setattr('redx.cli.diagnostic', boom)
    assert main(args) == 1
    assert len(Table.read(tmp_path / 'res.csv', format='csv')) == 2


def test_cli_error_row_and_output_in_input_folder(tmp_path):
    write_tbl(tmp_path / 'a_spherex_spectra.tbl', *raw('QSO_selsing_ebv0.00', 2.5))
    Table({'x': [1, 2]}).write(tmp_path / 'bad.csv', format='csv')
    out = tmp_path / 'redx_results.csv'
    for _ in range(2):
        assert main(['--dir', str(tmp_path), '--out', str(out)]) == 1
        res = Table.read(out, format='csv')
        assert list(res['id']) == ['a', 'bad'] and list(res['class']) == ['qso', 'error']
    assert 'quality' in res.colnames and str(res['reason'][1]).startswith('error:')


@pytest.fixture
def warm_star(monkeypatch):
    """A 5000 K black-body star added to the library (test only: RedX has no black-body stars)."""
    import redx.core as core
    from redx.templates import Template, _blackbody
    lam = np.geomspace(3000.0, 60000.0, 1500)
    star = Template('STAR_test_bb5000K', 'star', lam, _blackbody(lam, 5000.0), z_free=False, zmax=0.0)
    monkeypatch.setattr(core, '_LIBRARY', list(core._library()) + [star])
    return star


def _qso_star_mixture(a, star):
    qso, st = Binned(LIB['QSO_selsing_ebv0.10'], 0.0), Binned(star, 0.0)
    rng = np.random.default_rng(3)
    lam = np.sort(rng.uniform(0.76, 5.0, 260))
    fq, fs = qso.flux_obs(lam, 2.0), st.flux_obs(lam, 0.0)
    f = a * fq / np.median(fq) + (1 - a) * fs / np.median(fs)
    return redx.classify(noisy(lam, f * 100 / np.median(f), 25, rng))


def test_qso_star_mixture_is_not_fitted_as_a_sum(warm_star):
    r = _qso_star_mixture(0.6, warm_star)
    assert 'blend' not in r['_fit'] and r['class'] == 'inconclusive' and r['reason'] == 'poor_fit'
    box = ' '.join(t.get_text() for t in diagnostic(r).axes[0].texts)
    assert 'poor_fit' in box and 'quality' not in box


@pytest.mark.parametrize('name,true', [('STAR_SPIFF_T5', 85.0), ('STAR_SPIFF_L2', 72.0), ('STAR_SPIFF_M9', 69.0)])
def test_star_type(name, true):
    from redx.startype import type_code
    r = redx.classify(spectrum(name, 0.0, snr=30))
    lo, _, hi = r['star_type_range'].partition('-')
    assert r['class'] == 'star' and r['star_lum'] in ('dwarf', 'ambiguous')
    assert abs(type_code(r['star_type']) - true) <= 1 and type_code(lo) <= true <= type_code(hi or lo)


def test_star_type_codes():
    from redx.startype import type_code, lum_class, code_label
    assert [type_code(s) for s in ('M2.5 IIIab', 'L0 γ', 'sdT4', 'G8 IV')] == [62.5, 70.0, 84.0, 48.0]
    assert [lum_class(s) for s in ('M2.5 IIIab', 'L0 γ', 'd/sdL8', 'G8 IV', 'F0 Ia', 'C-N4.5C24.5')] == \
        ['giant', 'dwarf', 'subdwarf', 'dwarf', 'giant', '']
    assert code_label(62.5) == 'M2.5' and np.isnan(type_code('C-N4.5C24.5'))


def test_galaxy_with_qso_lines_is_inconclusive(monkeypatch):
    import redx.core as core
    real = core.fit_spectrum

    def fake(spec, lib, dz=0.002):
        f = real(spec, lib, dz=dz)
        f['class'] = 'galaxy'
        f['per_class']['galaxy']['redchi2'] = 1.2 * f['per_class']['qso']['redchi2']
        return f

    monkeypatch.setattr(core, 'fit_spectrum', fake)
    r = redx.classify(spectrum('QSO_selsing_ebv0.00', 3.5))
    assert r['class'] == 'inconclusive' and r['reason'] == 'galaxy_or_qso' and np.isnan(r['z_RedX'])


def test_import_irtf(tmp_path, monkeypatch):
    from astropy.io import fits
    from redx import sources, templates
    src = tmp_path / 'src'
    src.mkdir()
    lam = np.linspace(0.8, 5.0, 400)
    for fn, sp in (('M2V_HD1.fits', 'M2V'), ('M2V_HD1_ext.fits', 'M2V'), ('K0III_HD2.fits', None)):
        h = fits.PrimaryHDU(np.array([lam, lam ** -2.0, 0.01 * lam ** -2.0]))
        if sp:
            h.header['SPTYPE'] = sp
        h.writeto(src / fn)
    (src / 'bad.fits').write_text('<html>404</html>')
    monkeypatch.setattr(sources, 'CACHE', str(tmp_path / 'cache'))
    templates._irtf.cache_clear()
    try:
        assert sources.import_irtf(str(src)) == 1
        assert [n for n, sp, *_ in templates._irtf()] == ['STAR_IRTF_M2V_HD1']
        assert sum('_IRTF_' in t.name for t in build_library()) == 1
    finally:
        templates._irtf.cache_clear()


def test_star_template_is_physical():
    from redx.templates import typing_library, D_MAX_KPC
    irtf = [t for t in typing_library() if '_IRTF_' in t.name]
    if not irtf:
        pytest.skip('optional IRTF templates not in the cache')
    sg = next(t for t in irtf if 'G7Ia_HD333385' in t.name)
    assert sg.amp_min > 0 and not any(t.name.startswith('STAR_IRTF_plnt') for t in irtf)
    rng = np.random.default_rng(5)
    lam = np.sort(rng.uniform(0.76, 5.0, 260))
    f0 = Binned(sg, 0.0).flux_obs(lam, 0.0)
    for scale, allowed in ((10 * sg.amp_min, True), (1e-6 * sg.amp_min, False)):     # near and 1000x too far
        r = redx.classify(noisy(lam, scale * f0, 30, rng))
        assert (r['template'] == sg.name) == allowed and r['class'] != 'qso'      # far: no physical star fits
        assert r['class'] == 'star' if allowed else r['redchi2'] > 10
    assert D_MAX_KPC['young'] < D_MAX_KPC['old']


def test_chi2_amp_bound():
    from redx.fit import chi2_amp
    d, m, w = np.array([1.0, 2.0]), np.array([1.0, 2.0]), np.ones(2)
    c, a = chi2_amp(d, m, w)
    assert np.isclose(a, 1.0) and np.isclose(c, 0.0)
    c, a = chi2_amp(d, m, w, amp_min=3.0)
    assert np.isclose(a, 3.0) and np.isclose(c, 4.0 + 16.0)
    c, a = chi2_amp(-d, m, w)
    assert a == 0.0 and np.isclose(c, 5.0)


def test_weak_single_line_is_inconclusive(monkeypatch):
    import redx.core as core
    monkeypatch.setattr(core, 'line_support', lambda *a, **k: 1)
    monkeypatch.setattr(core, 'line_signif', lambda *a: 5.0)
    r = redx.classify(spectrum('QSO_selsing_ebv0.00', 2.4, snr=30.0))
    assert r['class'] == 'inconclusive' and r['reason'] == 'weak_line' and r['quality'] == ''
    monkeypatch.setattr(core, 'line_signif', lambda *a: 7.0)
    r = redx.classify(spectrum('QSO_selsing_ebv0.00', 2.4, snr=30.0))
    assert r['class'] == 'qso' and 'single_line' in r['warnings'] and r['quality'] != 'secure'


def test_poor_fit_is_inconclusive():
    rng = np.random.default_rng(5)
    lam = np.sort(rng.uniform(0.76, 5.0, 260))
    r = redx.classify(noisy(lam, np.where(lam < 2.0, 100.0, 300.0), 30, rng))      # a step: no template fits it
    assert r['class'] == 'inconclusive' and r['reason'] == 'poor_fit' and r['chi2_sys'] > 5


def test_absent_lines():
    from redx.lines import line_scale, _continuum
    from redx.plot import _model
    r = redx.classify(spectrum('QSO_selsing_ebv0.00', 2.4, snr=50.0))
    assert r['class'] == 'qso' and 0.7 < r['line_scale'] < 1.3 and r['chi2_sys'] < 2
    s = r['spec']
    m = _model(r, s, s.lam_um)
    flat = Spectrum(s.lam_um, _continuum(s.lam_um, m), s.sigma)      # the model without its lines
    a, deficit = line_scale(flat, m, r['z_RedX'])
    assert a < 0.3 and deficit > 5


def test_library_is_physical():
    from redx.templates import _irtf
    names = [t.name for t in LIB.values()]
    assert not any('_bb' in n or 'plnt' in n for n in names)      # stars are stellar spectra only
    assert all(a is not None and a >= 0 for *_, a in _irtf())


def test_low_snr_reports_only_qso(monkeypatch):
    monkeypatch.setattr(redx.core, '_low_quality', lambda spec, fit: True)
    q = redx.classify(spectrum('QSO_selsing_ebv0.00', 2.5))
    s = redx.classify(spectrum('STAR_SPIFF_L5', 0.0))
    g = redx.classify(spectrum('GAL_Sc', 0.8))
    assert q['class'] == 'qso' and 'low_snr' in q['warnings'] and q['quality'] == 'low'
    assert s['class'] == g['class'] == 'inconclusive' and np.isnan(g['z_RedX']) and s['star_type'] == '' and s['reason'] == g['reason'] == 'too_noisy'
