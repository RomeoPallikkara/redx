"""Score RedX against a labelled set: confusion matrix and QSO redshift accuracy.

Truth table: qid, otypeid (1 QSO, 4 galaxy, 5 star), z_spec and optionally z_spherex (visual redshift). Spectra:
<REDX_DATA>/spherex_qid_<qid>.tbl or <qid>_spherex_spectra.tbl.

    REDX_TRUTH=truth.fits REDX_DATA=spectra python -m redx.validate [NMAX]
"""
import os
import sys
import csv
import numpy as np
from astropy.table import Table
import redx

CLSMAP = {1: 'qso', 4: 'galaxy', 5: 'star'}
COLS = ['qid', 'true_class', 'z_true', 'z_visual', 'pred_class', 'z_RedX', 'z_lines', 'n_line_support',
        'max_line_snr', 'red_slope', 'signif', 'redchi2', 'median_snr']


def nmad(x):
    return 1.4826 * np.median(np.abs(x - np.median(x))) if len(x) else np.nan


def _r(v, nd=4):
    return round(v, nd) if isinstance(v, float) and np.isfinite(v) else v


def _spectrum(data, qid):
    for name in (f'spherex_qid_{qid}.tbl', f'{qid}_spherex_spectra.tbl'):
        if os.path.exists(os.path.join(data, name)):
            return os.path.join(data, name)


def main(nmax=400, dz=0.004, seed=0, truth=None, data=None, out_csv=None):
    truth = truth or os.environ.get('REDX_TRUTH', 'validation.fits')
    data = data or os.environ.get('REDX_DATA', 'spectra')
    out_csv = out_csv or os.environ.get('REDX_VALIDATION_CSV', 'redx_validation.csv')
    if not os.path.exists(truth):
        raise SystemExit(f'truth catalogue not found: {truth} (set REDX_TRUTH and REDX_DATA)')
    t = Table.read(truth)
    rows = list(t)
    np.random.default_rng(seed).shuffle(rows)
    conf, dzs, zts, n = {}, [], [], 0
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    with open(out_csv, 'w', newline='') as fh:
        wr = csv.writer(fh)
        wr.writerow(COLS)
        for r in rows:
            path = _spectrum(data, r['qid']) if int(r['otypeid']) in CLSMAP else None
            if path is None:
                continue
            try:
                res = redx.classify(path, dz=dz)
            except Exception:
                continue
            true, pred = CLSMAP[int(r['otypeid'])], res['class']
            zt = float(r['z_spec']) if r['z_spec'] is not np.ma.masked and float(r['z_spec']) > 0 else np.nan
            zv = r['z_spherex'] if 'z_spherex' in t.colnames else np.ma.masked
            conf[true, pred] = conf.get((true, pred), 0) + 1
            wr.writerow([r['qid'], true, _r(zt) if np.isfinite(zt) else '', '' if zv is np.ma.masked else float(zv),
                         pred, res['z_RedX'], res['z_lines'], res['n_line_support'], _r(res['max_line_snr']),
                         _r(res['red_slope']), res['signif'], res['redchi2'], round(res['median_snr'], 2)])
            if true == pred == 'qso' and np.isfinite(zt) and np.isfinite(res['z_RedX']):
                dzs.append((res['z_RedX'] - zt) / (1 + zt))
                zts.append(zt)
            n += 1
            if n % 100 == 0:
                print(f'  ...{n}', flush=True)
            if n >= nmax:
                break
    print('wrote', out_csv)
    print(f'\nN tested: {n}\nconfusion (true -> pred):')
    for true in ('qso', 'galaxy', 'star'):
        tot = sum(v for (tt, _), v in conf.items() if tt == true)
        if tot:
            line = '  '.join(f'{p}:{conf.get((true, p), 0)}' for p in ('qso', 'galaxy', 'star'))
            print(f'  {true:7s} (n={tot:3d}): {line}   acc={100 * conf.get((true, true), 0) / tot:.0f}%')
    dzs, zts = np.array(dzs), np.array(zts)
    if len(dzs):
        print('QSO-called redshift accuracy:')
        for lab, m in (('all z ', zts == zts), ('z>=3.5', zts >= 3.5), ('z<3.5 ', zts < 3.5)):
            d = dzs[m]
            stats = (f'  NMAD={nmad(d):.3f}  bias={np.median(d):+.3f}  '
                     f'outliers(|dz|>0.15)={100 * np.mean(np.abs(d) > 0.15):.0f}%') if len(d) else ''
            print(f'  {lab}: n={len(d)}{stats}')


if __name__ == '__main__':
    main(nmax=int(sys.argv[1]) if len(sys.argv) > 1 else 400)
