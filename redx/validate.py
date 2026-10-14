"""Score RedX against a labelled set of SPHEREx spectra.

Truth table (any astropy-readable format) with columns qid (object id), otypeid (1 = QSO, 4 = galaxy,
5 = star), z_spec, and optionally z_spherex (a visual SPHEREx redshift, copied to the output). Spectra are
looked up as spherex_qid_<qid>.tbl or <qid>_spherex_spectra.tbl. Reports the classification confusion matrix
and, for objects RedX calls QSO, the redshift accuracy overall and split at z = 3.5. Prints running progress
so a partial run is still informative.
"""
import os
import sys
import csv
import numpy as np
from astropy.table import Table
import redx

# Paths are site-specific: set REDX_TRUTH / REDX_DATA (and REDX_VALIDATION_CSV), or pass truth=/data=/out_csv=
# to main().
OUT_CSV = os.environ.get('REDX_VALIDATION_CSV', 'redx_validation.csv')
TRUTH = os.environ.get('REDX_TRUTH', 'validation.fits')
DATA = os.environ.get('REDX_DATA', 'spectra')
CLSMAP = {1: 'qso', 4: 'galaxy', 5: 'star'}


def nmad(x):
    x = np.asarray(x)
    return 1.4826 * np.median(np.abs(x - np.median(x))) if len(x) else np.nan


def main(nmax=400, dz=0.004, seed=0, truth=None, data=None, out_csv=None):
    global DATA, OUT_CSV
    truth = truth or TRUTH
    DATA = data or DATA
    OUT_CSV = out_csv or OUT_CSV
    if not os.path.exists(truth):
        raise SystemExit(
            f'truth catalogue not found: {truth}\n'
            'Set REDX_TRUTH (and REDX_DATA to the folder of .tbl spectra), '
            'or pass truth=/data= to main().')
    t = Table.read(truth)
    rows = [r for r in t]
    np.random.default_rng(seed).shuffle(rows)
    conf = {}
    dz_all, z_true_all = [], []
    n = 0
    d = os.path.dirname(os.path.abspath(OUT_CSV))
    os.makedirs(d, exist_ok=True)
    has_zvis = 'z_spherex' in t.colnames
    fh = open(OUT_CSV, 'w', newline='')
    wr = csv.writer(fh)
    wr.writerow(['qid', 'true_class', 'z_true', 'z_visual', 'pred_class', 'z_RedX', 'z_lines', 'n_line_support',
                 'max_line_snr', 'red_slope', 'signif', 'redchi2', 'median_snr'])
    for r in rows:
        ot = int(r['otypeid'])
        if ot not in CLSMAP:
            continue
        qid = str(r['qid'])
        p = os.path.join(DATA, f'spherex_qid_{qid}.tbl')
        if not os.path.exists(p):
            p = os.path.join(DATA, f'{qid}_spherex_spectra.tbl')
        if not os.path.exists(p):
            continue
        try:
            res = redx.classify(p, dz=dz)
        except Exception:
            continue
        true, pred = CLSMAP[ot], res['class']
        z = r['z_spec']
        zt = float(z) if (z is not np.ma.masked and float(z) > 0) else np.nan
        zse = r['z_spherex'] if has_zvis else np.ma.masked
        zse = float(zse) if zse is not np.ma.masked else ''         # blank: no visual redshift
        conf[(true, pred)] = conf.get((true, pred), 0) + 1
        def _r(v):
            return round(v, 4) if isinstance(v, float) and np.isfinite(v) else v
        wr.writerow([qid, true, round(zt, 4) if np.isfinite(zt) else '', zse, pred, res['z_RedX'], res['z_lines'],
                     res.get('n_line_support'), _r(res['max_line_snr']), _r(res['red_slope']), res['signif'],
                     res['redchi2'], round(res['median_snr'], 2)])
        if true == 'qso' and pred == 'qso' and np.isfinite(zt) and np.isfinite(res['z_RedX']):
            dz_all.append((res['z_RedX'] - zt) / (1 + zt))
            z_true_all.append(zt)
        n += 1
        if n % 100 == 0:
            print(f'  ...{n}', flush=True)
        if n >= nmax:
            break
    fh.close()
    print('wrote', OUT_CSV)

    print(f'\nN tested: {n}')
    print('confusion (true -> pred):')
    for true in ['qso', 'galaxy', 'star']:
        tot = sum(v for (tt, _), v in conf.items() if tt == true)
        if not tot:
            continue
        line = '  '.join(f'{pp}:{conf.get((true, pp), 0)}' for pp in ['qso', 'galaxy', 'star'])
        print(f'  {true:7s} (n={tot:3d}): {line}   acc={100*conf.get((true, true), 0)/tot:.0f}%')
    dz_all = np.array(dz_all)
    z_true_all = np.array(z_true_all)
    if len(dz_all):
        def stats(m, lab):
            d = dz_all[m]
            if not len(d):
                print(f'  {lab}: n=0'); return
            print(f'  {lab}: n={len(d)}  NMAD={nmad(d):.3f}  bias={np.median(d):+.3f}  '
                  f'outliers(|dz|>0.15)={100*np.mean(np.abs(d)>0.15):.0f}%')
        print('QSO-called redshift accuracy:')
        stats(np.ones(len(dz_all), bool), 'all z ')
        stats(z_true_all >= 3.5, 'z>=3.5')
        stats(z_true_all < 3.5, 'z<3.5 ')


if __name__ == '__main__':
    nmax = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    main(nmax=nmax)
