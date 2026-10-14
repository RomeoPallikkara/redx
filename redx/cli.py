"""RedX command line.

    redx SPEC.tbl                        # classify, print the result
    redx SPEC.tbl --show                 # ... and open the diagnostic figure in a window
    redx SPEC.tbl --plot out.png         # ... and write it to a PNG
    redx --dir FOLDER [--out results.csv] [--plots PLOTDIR]     # batch a folder (*.tbl, *.csv, *.fits)

The figure is redx.plot.diagnostic in every case, so what you see from the
terminal is exactly what the plotting scripts produce.
"""
import os
import sys
import glob
import csv
import argparse
import numpy as np
from .core import classify
from .plot import diagnostic

_COLS = ['id', 'class', 'z_RedX', 'z_lo', 'z_hi', 'n_line_support', 'line_signif', 'z_lines', 'n_lines', 'signif',
         'redchi2', 'n_bins', 'median_snr', 'template', 'flags', 'verdict']


def _stem(path):
    """Object id from a file name: no directory, extension or '_spherex_spectra' suffix."""
    stem = os.path.basename(str(path))
    for ext in ('.fits.gz', '.fits', '.fit', '.tbl', '.csv'):
        if stem.lower().endswith(ext):
            stem = stem[:-len(ext)]
            break
    return stem[:-len('_spherex_spectra')] if stem.endswith('_spherex_spectra') else stem


def _f(v, nd):
    return f'{v:.{nd}f}' if isinstance(v, (float, np.floating)) else str(v)


def _row(path, r):
    out = {'id': _stem(path)}
    for k in _COLS[1:]:
        v = r.get(k)
        out[k] = round(v, 4) if isinstance(v, float) and np.isfinite(v) else v
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog='redx', description='SPHEREx redshift + classification (RedX)')
    ap.add_argument('spectrum', nargs='?', help='a single .tbl, or a binned .csv or .fits spectrum')
    ap.add_argument('--dir', help='folder of spectra (*.tbl, *.csv, *.fits) to batch')
    ap.add_argument('--out', default='redx_results.csv', help='batch output CSV')
    ap.add_argument('--plot', help='save a diagnostic figure for the single-spectrum mode')
    ap.add_argument('--plots', help='folder to save one diagnostic figure per source (batch mode)')
    ap.add_argument('--show', dest='show', action='store_true', default=None,
                    help='open the diagnostic figure in a window (default for a single spectrum)')
    ap.add_argument('--no-show', dest='show', action='store_false',
                    help='suppress the window')
    ap.add_argument('--nbin', type=int, default=None,
                    help='rebin the spectrum to this many bins for display only')
    a = ap.parse_args(argv)

    if a.dir:
        files = sorted(f for ext in ('tbl', 'csv', 'fits') for f in glob.glob(os.path.join(a.dir, '*.' + ext)))
        if not files:
            print('no spectra found in', a.dir); return 1
        if a.plots:
            os.makedirs(a.plots, exist_ok=True)
        ok = 0
        failed = []
        with open(a.out, 'w', newline='') as fh:
            wr = csv.DictWriter(fh, fieldnames=_COLS)
            wr.writeheader()
            for i, f in enumerate(files):
                try:
                    r = classify(f)
                    wr.writerow(_row(f, r))
                    ok += 1
                    if a.plots and np.isfinite(r.get('z_RedX', np.nan)):
                        stem = _stem(f)
                        diagnostic(r, out=os.path.join(a.plots, stem + '.png'), title_id=stem)
                except Exception as e:
                    failed.append((os.path.basename(f), e))
                    print('  FAIL', os.path.basename(f), e)
                if (i + 1) % 100 == 0:
                    print(f'  {i + 1}/{len(files)}')
        print(f'wrote {a.out}  ({ok}/{len(files)} classified)')
        if failed:
            print(f'  {len(failed)} spectra failed; first: {failed[0][0]}: {failed[0][1]}')
            return 1
        return 0

    if not a.spectrum:
        ap.print_help(); return 1
    r = classify(a.spectrum)
    print(f"class   : {r['class']}")
    print(f"z_RedX  : {_f(r['z_RedX'], 3)}   (68% [{_f(r['z_lo'], 3)}, {_f(r['z_hi'], 3)}])")
    print(f"z_lines : {_f(r['z_lines'], 3)}   (n_lines={r.get('n_lines')})")
    print(f"signif  : {_f(r['signif'], 2)}   redchi2: {_f(r['redchi2'], 2)}   n_bins: {r['n_bins']}   "
          f"SNR: {_f(r['median_snr'], 2)}")
    print(f"verdict : {r['verdict']}")
    print(f"flags   : {r['flags']}")
    # A single spectrum shows its figure by default, unless we are only writing a
    # PNG, the user said --no-show, or there is no display to draw on.
    show = a.show
    if show is None:
        show = not a.plot and bool(os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'))
    if a.plot or show:
        spec = None
        if a.nbin:
            from .reader import read_spectrum
            spec = read_spectrum(a.spectrum, nbin=a.nbin)
        diagnostic(r, out=a.plot, show=show, spec=spec, title_id=_stem(a.spectrum))
        if a.plot:
            print('figure ->', a.plot)
    return 0


if __name__ == '__main__':
    sys.exit(main())
