"""Command line: redx SPEC [--plot out.png | --show]; redx --dir FOLDER [--recursive] [--out results.csv] [--plots DIR]."""
import os
import re
import sys
import glob
import csv
import argparse
import numpy as np
from .core import classify
from .plot import diagnostic

_COLS = ['id', 'class', 'z_RedX', 'z_lo', 'z_hi', 'n_line_support', 'line_signif', 'z_lines', 'n_lines', 'signif',
         'redchi2', 'n_bins', 'median_snr', 'template', 'flags', 'verdict']


SPIFF_BINNED = 'binned_spectrum.csv'


def _stem(path):
    """File name without directory, extension and '_spherex_spectra'; for a SPIFF binned_spectrum.csv the
    name of its run folder without '_RA<ra>_DEC<dec>'."""
    stem = os.path.basename(str(path))
    if stem == SPIFF_BINNED:
        return re.sub(r'_RA[-+0-9.]+_DEC[-+0-9.]+$', '', os.path.basename(os.path.dirname(os.path.abspath(path))))
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
    ap = argparse.ArgumentParser(prog='redx', description='SPHEREx redshift and classification')
    ap.add_argument('spectrum', nargs='?', help='a single .tbl, or a binned .csv or .fits spectrum')
    ap.add_argument('--dir', help='folder of spectra (*.tbl, *.csv, *.fits) to batch')
    ap.add_argument('--recursive', action='store_true',
                    help='with --dir: also every SPIFF binned_spectrum.csv in its subfolders (other SPIFF files skipped)')
    ap.add_argument('--out', default='redx_results.csv', help='batch output CSV')
    ap.add_argument('--plot', help='save a diagnostic figure for the single-spectrum mode')
    ap.add_argument('--plots', help='folder to save one diagnostic figure per source (batch mode)')
    ap.add_argument('--show', dest='show', action='store_true', default=None,
                    help='open the diagnostic figure in a window (default for a single spectrum)')
    ap.add_argument('--no-show', dest='show', action='store_false', help='no window')
    ap.add_argument('--nbin', type=int, help='bins for the displayed spectrum')
    a = ap.parse_args(argv)

    if a.dir:
        files = sorted(f for ext in ('tbl', 'csv', 'fits') for f in glob.glob(os.path.join(a.dir, '*.' + ext)))
        if a.recursive:
            files += sorted(glob.glob(os.path.join(a.dir, '*', '**', SPIFF_BINNED), recursive=True))
        if not files:
            print('no spectra found in', a.dir)
            return 1
        if a.plots:
            os.makedirs(a.plots, exist_ok=True)
        failed = []
        with open(a.out, 'w', newline='') as fh:
            wr = csv.DictWriter(fh, fieldnames=_COLS)
            wr.writeheader()
            for i, f in enumerate(files):
                try:
                    r = classify(f)
                    wr.writerow(_row(f, r))
                    if a.plots:
                        stem = _stem(f)
                        diagnostic(r, out=os.path.join(a.plots, stem + '.png'), title_id=stem)
                except Exception as e:
                    failed.append((os.path.basename(f), e))
                    print('  FAIL', os.path.basename(f), e)
                if (i + 1) % 100 == 0:
                    print(f'  {i + 1}/{len(files)}')
        print(f'wrote {a.out}  ({len(files) - len(failed)}/{len(files)} classified)')
        if failed:
            print(f'  {len(failed)} spectra failed; first: {failed[0][0]}: {failed[0][1]}')
            return 1
        return 0

    if not a.spectrum:
        ap.print_help()
        return 1
    r = classify(a.spectrum)
    print(f"class   : {r['class']}")
    print(f"z_RedX  : {_f(r['z_RedX'], 3)}   (68% [{_f(r['z_lo'], 3)}, {_f(r['z_hi'], 3)}])")
    print(f"z_lines : {_f(r['z_lines'], 3)}   (n_lines={r.get('n_lines')})")
    print(f"signif  : {_f(r['signif'], 2)}   redchi2: {_f(r['redchi2'], 2)}   n_bins: {r['n_bins']}   "
          f"SNR: {_f(r['median_snr'], 2)}")
    print(f"verdict : {r['verdict']}")
    print(f"flags   : {r['flags']}")
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
