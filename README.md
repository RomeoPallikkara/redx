<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/logo/redx_logo_dark.svg">
    <img src="docs/logo/redx_logo.svg" alt="RedX" width="360">
  </picture>
</p>

# RedX

Template-based redshift and classification for SPHEREx spectrophotometry.

RedX takes a single SPHEREx spectrum — an IPAC `.tbl` of forced photometry, or a
binned spectrum as CSV or FITS — fits a library of quasar, galaxy and stellar
templates to it, and returns a class, a redshift with a 68% interval, and a verdict
that says how much to trust the redshift. No machine learning: the redshift comes
from template fitting corroborated by an independent emission-line search.

At SPHEREx's resolving power (R ≈ 40 at 0.75–3.8 µm, R ≈ 120 at 3.8–5.0 µm) a single
emission line can be aliased to several redshifts, so RedX reports the number of
lines supporting the fit, compares the template redshift with the line-only one, and
flags doubtful solutions rather than quietly returning one number.

## Install

```bash
pip install -e .
```

Requires Python ≥ 3.9, numpy, astropy and matplotlib. Tests (synthetic spectra, a few
seconds): `pip install -e .[test]` then `pytest`.

Only the SUDA templates (CC BY 4.0) ship with RedX. On first use it downloads the
Selsing quasar composite and the SWIRE galaxy templates from their providers
(< 1 MB), checks their checksums and builds the templates into `~/.cache/redx` (or
`$REDX_CACHE`). Warm-star models are optional and large (about 540 MB from the SVO
theory server); they raise the fraction of stars recognised as stars from 44% to
54% on our test set:

```bash
python -m redx.sources --btsettl
```

## Input

| format | columns | treatment |
|---|---|---|
| `.tbl` (IPAC, as served by IRSA) | `lambda` [µm], `flux`, `flux_err` [µJy], optional `fit_ql`, `ra`, `dec` | finite points with positive error, 0.75–5.05 µm, 0 < `fit_ql` < 5; inverse-variance mean in 44 log-λ bins |
| `.csv` | `wavelength_angstrom`, `flux_ujy_weighted_mean`, `flux_ujy_weighted_sem`, optional `ignored` | used as binned; rows with `ignored` ≠ 0 dropped |
| `.fits` | as `.csv`, in a table extension | as `.csv`; scalar header keywords go to `spec.meta` |

Fluxes are f_ν in µJy, wavelengths observed-frame. Emission-line rest wavelengths
are vacuum. No cosmology is used.

## Command line

```bash
redx spectrum.tbl                      # classify; opens the diagnostic figure
redx spectrum.tbl --no-show            # print only
redx spectrum.tbl --plot out.png       # write the figure instead
redx spectrum.tbl --show --nbin 100    # finer binning for display (resolves Ha)

redx --dir FOLDER --out results.csv --plots PLOTDIR    # batch *.tbl, *.csv, *.fits
```

A single spectrum shows its figure by default when a display is available, and
stays silent when there isn't one, so the same command works over SSH.

Output for a synthetic quasar (Selsing composite, E(B−V) = 0.1, z = 4.5, S/N 15 per
raw point):

```
class   : qso
z_RedX  : 4.492   (68% [4.472, 4.512])
z_lines : 4.498   (n_lines=6)
signif  : 12.51   redchi2: 0.80   n_bins: 43   SNR: 28.52
verdict : QSO z=4.492 (3 lines, secure)
flags   : ok
```

In batch mode the object id is the file name without extension and without a
`_spherex_spectra` suffix.

## Python

```python
import redx

r = redx.classify("spectrum.tbl")
print(r['class'], r['z_RedX'], r['verdict'])
```

`classify()` returns

| key | meaning |
|---|---|
| `class` | `qso`, `galaxy`, `star` or `inconclusive` |
| `z_RedX`, `z_lo`, `z_hi` | best redshift and its 68% interval |
| `z_lines`, `n_lines`, `n_line_support` | independent emission-line redshift, lines found, lines supporting `z_RedX` |
| `line_signif` | combined significance of the strong lines at `z_RedX` (below) |
| `max_line_snr`, `red_slope` | strongest single residual above the continuum, continuum slope |
| `signif`, `redchi2`, `n_bins`, `median_snr` | fit quality and data quality |
| `template` | winning template name (for a blend, the QSO template) |
| `flags`, `verdict` | why to trust or distrust the result (below) |
| `spec`, `_fit` | the spectrum object and raw fit, for plotting |

A QSO is called `secure` when `line_signif` > 3, at least two lines support `z_RedX`
and `z_lines` agrees with it within 0.05 (1+z). Flags:

| flag | meaning |
|---|---|
| `single_line_alias_risk` | only one line supports `z_RedX` |
| `z_lines_disagree` | the line-only redshift differs from `z_RedX` by more than 0.05 (1+z) |
| `qso_no_line` | QSO from a red-rising continuum and a decisive χ² preference; lines below 3σ |
| `blend(T,qso_frac=f)` | QSO plus a z = 0 neighbour template T in the same pixel; f = QSO share of the model flux |
| `class_ambiguous(vs_C)` | the runner-up class C is within Δχ² = 2 |
| `marginal_snr`, `too_few_bins` | median S/N 2–4; fewer than 6 bins |

### Figures

One function draws every RedX figure, so what the CLI shows is exactly what a
script produces:

```python
from redx.plot import diagnostic

diagnostic(r, out='obj.png')                          # save
diagnostic(r, show=True)                              # window
diagnostic(r, out='obj.png',
           annotations={'z_spec': 4.83},
           title_id='obj1')
diagnostic(r, out='obj.png', info=False, lines=False)  # no text inside the axes
```

The backend is chosen when the figure is drawn, not at import, so the same call
saves headless on a cluster and opens a window on a laptop. Style lives in
`redx/style.py` — change it there and every figure follows.

## Method

Each template is compared with the binned spectrum as SPHEREx and the binning
measure it: averaged over one channel (Gaussian, FWHM λ/R; not applied to the SUDA
templates, which are SPHEREx spectra already) and over the log-λ bin width (the
median spacing of the spectrum in ln λ). For every template and trial redshift
(Δz = 0.002, 0 ≤ z ≤ 7 for quasars, ≤ 3 for galaxies, z = 0 for stars) a single
non-negative amplitude is solved in closed form and χ² recorded; the class is the
one whose template reaches the lowest χ².

When a star or galaxy wins, RedX also fits a quasar at z plus one z = 0 star or
galaxy template (two non-negative amplitudes): a quasar blended with a neighbour in
the same 6″ pixel. If this improves χ² by more than 49 (after the scaling below) and
its lines are detected (next), the object is called a QSO and flagged `blend`.

A quasar needs its emission lines where the template puts them: the residual S/N of
the spectrum above a smooth continuum, at the observed positions of the strong lines
(Lyα, C IV, C III], Mg II, Hβ, [O III], Hα) for the fitted z, combined as a matched
filter Σ w_l r_l / √Σ w_l² with w_l = W_l/λ_l, the rest equivalent width over
wavelength from Vanden Berk et al. (2001), must exceed 3 (`line_signif`). A
red-rising continuum with a decisive χ² preference is a weaker alternative
(`qso_no_line`). Spectra with too little signal are `inconclusive`.

P(z) = exp(−χ²/2s), with s = max(1, χ²_ν of the best fit). Among P(z) peaks above
0.2 of the highest, the redshift supported by the most lines is taken. `z_lo`,
`z_hi` are the 68% highest-density interval of that peak, widened in quadrature by
a systematic 0.0036 (1+z) (≈ 1100 km/s, about the shifts of quasar broad lines).
`signif` is √(Δχ²/s) to the best solution more than 0.05 away in z.

`z_lines` is independent of the templates: the redshift that maximises the same
matched filter, with at least two lines inside SPHEREx.

Performance. On synthetic spectra (Selsing quasars at 0.3 < z < 6.5, S/N 8 and 20
per raw point; SWIRE galaxies at 0.1 < z < 2.5) every object is classified correctly
and the NMAD of Δz/(1+z) is ≤ 0.0025. On public SPHEREx spectra: of the 1675 SUDA
ultracool dwarfs (Tu et al. 2026) 96.9% are called star and 1.8% QSO (a circular test:
the SUDA templates are built from these spectra); all 27 broad-absorption-line quasars
of Trump et al. (2006) we tested (0.58 < z < 3.65) are called QSO, NMAD 0.010, 3 with
|Δz|/(1+z) > 0.15. On a private test set of real SPHEREx spectra
of 1809 spectroscopic quasars, 140 stars and 40 galaxies: 94.5% of the quasars are
called QSO (97–98% at 1 < z < 4, 90–92% at 4 < z < 5, 78% at 5 < z < 6); for those, the
NMAD of Δz/(1+z) is 0.005 and 2.6% have |Δz|/(1+z) > 0.15 (1.0% among the 97% without
`z_lines_disagree`); the 68% interval contains the spectroscopic redshift for 65% of
them. 20 of the stars and 10 of the galaxies are called QSO (98.3% of the QSO calls
are quasars; 98.7% excluding `blend`). The `secure` QSOs, 64.5% of all quasars, are
99.1% pure with 0.7% redshift outliers. Classification is weakest for quasars at
z < 0.5 (49% called QSO) and for star versus galaxy: 44% of the stars are called
star and 73% of the galaxies galaxy, most of the others swapped (54% and 65% with the
optional BT-Settl stars; quasar results unchanged).

## Template library

73 templates, 82 with the optional BT-Settl stars:

- **8** quasar: the Selsing et al. (2016) composite with an f_ν ∝ ν^−0.5 tail beyond
  rest 1.135 µm, SMC-reddened (Pei 1992) at E(B−V) = 0, 0.05, 0.10, 0.20, 0.30, 0.50,
  and the unreddened one plus a 1243.6 K blackbody for hot dust (the qsogen default
  temperature of Temple et al. 2021) at 1× and 3× its f_ν at rest 2 µm; all with a
  coarse Lyman-forest / Lyman-limit step
- **4** SWIRE galaxy templates (Polletta et al. 2007): Ell5, Sc, M82, Arp220
- **57** SUDA empirical SPHEREx ultracool-dwarf templates (Tu et al. 2026),
  T_eff 700–3000 K
- **9** BT-Settl CIFIST model stars (Allard et al. 2012), T_eff 3200–7000 K,
  log g 5.0, from the SVO theory server (optional)
- **4** blackbodies, 3500–12000 K

Emission-line rest wavelengths (vacuum) and equivalent widths follow Vanden Berk et
al. (2001), Table 2.
Sources, download URLs and checksums, build recipes and licences are in
`redx/data/folder_description.txt`.

## Validation

`redx/validate.py` scores RedX against a labelled set — classification confusion
matrix, and redshift accuracy for objects called QSO, split at z = 3.5. The
labelled catalogue is site-specific:

```bash
export REDX_TRUTH=/path/to/validation.fits    # qid, otypeid (1 QSO, 4 galaxy, 5 star), z_spec; optional z_spherex
export REDX_DATA=/path/to/spectra             # spherex_qid_<qid>.tbl or <qid>_spherex_spectra.tbl
python -m redx.validate [NMAX]                # NMAX objects (default 400), written to redx_validation.csv
```

## Layout

```
redx/
  core.py        classify() -- the entry point
  fit.py         template fitting, redshift grid, confidence interval
  lines.py       independent emission-line finder
  lines_data.py  Vanden Berk line list, SPHEREx wavelength range
  templates.py   template library construction
  sources.py     first-use download and build of the Selsing, SWIRE and BT-Settl templates
  reader.py      .tbl, .csv and .fits readers, rebinning
  plot.py        the diagnostic figure
  style.py       figure style, defined once
  cli.py         command line
  validate.py    scoring against a labelled set
  data/          SUDA templates and the provenance of all templates (folder_description.txt)
tests/           pytest suite on synthetic spectra
docs/logo/       logo, light and dark variants (SVG)
```

## Licence

Code: MIT, see `LICENSE`. The SUDA templates are redistributed under CC BY 4.0;
the Selsing, SWIRE and BT-Settl data are not redistributed but downloaded from their
providers, whose terms apply. Cite the papers listed in
`redx/data/folder_description.txt`; if you use RedX in published work, see
`CITATION.cff`.
