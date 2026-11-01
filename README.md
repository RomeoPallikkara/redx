<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/logo/redx_logo_dark.svg">
    <img src="docs/logo/redx_logo.svg" alt="RedX" width="360">
  </picture>
</p>

# RedX

Redshift and classification (quasar, galaxy, star) of SPHEREx spectra by template
fitting, cross-checked with an independent emission-line search.

**Development version (0.1.0.dev0): interfaces and results may change.**

## Install

```bash
pip install git+https://github.com/RomeoPallikkara/redx
```

Python ≥ 3.9 with numpy, astropy and matplotlib. On first use RedX downloads the
Selsing and SWIRE templates from their providers (< 1 MB) into `~/.cache/redx`
(`$REDX_CACHE`). Optional warm-star models (BT-Settl): download about 540 MB with
`python -m redx.sources --btsettl`, or import local `btsettl_T<Teff>_g<logg>*.txt` files (wavelength [A],
f_lambda; Teff >= 3000 K, log g 4.5 and 5.0 are kept) with `python -m redx.sources --btsettl-from DIR`.
Optional IRTF Spectral Library stars (F-M dwarfs and giants, used for classification and star typing):
`python -m redx.sources --irtf` (about 230 FITS files from the IRTF server), or import the FITS files of a
local copy with `python -m redx.sources --irtf-from DIR`.
The `library` column of the results names the template set used.

## Use

```bash
redx --dir RUNS --recursive           # SPIFF runs: RUNS/<id>_RA*_DEC*/binned_spectrum.csv, id = <id>
redx RUN/binned_spectrum.csv          # one SPIFF run; --plot out.png writes the figure
redx spectrum.tbl                     # an IRSA .tbl
redx --dir FOLDER --out results.csv   # every *.tbl, *.csv, *.fits in FOLDER
```

```python
import redx
r = redx.classify('spectrum.tbl')
print(r['class'], r['z_RedX'], r['quality'], r['reason'], r['warnings'])
print(redx.summary(r))      # the same result as readable text
```

## Results

Each spectrum gets one row (CSV with `--out`; a dict from `redx.classify`).

| Column | Meaning |
|---|---|
| `class` | `qso`, `galaxy`, `star` or `inconclusive` |
| `z_RedX`, `z_lo`, `z_hi` | redshift and 68% interval (QSO and galaxy; empty otherwise) |
| `quality` | how far to trust a classified source: `secure`, `probable` or `low` (empty if inconclusive) |
| `reason` | why the result is `inconclusive`: one code (empty otherwise) |
| `warnings` | `;`-separated warnings on a classified source (empty if none) |
| `star_type`, `star_type_range`, `star_lum` | stars only: best spectral type, range of acceptable types, luminosity class (`dwarf`, `giant`, `subdwarf` or `ambiguous`) |

Inconclusive is preferred to a wrong answer: a spectrum that fails any test below is
`inconclusive`, with the first failed test in `reason`.

**quality**

| Value | Meaning |
|---|---|
| `secure` | no warning: a QSO with two or more emission lines at `z_RedX`, or a star with a clear class and median S/N per bin ≥ 4 |
| `probable` | classified, with at least one warning; every galaxy (stars can be fitted as low-z galaxies) |
| `low` | QSO from a `low_snr` spectrum whose emission lines pass every test |

**warnings**

| Value | Meaning |
|---|---|
| `single_line` | QSO redshift from one emission line: an alias is possible |
| `class_ambiguous` | the second-best class is within Δχ² = 2 |
| `marginal_snr` | median S/N per bin 2–4 |
| `low_snr` | median S/N per bin < 2, fewer than 10 bins, or fewer than 6 bins with flux > 1σ. The continuum then does not fix the class, so only a QSO is reported |

**reason**

| Value | Meaning |
|---|---|
| `no_data` | fewer than 6 usable bins |
| `too_noisy` | `low_snr` spectrum without QSO emission lines |
| `galaxy_or_qso` | galaxy fit, but the QSO fit has χ² within a factor 1.5 and emission lines of significance > 5 at its redshift |
| `no_lines` | QSO fit without emission lines at its redshift (`line_signif` ≤ 3) |
| `z_disagree` | the line-only redshift `z_lines` differs from `z_RedX` by more than 0.05 (1+z) |
| `weak_line` | a single emission line of significance < 6, or < 12 with reduced χ² > 20 |
| `lines_absent` | the emission lines of the QSO fit are observed at < 0.3 of their predicted strength, at > 5σ |
| `poor_fit` | no class fits: reduced χ² > 5 with a 10% template error added (`chi2_sys`) |

The same descriptions are in `redx.QUALITY`, `redx.WARNINGS` and `redx.REASONS`.
The figure (`--plot`) shows the spectrum, the best fit and these values.

**Diagnostic columns**

| Column | Meaning |
|---|---|
| `median_snr` | median S/N per bin of the binned spectrum |
| `n_bins`, `n_exp`, `n_clip` | bins; single exposures fitted (0: binned spectrum fitted); exposures rejected as outliers |
| `n_line_support` | emission lines that support the QSO `z_RedX` |
| `line_signif` | significance of the emission lines at the QSO `z_RedX` |
| `z_lines`, `n_lines` | redshift from the emission-line search alone, and the lines it uses |
| `redchi2` | reduced χ² of the best fit |
| `chi2_sys` | reduced χ² with a 10% template error added |
| `line_scale` | observed / predicted strength of the QSO emission lines |
| `signif` | √Δχ² between the best redshift and the best one > 0.05 away |
| `template`, `library` | best-fitting template; identifier of the template set |

## Method

The templates are fitted to the single exposures (SPIFF `results.csv` or the rows
of a `.tbl`), each a bandpass-averaged flux: the template is convolved with the
bandpass of the exposure's band (R = 39.5, 41.3, 40.7, 35.0, 111.5, 127.7 for bands
1–6) and evaluated at its central wavelength, with one non-negative amplitude per
redshift (Δz = 0.002; quasars 0 ≤ z ≤ 7, galaxies ≤ 3, stars z = 0). Exposures are
selected as SPIFF bins them (`.tbl`: `fit_ql` 1–4, no `FIT_ERROR` flag); outliers
(> 10σ and > 50% from the median of their neighbours within 0.02 in ln λ, `n_clip`)
are dropped, and 3% of the flux, the repeatability of a single SPHEREx measurement
(SPHEREx Explanatory Supplement v2.0), is added to each error. With fewer than 20
exposures the binned spectrum is fitted instead (templates averaged over a channel,
R ≈ 40 below 3.8 µm, ≈ 120 above, and over the bins). The line search and the
tests under Results use the binned spectrum. The class with the lowest χ² wins; a quasar also
needs its emission lines at the fitted redshift. P(z) ∝ exp(−χ²/2s) with
s = max(1, χ²_ν); the 68% interval includes a systematic 0.0033 (1+z).

Physical limits: stars are observed or model stellar spectra (no black bodies).
The IRTF stars are absolutely calibrated and have Gaia or Hipparcos
parallaxes, so a fit places the source at a distance: it may be no farther than
30 kpc for supergiants and bright giants (classes I, II; Galactic disc) and 300 kpc
for other stars (stellar halo), with the template star at its parallax + 2σ distance.
IRTF giants without a parallax and the planets are not used. Each source is fitted
by one template: RedX does not fit sums of sources (e.g. a quasar plus a star in the
same pixel).

A star is typed by comparing it with the SPIFF standards and the IRTF spectra over
the bins each one measures (only those holding ≥ 85% of the spectrum's (S/N)²),
under the same distance limits. `star_type` is the best match; `star_type_range`
spans the matches within 2 (2/dof)^½ max(1, χ²_ν) in reduced χ², and at least ±1
subtype.

Templates: Selsing et al. (2016) quasar composite, SMC-reddened (Pei 1992) and with
hot dust (Temple et al. 2021); SWIRE galaxies (Polletta et al. 2007); SUDA SPHEREx
ultracool dwarfs (Tu et al. 2026); SPIFF SPHEREx ultracool-dwarf standards (Gagné et
al. 2026); IRTF Spectral Library (Rayner et al. 2009, optional); BT-Settl stars
(Allard et al. 2012, optional). Line list: Vanden Berk et al. (2001).
Sources, checksums and licences: `redx/data/folder_description.txt`.

## Citation and licence

See `CITATION.cff`. Code: MIT. SUDA and SPIFF templates: CC BY 4.0; the other templates
are downloaded from their providers, whose terms apply.
