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
print(r['class'], r['z_RedX'], r['z_lo'], r['z_hi'], r['flags'], r['star_type'])
```

Input: a SPIFF run (`binned_spectrum.csv` with its `results.csv` in the same folder),
an IRSA forced-photometry `.tbl` (binned to 44 log-λ bins), or a binned
`.csv`/`.fits` with `wavelength_angstrom`, `flux_ujy_weighted_mean`,
`flux_ujy_weighted_sem`. Fluxes are f_ν in µJy. Keep `results.csv` next to
`binned_spectrum.csv`: the templates are fitted to the single exposures it lists
(`n_exp` in the results); without it RedX fits the binned spectrum.

Main outputs: `class` (`qso`, `galaxy`, `star`, `inconclusive`), `z_RedX` with its
68% interval `z_lo`–`z_hi`, the line-only redshift `z_lines`, `verdict` and `flags`;
for stars `star_type`, `star_type_range` and `star_lum` (`dwarf`, `giant`, `subdwarf`
or `ambiguous`).
A QSO needs emission lines at `z_RedX` (`line_signif` > 3) and a line-only redshift
`z_lines` within 0.05 (1+z); it is `secure` when at least two lines support it,
otherwise flagged `single_line_alias_risk`. Other flags: `class_ambiguous`,
`marginal_snr`. A fit that does not meet these is `inconclusive` with the reason in
`verdict` and `flags`: `z_lines_disagree`, `misfit_single_line` (reduced χ² > 20 with
one line of significance < 12), `weak_single_line` (one line, of significance < 6), `galaxy_or_qso` (galaxy fit, but the quasar fit is
within a factor 1.5 in χ² and has lines of significance > 5 at its redshift),
`lines_absent` (the emission lines of the quasar fit are observed at < 0.3 of their
predicted strength, at > 5σ; `line_scale` is that ratio), and for any class `poor_fit`
(no class fits: the best fit has reduced χ² > 5 with a 10% template systematic added
to the errors, `chi2_sys`).
The figure shows `SECURE` only for a secure QSO, and lists the flags.

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
rules below use the binned spectrum. The class with the lowest χ² wins; a quasar also
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
