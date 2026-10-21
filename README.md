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
(`$REDX_CACHE`). Optional warm-star models (BT-Settl, about 540 MB):
`python -m redx.sources --btsettl`.

## Use

```bash
redx spectrum.tbl                     # classify; --plot out.png writes the figure
redx --dir FOLDER --out results.csv   # every *.tbl, *.csv, *.fits in FOLDER
redx --dir RUNS --recursive           # also SPIFF RUNS/<id>_RA*_DEC*/binned_spectrum.csv, id = <id>
```

```python
import redx
r = redx.classify('spectrum.tbl')
print(r['class'], r['z_RedX'], r['z_lo'], r['z_hi'], r['flags'])
```

Input: an IRSA forced-photometry `.tbl` (binned to 44 log-λ bins), or a binned
`.csv`/`.fits` with `wavelength_angstrom`, `flux_ujy_weighted_mean`,
`flux_ujy_weighted_sem`. Fluxes are f_ν in µJy.

Main outputs: `class` (`qso`, `galaxy`, `star`, `inconclusive`), `z_RedX` with its
68% interval `z_lo`–`z_hi`, the line-only redshift `z_lines`, `verdict` and `flags`.
A QSO is `secure` when its lines are detected at `z_RedX` (`line_signif` > 3), at
least two lines support it, and `z_lines` agrees within 0.05 (1+z). Flags:
`single_line_alias_risk`, `z_lines_disagree`, `qso_no_line`, `blend` (quasar plus a
neighbour in the same pixel), `class_ambiguous`, `marginal_snr`.

## Method

Each template is averaged over a SPHEREx channel (R ≈ 40 below 3.8 µm, ≈ 120 above)
and over the spectrum's bins, then fitted with one non-negative amplitude per
redshift (Δz = 0.002; quasars 0 ≤ z ≤ 7, galaxies ≤ 3, stars z = 0). The class with
the lowest χ² wins; a quasar also needs its emission lines at the fitted redshift.
P(z) ∝ exp(−χ²/2s) with s = max(1, χ²_ν); the 68% interval includes a systematic
0.0036 (1+z).

Templates: Selsing et al. (2016) quasar composite, SMC-reddened (Pei 1992) and with
hot dust (Temple et al. 2021); SWIRE galaxies (Polletta et al. 2007); SUDA SPHEREx
ultracool dwarfs (Tu et al. 2026); BT-Settl stars (Allard et al. 2012, optional);
black bodies. Line list: Vanden Berk et al. (2001). Sources, checksums and licences:
`redx/data/folder_description.txt`.

## Citation and licence

See `CITATION.cff`. Code: MIT. SUDA templates: CC BY 4.0; the other templates are
downloaded from their providers, whose terms apply.
