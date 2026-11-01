"""Diagnostic figure: spectrum with 1-sigma band, fitted model and emission lines at the fitted redshift."""
import os
import numpy as np
from . import style
from .templates import template_origin
from .lines_data import STRONG_LINES, SPHEREX_MIN_UM, SPHEREX_MAX_UM

_LABEL = {'Lya': r'Ly$\alpha$', 'CIV': 'C IV', 'CIII': 'C III]', 'MgII': 'Mg II',
          'Hb': r'H$\beta$', 'OIII': '[O III]', 'Ha': r'H$\alpha$'}
_XTICKS = [0.8, 1, 1.5, 2, 3, 4, 5]
_PLT = None


def _pyplot(show):
    """pyplot with a backend that can show a window if asked, Agg when headless."""
    global _PLT
    import matplotlib
    if _PLT is None:
        if not show and not os.environ.get('DISPLAY') and not os.environ.get('WAYLAND_DISPLAY'):
            matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        _PLT = plt
        style.apply(plt)
    if show and matplotlib.get_backend().lower() == 'agg':
        for backend in ('QtAgg', 'TkAgg', 'GTK3Agg', 'MacOSX'):
            try:
                _PLT.switch_backend(backend)
                break
            except Exception:
                continue
    return _PLT


def fmt_z(v):
    """Redshift to 3 decimals; strings unchanged."""
    if isinstance(v, str):
        return v
    try:
        v = float(v)
    except (TypeError, ValueError):
        return 'n/a'
    return f'{v:.3f}' if np.isfinite(v) else 'n/a'


def _model(result, spec, lam):
    """Fitted model on lam, scaled to spec; None without a fit."""
    from .core import _library
    from .fit import binned_library, model
    fit = result.get('_fit') or {}
    if 'dln' not in fit:
        return None
    lib = {t.name: t for t in binned_library(_library(), fit['dln'])}
    z = result['z_RedX'] if np.isfinite(result['z_RedX']) else fit['z']
    return model(spec, lib[result['template']], z, lam)


def diagnostic(result, out=None, show=False, annotations=None, spec=None,
               title_id=None, lines=True, info=True, dpi=style.DPI, figsize=style.FIGSIZE):
    """Figure for a classify() result. out: PNG path; annotations: extra box rows, e.g. {'z_spec': 4.8};
    spec: spectrum to draw instead (e.g. finer binning); lines: mark QSO lines; info: draw the text box."""
    plt = _pyplot(show)
    spec = spec if spec is not None else result['spec']
    cls, z = result.get('class', ''), result.get('z_RedX', np.nan)
    lam, fl, sig = spec.lam_um, spec.flux, spec.sigma
    fig, ax = plt.subplots(figsize=figsize)
    ax.fill_between(lam, fl - sig, fl + sig, color=style.BAND_COLOUR, alpha=0.55, lw=0, zorder=1)
    ax.plot(lam, fl, '-', color=style.SPEC_COLOUR, lw=1.2, zorder=3, label='SPHEREx')
    lg = np.geomspace(SPHEREX_MIN_UM, SPHEREX_MAX_UM, 900)
    model = _model(result, spec, lg)
    if model is not None:
        ax.plot(lg, model, '-', lw=1.5, zorder=2, color=style.CLASS_COLOUR.get(cls, '#b40426'), label='RedX model')

    rows = []
    if info:
        rows = ([f'ID {title_id}'] if title_id is not None else []) + [f'class    = {cls.upper()}']
        if cls in ('qso', 'galaxy'):
            rows.append(f'z_RedX   = {fmt_z(z)}')
        rows += [f'{k:<9}= {fmt_z(v)}' for k, v in (annotations or {}).items()]
        if result.get('quality'):
            rows.append(f"quality  = {result['quality']}")
        if result.get('reason'):
            rows.append(f"reason   = {result['reason']}")
        if result.get('warnings'):
            rows.append(f"warnings = {result['warnings'].replace(';', ', ')}")
        if template_origin(result.get('template')):
            rows.append(f"template = {template_origin(result.get('template'))}")
    f = fl[np.isfinite(fl)]
    if f.size >= 5 and np.percentile(f, 99) > np.percentile(f, 1):
        lo, hi = np.percentile(f, [1, 99])
        box = (12.0 * len(rows) + 14.0) / (72.0 * fig.get_figheight() * ax.get_position().height)   # axes fraction
        y0, y1 = lo - 0.12 * (hi - lo), hi + 0.28 * (hi - lo)
        if rows:
            y1 = max(y1, y0 + (f.max() - y0) / max(0.94 - box, 0.3))     # keep the data below the text box
        ax.set_ylim(y0, y1)
    ymin, ymax = ax.get_ylim()
    if lines and cls == 'qso' and np.isfinite(z):
        for name, rest, _ in STRONG_LINES:
            lo = rest * (1 + z) / 1e4
            if SPHEREX_MIN_UM <= lo <= SPHEREX_MAX_UM:
                ax.axvline(lo, color=style.LINE_COLOUR, ls='--', lw=0.8, alpha=0.8, zorder=2)
                ax.text(lo, ymin + 0.03 * (ymax - ymin), _LABEL[name], rotation=90, va='bottom',
                        ha='right', fontsize=9, color=style.LINE_COLOUR)
    ax.set_xscale('log')
    ax.set_xlim(SPHEREX_MIN_UM, SPHEREX_MAX_UM)
    ax.set_xticks(_XTICKS)
    ax.set_xticklabels([str(t) for t in _XTICKS])
    ax.set_xlabel(r'observed wavelength $\lambda$ [$\mu$m]')
    ax.set_ylabel(r'$f_\nu$ [$\mu$Jy]')

    if rows:
        ax.text(0.025, 0.965, '\n'.join(rows), transform=ax.transAxes, va='top', ha='left', fontsize=10,
                family='monospace', bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='#888', alpha=0.92))
    ax.legend(loc='best', fontsize=9, frameon=False)
    fig.tight_layout()
    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        fig.savefig(out, dpi=dpi)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig
