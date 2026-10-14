"""The one place RedX figure style is defined.

Publication style: serif, ticks pointing in on all four sides, minimal in-plot
text. Import `apply()` rather than calling `plt.rcParams.update` in a script.
"""

RC = {
    'font.family': 'serif',
    'font.size': 13,
    'axes.linewidth': 1.0,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    'xtick.minor.visible': True,
    'ytick.minor.visible': True,
    'savefig.bbox': 'tight',
}

# Figure geometry and per-class template colours, shared by every RedX figure.
FIGSIZE = (7.6, 4.8)
DPI = 200
CLASS_COLOUR = {'qso': '#b40426', 'star': '#20558a', 'galaxy': '#2e7d32'}
SPEC_COLOUR = '#1a1a1a'
BAND_COLOUR = '#c4ccd6'
LINE_COLOUR = '#1f5fbf'


def apply(plt):
    """Apply the RedX rcParams to an already-imported pyplot."""
    plt.rcParams.update(RC)
