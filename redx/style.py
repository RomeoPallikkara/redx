"""Figure style: serif, ticks in on all sides."""

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
FIGSIZE = (7.6, 4.8)
DPI = 200
CLASS_COLOUR = {'qso': '#b40426', 'star': '#20558a', 'galaxy': '#2e7d32'}
SPEC_COLOUR = '#1a1a1a'
BAND_COLOUR = '#c4ccd6'
LINE_COLOUR = '#1f5fbf'


def apply(plt):
    plt.rcParams.update(RC)
