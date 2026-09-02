"""Shared publication style: two-column conference format.

`apply()` mutates the global rcParams, so call it once per session (the plotting
helpers call it for you).  Colors are indexed by agent, line styles by method, so
a reader tracks who is who by hue and what is what by dash pattern -- the two
stay legible when a figure is printed in grayscale.
"""
import matplotlib.pyplot as plt

STYLE = {
    'font.family': 'serif',
    'font.serif': ['Times', 'Times New Roman', 'Computer Modern Roman'],
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'legend.fontsize': 14,
    'xtick.labelsize': 10,      # ticks stay small; everything else is larger
    'ytick.labelsize': 10,
    'lines.linewidth': 1.5,
    'text.usetex': False,
    'mathtext.fontset': 'cm',
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
    'xtick.major.pad': 4,       # tick mark to tick label
    'ytick.major.pad': 4,
    'axes.labelpad': 6,         # tick labels to axis label
}

# Palette
BLUE = '#1F77B4'
RED = '#D62728'
PINK = '#D81B5F'
ORANGE = '#FE6100'
GRAY = '#AAAAAA'
GREEN = '#008000'
PURPLE = '#9467BD'
CYAN = '#17BECF'
BROWN = '#8C564B'
TEAL = '#2CA02C'

#: One color per agent, cycled if there are more agents than entries.
AGENT_COLORS = [BLUE, RED, ORANGE, GREEN, PURPLE, CYAN, BROWN, TEAL]

#: One line style per method, in the order [nominal, then each comparison].
METHOD_STYLES = [
    dict(linestyle='-',               linewidth=1.8, alpha=1.0),
    dict(linestyle=(0, (5, 2)),       linewidth=1.5, alpha=0.9),
    dict(linestyle=(0, (1, 1)),       linewidth=1.5, alpha=0.9),
    dict(linestyle=(0, (3, 1, 1, 1)), linewidth=1.5, alpha=0.9),
]


def agent_color(i):
    """Color for agent `i`, cycling past the end of the palette.

    Parameters
    ----------
    i : int
        Agent index, zero-based.

    Returns
    -------
    color : str
        A hex color string from `AGENT_COLORS`.
    """
    return AGENT_COLORS[i % len(AGENT_COLORS)]


def method_style(j):
    """Line style for method `j` (0 = nominal), cycling past the end.

    Parameters
    ----------
    j : int
        Method index, zero-based, where 0 is the nominal baseline.

    Returns
    -------
    style : dict
        Keyword arguments for `Axes.plot` -- `linestyle`, `linewidth`, `alpha`.
        This is the shared dict from `METHOD_STYLES`, not a copy: mutating it
        changes the palette for every later call.  To override one key, build a
        new dict instead, e.g. ``{**method_style(1), 'alpha': 0.55}``.
    """
    return METHOD_STYLES[j % len(METHOD_STYLES)]


def apply():
    """Install the style globally by updating `matplotlib.rcParams`.

    Idempotent, and called by every helper in `plotting.py`, so notebooks do not
    have to remember it.

    Returns
    -------
    None
    """
    plt.rcParams.update(STYLE)
