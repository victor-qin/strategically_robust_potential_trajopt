"""Publication figures for the trajectory-game solvers.

Every function here is optional: nothing in `solvers.py` or `adversary.py`
imports it, so the solver and the benchmarks run with no plotting stack
installed.

Two conventions, both departures from typical research-code plotting:

* Nothing is written to disk unless you pass `save_path`.  A figure helper that
  silently drops PNGs in the working directory is a nuisance in a notebook and a
  hazard in a batch job.
* Every helper accepts `ax=` and returns the axis, so panels compose into a
  larger figure instead of each owning a figure of its own.

Figures are sized for a two-column page: 3.5 inches square, which is one column
at typical margins.
"""
import warnings
from itertools import combinations

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D

import plot_style
from adversary import precompute_selected, selection_matrix, worst_case_relative
from solvers import split_solution

PANEL = 3.5           # inches; total figure width, one column of a two-column page
BOX = 3.10            # inches; the square plot area, identical in every figure
MARGIN_LEFT = 0.36    # room for the widest y tick labels plus their pad
MARGIN_RIGHT = 0.04
MARGIN_TOP = 0.05
MARGIN_XTICK = 0.24   # room for the x tick labels below the box
LEGEND_ROW = 0.33     # height of one legend row, inches
LEGEND_GAP = 0.22     # inches from the x tick labels to the legend top
LEGEND_BOTTOM = 0.04  # margin below the last legend row
LEGEND_COLSPACE_PLAIN = 0.5     # column gap in the per-agent layout, font units
LEGEND_COLSPACE_GROUPED = 1.2   # swatch-to-methods gap, font units
METHOD_LENGTH = 1.5   # method-line handle width, font units
SWATCH_SEGMENT = 0.75 # one agent segment: half a method line
SWATCH_HEIGHT = 1.6   # agent-swatch handle height, font units
SWATCH_PAD = 0.10     # legend padding around that handle box, inches


def _strip_below(legend_height):
    """Extra figure height a legend of `legend_height` inches needs.

    The legend top sits LEGEND_GAP below the *plot box*, and MARGIN_XTICK inches
    of that span is already reserved for the tick labels, so only the remainder
    is new height.  Adding LEGEND_GAP outright double-counts that overlap and
    leaves a band of whitespace under every figure.

    Parameters
    ----------
    legend_height : float
        Height of the tallest legend block, inches.

    Returns
    -------
    strip : float
        Inches to add below the tick-label zone.
    """
    return max(0.0, LEGEND_GAP - MARGIN_XTICK) + legend_height + LEGEND_BOTTOM


def _fixed_box_figure(strip):
    """Figure whose plot box is exactly BOX x BOX inches, whatever the labels say.

    `tight_layout` hands the axes whatever the decorations leave over, so a plot
    with single-digit tick labels gets a larger box than one with decimals -- the
    8-agent grid came out 3% wider than the head-on grid for no reason but the
    text beside it.  Placing the axes at an explicit rectangle instead makes every
    grid identical, which is what a figure set compared side by side needs.  Only
    the legend strip then varies the overall height.

    Parameters
    ----------
    strip : float
        Height in inches to reserve below the box for the legend.

    Returns
    -------
    fig : matplotlib Figure
    ax : matplotlib Axes
    """
    width = MARGIN_LEFT + BOX + MARGIN_RIGHT
    height = MARGIN_TOP + BOX + MARGIN_XTICK + strip
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes([MARGIN_LEFT / width, (strip + MARGIN_XTICK) / height,
                       BOX / width, BOX / height])
    return fig, ax


def _legend_shape(n_entries):
    """Columns and reserved height for a figure-level legend of `n_entries`.

    The strip has to grow with the row count: a fixed reservation that fits a
    2-agent legend puts an 8-agent one on top of the x tick labels.  It is sized
    to the legend itself, so the legend can be anchored to the *top* of the strip
    and sit just under the axis rather than floating at the foot of the figure.

    Parameters
    ----------
    n_entries : int
        Total legend entries, i.e. one per agent plus one per method.

    Returns
    -------
    ncol : int
        Number of legend columns.
    strip : float
        Figure height in inches to reserve below the axis, padding included.
    """
    ncol = 3 if n_entries <= 8 else 4
    nrow = -(-n_entries // ncol)                      # ceiling division
    return ncol, _strip_below(LEGEND_ROW * nrow)


def _trajectories(results, n_a, H, sdim, cdim):
    """Unpack a list of OptimizeResults into a list of state trajectories.

    Parameters
    ----------
    results : list of OptimizeResult
    n_a, H, sdim, cdim : int
        Number of agents, horizon length, state and control dimensions.

    Returns
    -------
    trajectories : list of ndarray
        One array of shape (n_a, H+1, sdim) per result, in the order given.
    """
    return [split_solution(r, n_a, H, sdim, cdim)[0] for r in results]


def _save(fig, save_path, tight=True):
    """Write a figure to disk, or do nothing when no path was requested.

    Parameters
    ----------
    fig : matplotlib Figure
    save_path : str, pathlib.Path or None
        Destination.  None is the default everywhere, so no helper writes a file
        as a side effect.
    tight : bool
        Crop to the drawn content.  False writes the figure at its declared size,
        which is what the fixed-box plots need: cropping trims a different amount
        of whitespace depending on how wide the tick labels are, so two figures
        with identical plot boxes would come out different widths and a reader
        scaling both to one column width would see the boxes differ again.

    Returns
    -------
    None
    """
    if save_path is not None:
        # bbox_inches=None would fall back to rcParams['savefig.bbox'], which
        # plot_style sets to 'tight'; an explicit box is the only way to opt out.
        fig.savefig(save_path, dpi=300,
                    bbox_inches='tight' if tight else fig.bbox_inches,
                    pad_inches=0.02 if tight else 0.0)


class HandlerAgentSwatch(HandlerBase):
    """Draw a tuple of lines as a grid of segments inside one legend handle box.

    `HandlerTuple` packs sub-handles into a single row, which at eight agents
    leaves each color a sliver.  Wrapping them over `nrow` rows lets every
    segment be a readable fraction of a method line's length while the entry
    still costs one legend slot.

    Parameters
    ----------
    nrow : int
        Rows to wrap the segments over.
    gap : float
        Fraction of each cell left blank between segments.
    """

    def __init__(self, nrow=2, gap=0.16, **kw):
        """Store the grid shape.

        Parameters
        ----------
        nrow : int
            Rows to wrap the segments over.
        gap : float
            Fraction of each cell left blank between segments.
        **kw
            Forwarded to `HandlerBase`.

        Returns
        -------
        None
        """
        self.nrow = nrow
        self.gap = gap
        super().__init__(**kw)

    def create_artists(self, legend, orig_handle, xdescent, ydescent,
                       width, height, fontsize, trans):
        """Lay the sub-handles out row-major inside the handle box.

        Parameters
        ----------
        legend : matplotlib Legend
            The legend being drawn.  Unused; part of the handler protocol.
        orig_handle : tuple of Line2D
            The sub-handles to grid, from `_agent_swatch`.
        xdescent, ydescent, width, height : float
            Handle box geometry, in display units.
        fontsize : float
            Legend font size.  Unused; the box is already scaled by it.
        trans : matplotlib Transform
            Transform mapping the handle box into figure space.

        Returns
        -------
        artists : list of Line2D
            One short segment per sub-handle, placed in the grid.
        """
        n = len(orig_handle)
        nrow = max(1, min(self.nrow, n))
        ncol = -(-n // nrow)
        cell = width / ncol
        artists = []
        for k, proto in enumerate(orig_handle):
            r, c = divmod(k, ncol)
            y = height * (1.0 - (r + 0.5) / nrow) - ydescent
            x0 = -xdescent + c * cell
            line = Line2D([x0 + self.gap * cell, x0 + (1.0 - self.gap) * cell], [y, y])
            line.update_from(proto)
            line.set_transform(trans)
            artists.append(line)
        return artists


def _swatch_rows(n_a):
    """How many rows to wrap an n_a-color swatch over.

    Parameters
    ----------
    n_a : int

    Returns
    -------
    nrow : int
    """
    return 1 if n_a <= 3 else -(-n_a // 3)          # at most three columns wide


def _swatch_length(n_a):
    """Handle width that gives each agent segment a consistent length.

    The swatch shares one handle box across all its columns, so a fixed width
    would make each segment shrink as agents are added -- and at four agents,
    where the grid is only two columns wide, a width sized for eight leaves the
    swatch as wide as the whole method column and the two legends collide.

    Parameters
    ----------
    n_a : int
        Number of agents.

    Returns
    -------
    length : float
        Handle width in font units, so each segment is `SWATCH_SEGMENT` long.
    """
    ncol = -(-n_a // _swatch_rows(n_a))
    return SWATCH_SEGMENT * ncol


def _agent_swatch(n_a):
    """One legend handle carrying every agent color side by side.

    Lets a legend say "agents" once over a band of all n_a colors instead of
    spending an entry per agent.  Drawn by `HandlerAgentSwatch`, which grids the
    sub-handles inside a single handle box.

    Parameters
    ----------
    n_a : int
        Number of agents.

    Returns
    -------
    handle : tuple of Line2D
        Pass to `Axes.legend` with `handler_map={tuple: HandlerAgentSwatch(...)}`.
    """
    return tuple(Line2D([0], [0], color=plot_style.agent_color(i), lw=2.4)
                 for i in range(n_a))


def _row_major(items, ncol):
    """Reorder a legend list so matplotlib's column-major fill reads row-major.

    `Axes.legend` splits its handle list into `ncol` consecutive chunks and
    stacks each one vertically, so a 3-entry legend in 2 columns reads
    1, 3, 2 across the page.  Permuting the input undoes that.

    Parameters
    ----------
    items : list
        Entries in the order they should read, left to right then down.
    ncol : int
        Column count the legend will use.

    Returns
    -------
    reordered : list
        Same entries, ordered so the rendered grid reads as `items` does.
    """
    n = len(items)
    nrow = -(-n // ncol)
    return [items[r * ncol + c] for c in range(ncol) for r in range(nrow)
            if r * ncol + c < n]


def _legend_within_axes(ax, handles, labels, ncol, handlelength, reorder=True,
                        tol=0.04, **kw):
    """Place a legend under the axes, narrowing it until it fits the axes width.

    A figure legend that overhangs the plot box reads as a separate object; one
    bounded by the grid reads as part of it.  matplotlib will not wrap a legend
    that is too wide, so the column count is reduced and the legend redrawn until
    it fits, down to a single column.

    Parameters
    ----------
    ax : matplotlib Axes
    handles, labels : list
        As for `Axes.legend`.
    ncol : int
        Starting column count; only ever reduced.
    handlelength : float
        Handle width in font units.
    tol : float
        Fractional overhang tolerated before a column is dropped.  A legend a
        couple of percent wider than the plot box still reads as belonging to it;
        insisting on an exact fit costs more in cramped spacing than it buys.
    reorder : bool
        Permute the entries so the grid reads row-major.  False keeps
        matplotlib's own column-major fill, which is what you want when
        consecutive entries belong together -- agents in one column, methods in
        the next -- rather than forming a flat list.
    **kw
        Forwarded to `Axes.legend`.

    Returns
    -------
    legend : matplotlib Legend
    """
    fig = ax.figure
    order = _row_major if reorder else (lambda items, _: items)
    for cols in range(ncol, 0, -1):
        leg = ax.legend(order(handles, cols), order(labels, cols),
                        ncol=cols, handlelength=handlelength, **kw)
        fig.canvas.draw()
        if leg.get_window_extent().width <= ax.get_window_extent().width * (1 + tol):
            return leg
        leg.remove()
    return ax.legend(handles, labels, ncol=1, handlelength=handlelength, **kw)


def legend_right(ax, handles=None, labels=None, widen=True, gap=0.03, **kw):
    """Put the legend outside the right edge of the axes, where it cannot cover data.

    An in-axes legend has to find empty space among the curves, and with several
    long entries there is none.  Outside the box it never overlaps the data.

    Parameters
    ----------
    ax : matplotlib Axes
    handles, labels : list, optional
        As for `Axes.legend`.  Taken from the labelled artists when omitted.
    widen : bool
        Grow the figure by the legend's width, then re-run `tight_layout`, so the
        axes keep the size they had.  Without this the legend's width comes out
        of the plot area.  Pass False when `ax` is one panel of a layout the
        caller owns.
    gap : float
        Space between the axes and the legend, as a fraction of the axes width.
    **kw
        Forwarded to `Axes.legend`.

    Returns
    -------
    legend : matplotlib Legend
    """
    if handles is None:
        handles, labels = ax.get_legend_handles_labels()
    kw = {'frameon': False, 'borderaxespad': 0.0, **kw}
    leg = ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(1.0 + gap, 1.0), **kw)
    if widen:
        fig = ax.figure
        fig.canvas.draw()
        extra = (leg.get_window_extent().width
                 + gap * ax.get_window_extent().width) / fig.dpi
        w, h = fig.get_size_inches()
        fig.set_size_inches(w + extra, h)
        fig.tight_layout()
    return leg


def _method_style_legend(ax, labels, styles, widen):
    """Side legend that keys color to method and line style to what is drawn.

    Labelling every curve would list each method-style pair ("wider, worst
    case", ...), six long entries for three methods.  Keying the two encodings
    separately needs one entry per method plus one per style, all of them short.

    Parameters
    ----------
    ax : matplotlib Axes
    labels : list of str
        Method names, in palette order.
    styles : list of (dict, str)
        Line keyword arguments and a label for each style drawn, e.g. solid for
        the plan and dashed for the worst case.
    widen : bool
        Forwarded to `legend_right`.

    Returns
    -------
    legend : matplotlib Legend
    """
    methods = [Line2D([0], [0], color=plot_style.agent_color(k), lw=2.4)
               for k in range(len(labels))]
    spacer = Line2D([], [], linestyle='none')
    keys = [Line2D([0], [0], color='0.35', **style) for style, _ in styles]
    return legend_right(ax, methods + [spacer] + keys,
                        list(labels) + [''] + [name for _, name in styles],
                        widen=widen)


# ---------------------------------------------------------------------------
# Trajectories in the plane
# ---------------------------------------------------------------------------
def plot_trajectories(result_nominal, result_others, n_a, H, sdim, cdim, xf,
                      method_labels=None, ax=None, title=None, xlim=None,
                      ylim=None, show_legend=True, group_agents=None,
                      legend_fontsize=None, legend_columnspacing=None,
                      save_path=None):
    """Overlay each method's trajectories, one color per agent.

    Agents are distinguished by color and methods by dash pattern, so a single
    panel can carry n_a * (1 + len(result_others)) curves and still be read.
    Start states are circles, targets are stars.

    Parameters
    ----------
    result_nominal : OptimizeResult
        Drawn solid; the baseline every other method is compared against.
    result_others : list of OptimizeResult
        Drawn dashed/dotted in the order given.
    n_a, H, sdim, cdim : int
        Number of agents, horizon length, state and control dimensions.
    xf : ndarray, shape (n_a, sdim)
        Targets, marked with stars.
    method_labels : list of str, optional
        One label per method including the nominal, e.g.
        ``['nominal', 'strat. robust', 'wider']``.  Must match the order
        `result_nominal` then `result_others`.
    ax : matplotlib Axes, optional
        Draw into an existing axis instead of making a figure.  The figure-level
        legend is drawn only when this helper owns the figure.
    title : str, optional
        Axis title, e.g. a panel label.
    xlim, ylim : (float, float), optional
        Axis limits.  Left to matplotlib when None.
    show_legend : bool
        Draw a legend below the axis, sized to its own row count and never wider
        than the plot box.
    group_agents : bool, optional
        Collapse the per-agent entries into a single "agents" swatch carrying
        every color, which takes a 10-entry legend at 8 agents down to 3.  None
        decides by agent count: two agents keep their own labels, since "Agent 1"
        and "Agent 2" cost two slots and a two-color band is not worth reading;
        beyond that the swatch wins.  True or False forces it.
    legend_fontsize : float, optional
        Legend text size in points.  None uses `plot_style`'s `legend.fontsize`.
        Raising it may force the legend onto more rows, since it is kept inside
        the plot box.
    legend_columnspacing : float, optional
        Horizontal separation between legend columns, in font units.  In the
        per-agent layout it is matplotlib's own `columnspacing`, defaulting to
        `LEGEND_COLSPACE_PLAIN`.  In the grouped layout it is the gap between the
        agent swatch and the method block, which are centred together beneath the
        plot: smaller values tighten the pair, larger ones spread it, and it is
        clamped once the pair reaches the edges of the box.  None uses
        `LEGEND_COLSPACE_GROUPED`.
    save_path : str or pathlib.Path, optional
        Write the figure here.  Nothing is written when None.

    Returns
    -------
    ax : matplotlib Axes
    """
    plot_style.apply()
    if group_agents is None:
        group_agents = n_a > 2
    x_nom = split_solution(result_nominal, n_a, H, sdim, cdim)[0]
    x_others = _trajectories(result_others, n_a, H, sdim, cdim)
    n_methods = 1 + len(x_others)
    if method_labels is None:
        method_labels = ['nominal'] + [f'method {j + 1}' for j in range(len(x_others))]

    if group_agents:
        # The swatch occupies a single handle box however many colour rows it
        # grids into -- its height comes from SWATCH_HEIGHT, not from the row
        # count -- so reserve the taller of the two blocks by measure rather than
        # by row.  Counting a row per colour row left the 8-agent figure 0.25in
        # of empty space below its legend.
        fs_strip = legend_fontsize or plt.rcParams['legend.fontsize']
        swatch_h = SWATCH_HEIGHT * fs_strip / 72.0 + SWATCH_PAD
        ncol = 1
        strip = _strip_below(max(n_methods * LEGEND_ROW, swatch_h))
    else:
        ncol, strip = _legend_shape(n_a + n_methods)
    own_fig = ax is None
    if own_fig:
        fig, ax = _fixed_box_figure(strip if show_legend else 0.0)
    else:
        fig = ax.figure

    for i in range(n_a):
        c = plot_style.agent_color(i)
        ax.plot(x_nom[i, :, 0], x_nom[i, :, 1], color=c, **plot_style.method_style(0))
        ax.scatter(x_nom[i, 0, 0], x_nom[i, 0, 1], marker='o', s=30, zorder=5,
                   color=c, edgecolors='black', linewidths=0.5)
        ax.scatter(xf[i, 0], xf[i, 1], marker='*', s=50, zorder=5,
                   color=c, edgecolors='black', linewidths=0.3)
        for j, x_other in enumerate(x_others):
            ax.plot(x_other[i, :, 0], x_other[i, :, 1], color=c,
                    **plot_style.method_style(j + 1))

    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_box_aspect(1)
    ax.grid(True, linewidth=0.3, alpha=0.3)
    ax.tick_params(direction='in', top=True, right=True)
    if title is not None:
        ax.set_title(title, pad=4)

    if show_legend and own_fig:
        method_handles = [Line2D([0], [0], color='gray',
                                 **{k: v for k, v in plot_style.method_style(m).items()
                                    if k in ('linestyle', 'linewidth')})
                          for m in range(n_methods)]
        # Anchored to the axes box rather than to figure fractions, so the gap
        # below the tick labels stays constant whatever the figure height and
        # however many rows the legend are needed.
        if group_agents:
            # Two legends stacked, both pinned to the left edge of the plot box.
            # A single legend forces one handlelength on every entry, which caps
            # each agent segment at a fraction of a method line; giving the swatch
            # its own legend lets it run long while the method dashes stay normal.
            # Anchoring at 0 and 1 bounds the pair by the plot box.  At 14pt the
            # handles have to be a little shorter than at 12pt for the two to
            # clear each other across 3.5in -- see METHOD_LENGTH/SWATCH_SEGMENT.
            fs = legend_fontsize or plt.rcParams['legend.fontsize']
            agents = ax.legend([_agent_swatch(n_a)], ['agents'], loc='upper left',
                               bbox_to_anchor=(0.0, -LEGEND_GAP / BOX), frameon=False,
                               fontsize=fs, handlelength=_swatch_length(n_a),
                               handleheight=SWATCH_HEIGHT, handletextpad=0.8,
                               borderaxespad=0.0,
                               handler_map={tuple: HandlerAgentSwatch(_swatch_rows(n_a))})
            ax.add_artist(agents)
            methods = ax.legend(method_handles, list(method_labels[:n_methods]),
                                loc='upper left', bbox_to_anchor=(0.0, -LEGEND_GAP / BOX),
                                frameon=False, ncol=1, fontsize=fs,
                                handlelength=METHOD_LENGTH, handletextpad=0.7,
                                labelspacing=0.7, borderaxespad=0.0)
            # Measure both blocks, then centre the pair with the requested gap
            # between them.  Placing by measurement rather than pinning either one
            # to an edge is what lets the gap open as well as close: the pair grows
            # outwards from the middle until it reaches the box.
            fig.canvas.draw()
            inv = ax.transAxes.inverted()
            w_a = agents.get_window_extent().transformed(inv).width
            w_m = methods.get_window_extent().transformed(inv).width
            spacing = (LEGEND_COLSPACE_GROUPED if legend_columnspacing is None
                       else legend_columnspacing)
            gap = min(spacing * (fs / 72.0) / BOX, max(0.0, 1.0 - w_a - w_m))
            left = max(0.0, 0.5 - (w_a + gap + w_m) / 2.0)
            agents.set_bbox_to_anchor((left, -LEGEND_GAP / BOX), transform=ax.transAxes)
            methods.set_bbox_to_anchor((left + w_a + gap, -LEGEND_GAP / BOX),
                                       transform=ax.transAxes)
            if w_a + w_m > 1.0:
                # No gap exists: the two blocks are wider than the plot box.
                warnings.warn(
                    f'legend blocks are {100 * (w_a + w_m):.0f}% of the plot width '
                    f'at {n_a} agents and fontsize {fs:g}, so they overlap; reduce '
                    f'legend_fontsize, shorten the method labels, or pass '
                    f'group_agents=False', stacklevel=2)
        else:
            handles = [Line2D([0], [0], color=plot_style.agent_color(i), lw=1.5,
                              linestyle='-') for i in range(n_a)] + method_handles
            labels = [f'Agent {i + 1}' for i in range(n_a)] + list(method_labels[:n_methods])
            # Three columns, filled column-major so the agents occupy one column
            # and the methods the next -- the grouping carries meaning here, so
            # the row-major reorder is switched off.  It only fits the plot box
            # with tighter handles than the grouped path needs.
            _legend_within_axes(ax, handles, labels, max(ncol, 3), 0.9,
                                reorder=False, loc='upper center', frameon=False,
                                bbox_to_anchor=(0.5, -LEGEND_GAP / BOX),
                                fontsize=legend_fontsize or plt.rcParams['legend.fontsize'],
                                columnspacing=(LEGEND_COLSPACE_PLAIN
                                               if legend_columnspacing is None
                                               else legend_columnspacing),
                                handletextpad=0.4, labelspacing=0.6,
                                borderaxespad=0.0)

    _save(fig, save_path, tight=False)
    return ax


def plot_trajectories_by_method(results, labels, n_a, H, sdim, cdim, xf, ax=None,
                                title=None, xlim=None, ylim=None, show_legend=True,
                                save_path=None):
    """Every agent's path, colored by method rather than by agent.

    The companion to `plot_trajectories`, for scenarios where individual agent
    identity carries no argument -- the 8-agent circle, where the claim is about
    what happens to the formation, not to agent 5.  Pooling the agents into one
    color per method turns a 10-entry legend into a 2-entry one, so the labels
    that do matter can be read at full size, and it puts the two bundles of
    curves in direct contrast instead of asking the reader to pair up dash
    patterns across eight hues.

    Start states are drawn once as small gray dots and targets as gray stars,
    since they are shared across methods.

    Parameters
    ----------
    results : list of OptimizeResult
        One per method, drawn in palette order.
    labels : list of str
        One per entry in `results`.
    n_a, H, sdim, cdim : int
        Number of agents, horizon length, state and control dimensions.
    xf : ndarray, shape (n_a, sdim)
        Targets, marked with stars.
    ax : matplotlib Axes, optional
        Draw into an existing axis instead of making a figure.
    title : str, optional
        Axis title.
    xlim, ylim : (float, float), optional
        Axis limits.  Left to matplotlib when None.
    show_legend : bool
        Draw a legend below the axis, one entry per method.
    save_path : str or pathlib.Path, optional
        Write the figure here.  Nothing is written when None.

    Returns
    -------
    ax : matplotlib Axes
    """
    plot_style.apply()
    trajectories = _trajectories(results, n_a, H, sdim, cdim)

    ncol, strip = _legend_shape(len(results))
    own_fig = ax is None
    if own_fig:
        fig, ax = _fixed_box_figure(strip if show_legend else 0.0)
    else:
        fig = ax.figure

    for k, x in enumerate(trajectories):
        color = plot_style.agent_color(k)
        # Thinner than a per-agent plot: n_a curves share each color here, so the
        # default weight turns the crossing region into a solid block.
        style = {**plot_style.method_style(0), 'color': color, 'linewidth': 1.4}
        for i in range(n_a):
            ax.plot(x[i, :, 0], x[i, :, 1], **style)

    x_first = trajectories[0]
    ax.scatter(x_first[:, 0, 0], x_first[:, 0, 1], marker='o', s=26, zorder=5,
               color='0.35', edgecolors='white', linewidths=0.6)
    ax.scatter(xf[:, 0], xf[:, 1], marker='*', s=60, zorder=5,
               color='0.35', edgecolors='white', linewidths=0.4)

    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_box_aspect(1)
    ax.grid(True, linewidth=0.3, alpha=0.3)
    ax.tick_params(direction='in', top=True, right=True)
    if title is not None:
        ax.set_title(title, pad=4)

    if show_legend and own_fig:
        handles = [Line2D([0], [0], color=plot_style.agent_color(k), lw=2.2,
                          linestyle='-') for k in range(len(labels))]
        _legend_within_axes(ax, handles, list(labels), ncol, 2.0,
                            loc='upper center', frameon=False,
                            bbox_to_anchor=(0.5, -LEGEND_GAP / BOX),
                            columnspacing=1.4, borderaxespad=0.0)

    _save(fig, save_path, tight=False)
    return ax


def animate_trajectories(result_nominal, result_others, n_a, H, sdim, cdim, xf, dt,
                         which=0, method_labels=None, xlim=None, ylim=None,
                         show_legend=True, save_path=None, fps=5):
    """Animate one method's motion against the other methods' faint paths.

    Every method keeps the dash pattern `plot_trajectories` gives it, so a frame
    reads with the static figure's key.  The methods not animated are drawn in
    full from the first frame, faded, as the fixed reference the moving one is
    judged against.

    Parameters
    ----------
    result_nominal : OptimizeResult
        Drawn as a faint static path unless `which` animates it.
    result_others : list of OptimizeResult
        Candidates to animate; `which` selects among them, and the rest are
        drawn as faint static paths.
    n_a, H, sdim, cdim : int
        Number of agents, horizon length, state and control dimensions.
    xf : ndarray, shape (n_a, sdim)
        Targets, marked with stars.
    dt : float
        Timestep, used for the on-figure clock.
    which : int or sequence of int
        Index into `result_others` to animate, or several to animate together,
        each marking its current position with its own marker shape.  -1
        animates the nominal itself, as does an empty `result_others`.
    method_labels : list of str, optional
        One label per method including the nominal, in the order
        `result_nominal` then `result_others`, as for `plot_trajectories`.
    xlim, ylim : (float, float), optional
        Axis limits.  Both must be given to take effect; otherwise the limits
        are taken from the data with a 0.5 margin.
    show_legend : bool
        Key the methods below the axis.  Agents are not keyed: color is the only
        thing distinguishing them, and it is the same in every frame.
    save_path : str or pathlib.Path, optional
        Write a GIF here (requires pillow).  Nothing is written when None.
    fps : int
        Frames per second of the saved GIF.

    Returns
    -------
    fig, anim : Figure, FuncAnimation
        Keep a reference to `anim` -- matplotlib drops the animation when it is
        garbage collected.  In a notebook, render with
        ``IPython.display.HTML(anim.to_jshtml())``, or embed the saved GIF, which
        keeps the notebook file far smaller.
    """
    plot_style.apply()
    x_nom = split_solution(result_nominal, n_a, H, sdim, cdim)[0]
    x_others = _trajectories(result_others, n_a, H, sdim, cdim)
    if not x_others:
        which = [-1]
    elif np.ndim(which) == 0:
        which = [which]
    # Method index as plot_style counts it: 0 is the nominal, j + 1 is result_others[j].
    methods = [0 if j < 0 else j + 1 for j in which]
    x_all = [x_nom] + x_others
    markers = 'os^D'
    faint = 0.15
    if method_labels is None:
        method_labels = ['nominal'] + [f'method {j + 1}' for j in range(len(x_others))]

    ncol, strip = _legend_shape(len(x_all))
    fig, ax = _fixed_box_figure(strip if show_legend else 0.0)
    ax.set_box_aspect(1)
    ax.grid(True, linewidth=0.3, alpha=0.3)
    ax.tick_params(direction='in', top=True, right=True)

    for m, x in enumerate(x_all):              # faint static paths for context
        if m in methods:
            continue
        # A dash pattern lays down a fraction of a solid line's ink, so at the
        # same alpha a faded dotted path all but disappears; draw it heavier to
        # fade to the solid one's weight.
        solid = plot_style.method_style(m)['linestyle'] == '-'
        style = {**plot_style.method_style(m), 'alpha': faint if solid else 2 * faint,
                 'linewidth': 1.0 if solid else 1.3}
        for i in range(n_a):
            ax.plot(x[i, :, 0], x[i, :, 1], color=plot_style.agent_color(i), **style)

    if xlim is not None and ylim is not None:
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
    else:
        stacked = np.concatenate([x_nom] + x_others, axis=0)
        margin = 0.5
        ax.set_xlim(stacked[:, :, 0].min() - margin, stacked[:, :, 0].max() + margin)
        ax.set_ylim(stacked[:, :, 1].min() - margin, stacked[:, :, 1].max() + margin)

    # One (trajectory, line, point) per animated method and agent.
    tracks = []
    for m in methods:
        for i in range(n_a):
            c = plot_style.agent_color(i)
            line, = ax.plot([], [], color=c, **plot_style.method_style(m))
            point, = ax.plot([], [], markers[m % len(markers)], color=c, markersize=6,
                             markeredgecolor='black', markeredgewidth=0.5)
            tracks.append((x_all[m][i], line, point))
    for i in range(n_a):
        ax.scatter(xf[i, 0], xf[i, 1], marker='*', s=50, color=plot_style.agent_color(i),
                   edgecolors='black', linewidths=0.3, alpha=0.6, zorder=4)
    lines = [line for _, line, _ in tracks]
    points = [point for _, _, point in tracks]
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=11)

    if show_legend:
        handles = []
        for m in range(len(x_all)):
            dash = plot_style.method_style(m)['linestyle']
            if m in methods:
                handles.append(Line2D([0], [0], color='gray', linestyle=dash,
                                      linewidth=plot_style.method_style(m)['linewidth'],
                                      marker=markers[m % len(markers)], markersize=5,
                                      markeredgecolor='black', markeredgewidth=0.5))
            else:
                # Grey at the lines' own alpha all but vanishes in a short handle.
                handles.append(Line2D([0], [0], color='gray', linestyle=dash,
                                      linewidth=1.0, alpha=0.35))
        labels = list(method_labels[:len(x_all)])
        # At the style's legend size, three marker handles overrun the plot box
        # and wrap into a second row the strip has no room for; the tick-label
        # size keeps them on one.
        _legend_within_axes(ax, handles, labels, ncol, 1.4, loc='upper center',
                            frameon=False, bbox_to_anchor=(0.5, -LEGEND_GAP / BOX),
                            fontsize=plt.rcParams['font.size'],
                            columnspacing=LEGEND_COLSPACE_PLAIN, handletextpad=0.4,
                            borderaxespad=0.0)

    def init():
        """Clear every artist before the first frame."""
        for line, point in zip(lines, points):
            line.set_data([], [])
            point.set_data([], [])
        time_text.set_text('')
        return lines + points + [time_text]

    def animate(frame):
        """Draw the path up to one timestep."""
        for x, line, point in tracks:
            line.set_data(x[:frame + 1, 0], x[:frame + 1, 1])
            point.set_data([x[frame, 0]], [x[frame, 1]])
        time_text.set_text(f't = {frame * dt:.3f}s')
        return lines + points + [time_text]

    anim = FuncAnimation(fig, animate, init_func=init, frames=H + 1,
                         interval=100, blit=True)
    if save_path is not None:
        anim.save(save_path, writer='pillow', fps=fps, dpi=150)
    return fig, anim


# ---------------------------------------------------------------------------
# Separation: nominal vs worst case
# ---------------------------------------------------------------------------
def pairwise_distances(x, pdim):
    """Nominal separation of every unordered pair, over the horizon.

    Parameters
    ----------
    x : ndarray, shape (n_a, H+1, sdim)
        State trajectory of every agent.
    pdim : int
        Number of leading state components treated as position.

    Returns
    -------
    dists : ndarray, shape (n_pairs, H+1)
    pairs : list of (i, j)
    """
    pairs = list(combinations(range(x.shape[0]), 2))
    dists = np.array([np.linalg.norm(x[i, :, :pdim] - x[j, :, :pdim], axis=1)
                      for i, j in pairs])
    return dists, pairs


def worst_case_distances(x, A, B, H, sdim, pdim, eps):
    """Separation of every pair under the worst-case perturbation, over the horizon.

    Uses the same oracle the robust solver optimizes against
    (`adversary.compute_selected_worst`), so a figure drawn with this shows the
    quantity that was actually minimized rather than an approximation of it.

    Parameters
    ----------
    x : ndarray, shape (n_a, H+1, sdim)
        State trajectory of every agent.
    A, B : ndarray
        The dynamics the trajectory was planned under.
    H, sdim, pdim : int
        Horizon length, per-agent state dimension, number of position
        components.
    eps : float
        Adversary's per-step energy budget.

    Returns
    -------
    dists : ndarray, shape (n_pairs, H+1)
    pairs : list of (i, j)
    """
    C = selection_matrix(pdim, sdim)
    selected = precompute_selected(A, B, C, H)
    pairs = list(combinations(range(x.shape[0]), 2))
    dists = np.array([
        np.linalg.norm(worst_case_relative(x[i], x[j], C, selected, eps, H)[0], axis=1)
        for i, j in pairs])
    return dists, pairs


def plot_avg_distances(results, labels, n_a, H, sdim, cdim, pdim, eps, A, B, dt,
                       ax=None, show_nominal=True, save_path=None):
    """Mean pairwise separation over time: as flown, and under the worst case.

    The gap between a method's two curves is what the adversary can take away.
    The robust solve is the one whose worst-case curve stays up -- matching its
    own nominal curve is not the goal, and a nominal curve sitting *lower* than
    the baseline's is not a failure as long as the worst case is higher.

    Parameters
    ----------
    results : list of OptimizeResult
        The methods to compare, drawn in palette order.
    labels : list of str
        One per entry in `results`.
    n_a, H, sdim, cdim, pdim : int
        Number of agents, horizon length, state, control and position
        dimensions.
    eps : float
        Adversary's per-step energy budget.
    A, B : ndarray
        The dynamics the trajectories were planned under.
    dt : float
        Timestep, used for the time axis.
    ax : matplotlib Axes, optional
        Draw into an existing axis instead of making a figure.
    show_nominal : bool
        Also draw the as-flown separation.  False plots worst cases only, which
        is the comparison the paper actually rests on.
    save_path : str or pathlib.Path, optional
        Write the figure here.  Nothing is written when None.

    Returns
    -------
    ax : matplotlib Axes
    """
    plot_style.apply()
    t = np.arange(H + 1) * dt

    own_fig = ax is None
    fig, ax = (plt.subplots(figsize=(PANEL * 1.4, PANEL))
               if own_fig else (ax.figure, ax))

    for k, res in enumerate(results):
        x = split_solution(res, n_a, H, sdim, cdim)[0]
        color = plot_style.agent_color(k)
        if show_nominal:
            ax.plot(t, pairwise_distances(x, pdim)[0].mean(axis=0), '-o',
                    color=color, markersize=3,
                    **{k_: v for k_, v in plot_style.method_style(0).items()
                       if k_ != 'linestyle'})
        wc = worst_case_distances(x, A, B, H, sdim, pdim, eps)[0].mean(axis=0)
        ax.plot(t, wc, color=color, marker='^', markersize=3, alpha=0.75,
                **{k_: v for k_, v in plot_style.method_style(1).items()
                   if k_ != 'alpha'})

    ax.set_xlabel('time (s)')
    ax.set_ylabel('mean pairwise separation')
    ax.grid(True, linewidth=0.3, alpha=0.3)
    ax.tick_params(direction='in', top=True, right=True)
    if own_fig:
        fig.tight_layout()
    planned = dict(linestyle='-', marker='o', markersize=3, linewidth=1.8)
    worst = dict(linestyle=plot_style.method_style(1)['linestyle'], marker='^',
                 markersize=3, linewidth=1.5)
    styles = ([(planned, 'as planned')] if show_nominal else []) + [(worst, 'worst case')]
    _method_style_legend(ax, labels, styles, widen=own_fig)
    _save(fig, save_path)
    return ax


def min_separation_table(results, labels, n_a, H, sdim, cdim, pdim, eps, A, B):
    """Closest approach of any pair, as flown and under the worst case.

    The single number the safety claim rests on: what the worst case does to the
    tightest pair.  Mean separation is a weak substitute -- averaging over pairs
    lets slack in most of them cover a tight one.

    Parameters
    ----------
    results : list of OptimizeResult
        The methods to compare.
    labels : list of str
        One per entry in `results`; becomes the `method` field.
    n_a, H, sdim, cdim, pdim : int
        Number of agents, horizon length, state, control and position
        dimensions.
    eps : float
        Adversary's per-step energy budget.
    A, B : ndarray
        The dynamics the trajectories were planned under.

    Returns
    -------
    rows : list of dict
        One row per method, with keys `method`, `min separation` and
        `min separation, worst case`, each a minimum over all pairs and all
        steps.  Plain dicts, so the caller may render them with pandas or print
        them directly.
    """
    rows = []
    for res, label in zip(results, labels):
        x = split_solution(res, n_a, H, sdim, cdim)[0]
        nominal = pairwise_distances(x, pdim)[0].min()
        worst = worst_case_distances(x, A, B, H, sdim, pdim, eps)[0].min()
        rows.append({'method': label,
                     'min separation': float(nominal),
                     'min separation, worst case': float(worst)})
    return rows


def plot_relative_trajectory(results, labels, n_a, H, sdim, cdim, pdim, eps, A, B,
                             pair=(0, 1), ax=None, save_path=None):
    """One pair's relative state z = x_i - x_j, nominal and worst case.

    The clearest single view of what strategic robustness buys: the origin is a
    collision, so a curve that bends away from it is separation.  Solid curves
    are the trajectories as flown; faint curves are where the adversary can drag
    them, one point per prefix length.

    Parameters
    ----------
    results : list of OptimizeResult
        The methods to compare, drawn in palette order.
    labels : list of str
        One per entry in `results`.
    n_a, H, sdim, cdim, pdim : int
        Number of agents, horizon length, state, control and position
        dimensions.
    eps : float
        Adversary's per-step energy budget.
    A, B : ndarray
        The dynamics the trajectories were planned under.
    pair : (int, int)
        Which agents' relative state to draw.
    ax : matplotlib Axes, optional
        Draw into an existing axis instead of making a figure.
    save_path : str or pathlib.Path, optional
        Write the figure here.  Nothing is written when None.

    Returns
    -------
    ax : matplotlib Axes
    """
    plot_style.apply()
    i, j = pair
    C = selection_matrix(pdim, sdim)
    selected = precompute_selected(A, B, C, H)

    own_fig = ax is None
    fig, ax = plt.subplots(figsize=(PANEL, PANEL)) if own_fig else (ax.figure, ax)

    for k, res in enumerate(results):
        x = split_solution(res, n_a, H, sdim, cdim)[0]
        z = x[i, :, :pdim] - x[j, :, :pdim]
        zw = worst_case_relative(x[i], x[j], C, selected, eps, H)[0]
        color = plot_style.agent_color(k)
        worst_style = {**plot_style.method_style(1), 'alpha': 0.55}
        ax.plot(z[:, 0], z[:, 1], color=color, **plot_style.method_style(0))
        ax.plot(zw[:, 0], zw[:, 1], color=color, **worst_style)

    ax.scatter([0], [0], marker='x', s=60, color='black', zorder=6, linewidths=1.2)
    # Below-left of the marker: the origin often sits at the top edge of the
    # data, where a label above it runs into the frame, and the worst-case
    # curves tend to finish just to its right.
    ax.annotate('collision', (0, 0), textcoords='offset points', xytext=(-6, -6),
                ha='right', va='top')
    ax.set_xlabel(r'$z_x$')
    ax.set_ylabel(r'$z_y$')
    ax.set_box_aspect(1)
    ax.grid(True, linewidth=0.3, alpha=0.3)
    ax.tick_params(direction='in', top=True, right=True)
    if own_fig:
        fig.tight_layout()
    styles = [(plot_style.method_style(0), 'as planned'),
              ({**plot_style.method_style(1), 'alpha': 0.55}, 'worst case')]
    _method_style_legend(ax, labels, styles, widen=own_fig)
    _save(fig, save_path)
    return ax
