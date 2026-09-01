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
import numpy as np
from itertools import combinations

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D

import plot_style
from adversary import precompute_selected, selection_matrix, worst_case_relative
from solvers import split_solution

PANEL = 3.5           # inches; one column of a two-column page
LEGEND_ROW = 0.22     # height of one legend row, inches
LEGEND_PAD = 0.30     # padding below the axis before the legend starts


def _legend_shape(n_entries):
    """Columns and reserved height for a figure-level legend of `n_entries`.

    The strip has to grow with the row count: a fixed reservation that fits a
    2-agent legend puts an 8-agent one on top of the x tick labels.

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
    return ncol, LEGEND_PAD + LEGEND_ROW * nrow


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


def _save(fig, save_path):
    """Write a figure to disk, or do nothing when no path was requested.

    Parameters
    ----------
    fig : matplotlib Figure
    save_path : str, pathlib.Path or None
        Destination.  None is the default everywhere, so no helper writes a file
        as a side effect.

    Returns
    -------
    None
    """
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.02)


# ---------------------------------------------------------------------------
# Trajectories in the plane
# ---------------------------------------------------------------------------
def plot_trajectories(result_nominal, result_others, n_a, H, sdim, cdim, xf,
                      method_labels=None, ax=None, title=None, xlim=None,
                      ylim=None, show_legend=True, save_path=None):
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
        Draw a legend below the axis, sized to its own row count.
    save_path : str or pathlib.Path, optional
        Write the figure here.  Nothing is written when None.

    Returns
    -------
    ax : matplotlib Axes
    """
    plot_style.apply()
    x_nom = split_solution(result_nominal, n_a, H, sdim, cdim)[0]
    x_others = _trajectories(result_others, n_a, H, sdim, cdim)
    n_methods = 1 + len(x_others)
    if method_labels is None:
        method_labels = ['nominal'] + [f'method {j + 1}' for j in range(len(x_others))]

    ncol, strip = _legend_shape(n_a + n_methods)
    own_fig = ax is None
    if own_fig:
        extra = strip if show_legend else 0.0
        fig, ax = plt.subplots(figsize=(PANEL, PANEL + extra))
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
        ax.set_title(title, fontsize=9, pad=4)

    if show_legend and own_fig:
        fig.tight_layout(rect=[0, strip / (PANEL + strip), 1, 1])
        handles = [Line2D([0], [0], color=plot_style.agent_color(i), lw=1.5,
                          linestyle='-', label=f'Agent {i + 1}') for i in range(n_a)]
        handles += [Line2D([0], [0], color='gray', label=method_labels[m],
                           **{k: v for k, v in plot_style.method_style(m).items()
                              if k in ('linestyle', 'linewidth')})
                    for m in range(n_methods)]
        fig.legend(handles=handles, loc='lower center', ncol=ncol, frameon=False,
                   bbox_to_anchor=(0.52, 0.01), columnspacing=1.0,
                   handlelength=2.0, fontsize=9)
    elif own_fig:
        fig.tight_layout()

    _save(fig, save_path)
    return ax


def animate_trajectories(result_nominal, result_others, n_a, H, sdim, cdim, xf, dt,
                         which=0, xlim=None, ylim=None, save_path=None, fps=5):
    """Animate one method's motion against the faint nominal paths.

    Parameters
    ----------
    result_nominal : OptimizeResult
        Drawn as faint static paths for context, whichever method is animated.
    result_others : list of OptimizeResult
        Candidates to animate; `which` selects one.
    n_a, H, sdim, cdim : int
        Number of agents, horizon length, state and control dimensions.
    xf : ndarray, shape (n_a, sdim)
        Targets, marked with stars.
    dt : float
        Timestep, used for the on-figure clock.
    which : int
        Index into `result_others` to animate; -1 animates the nominal itself,
        as does an empty `result_others`.
    xlim, ylim : (float, float), optional
        Axis limits.  Both must be given to take effect; otherwise the limits
        are taken from the data with a 0.5 margin.
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
    x_anim = x_nom if which < 0 or not x_others else x_others[which]

    fig, ax = plt.subplots(figsize=(PANEL, PANEL))
    ax.set_box_aspect(1)
    ax.grid(True, linewidth=0.3, alpha=0.3)
    ax.tick_params(direction='in', top=True, right=True)

    for i in range(n_a):                       # faint nominal paths for context
        ax.plot(x_nom[i, :, 0], x_nom[i, :, 1],
                color=plot_style.agent_color(i), alpha=0.15, lw=1.0)

    if xlim is not None and ylim is not None:
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
    else:
        stacked = np.concatenate([x_nom] + x_others, axis=0)
        margin = 0.5
        ax.set_xlim(stacked[:, :, 0].min() - margin, stacked[:, :, 0].max() + margin)
        ax.set_ylim(stacked[:, :, 1].min() - margin, stacked[:, :, 1].max() + margin)

    lines, points = [], []
    for i in range(n_a):
        c = plot_style.agent_color(i)
        line, = ax.plot([], [], color=c, **plot_style.method_style(0))
        point, = ax.plot([], [], 'o', color=c, markersize=6,
                         markeredgecolor='black', markeredgewidth=0.5)
        lines.append(line)
        points.append(point)
        ax.scatter(xf[i, 0], xf[i, 1], marker='*', s=50, color=c,
                   edgecolors='black', linewidths=0.3, alpha=0.6, zorder=4)
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=9)

    def init():
        """Clear every artist before the first frame."""
        for line, point in zip(lines, points):
            line.set_data([], [])
            point.set_data([], [])
        time_text.set_text('')
        return lines + points + [time_text]

    def animate(frame):
        """Draw the path up to one timestep."""
        for i in range(n_a):
            lines[i].set_data(x_anim[i, :frame + 1, 0], x_anim[i, :frame + 1, 1])
            points[i].set_data([x_anim[i, frame, 0]], [x_anim[i, frame, 1]])
        time_text.set_text(f't = {frame * dt:.3f}s')
        return lines + points + [time_text]

    fig.tight_layout()
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

    for k, (res, label) in enumerate(zip(results, labels)):
        x = split_solution(res, n_a, H, sdim, cdim)[0]
        color = plot_style.agent_color(k)
        if show_nominal:
            ax.plot(t, pairwise_distances(x, pdim)[0].mean(axis=0), '-o',
                    color=color, markersize=3, label=label,
                    **{k_: v for k_, v in plot_style.method_style(0).items()
                       if k_ != 'linestyle'})
        wc = worst_case_distances(x, A, B, H, sdim, pdim, eps)[0].mean(axis=0)
        ax.plot(t, wc, color=color, marker='^', markersize=3, alpha=0.75,
                label=f'{label}, worst case',
                **{k_: v for k_, v in plot_style.method_style(1).items()
                   if k_ != 'alpha'})

    ax.set_xlabel('time (s)')
    ax.set_ylabel('mean pairwise separation')
    ax.grid(True, linewidth=0.3, alpha=0.3)
    ax.tick_params(direction='in', top=True, right=True)
    ax.legend(frameon=False, fontsize=8)
    if own_fig:
        fig.tight_layout()
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

    for k, (res, label) in enumerate(zip(results, labels)):
        x = split_solution(res, n_a, H, sdim, cdim)[0]
        z = x[i, :, :pdim] - x[j, :, :pdim]
        zw = worst_case_relative(x[i], x[j], C, selected, eps, H)[0]
        color = plot_style.agent_color(k)
        worst_style = {**plot_style.method_style(1), 'alpha': 0.55}
        ax.plot(z[:, 0], z[:, 1], color=color, label=label, **plot_style.method_style(0))
        ax.plot(zw[:, 0], zw[:, 1], color=color,
                label=f'{label}, worst case', **worst_style)

    ax.scatter([0], [0], marker='x', s=60, color='black', zorder=6, linewidths=1.2)
    ax.annotate('collision', (0, 0), textcoords='offset points', xytext=(6, 6),
                fontsize=8)
    ax.set_xlabel(r'$z_x$')
    ax.set_ylabel(r'$z_y$')
    ax.set_box_aspect(1)
    ax.grid(True, linewidth=0.3, alpha=0.3)
    ax.tick_params(direction='in', top=True, right=True)
    ax.legend(frameon=False, fontsize=8)
    if own_fig:
        fig.tight_layout()
    _save(fig, save_path)
    return ax
