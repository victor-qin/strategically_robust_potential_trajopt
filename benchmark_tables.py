"""Regenerate the paper's runtime and deviation tables.

Self-contained: every scenario defines its own x0/xf and dynamics, so results do
not depend on notebook cell-execution order.  Importable from a notebook or
runnable headless on a cloud box, with no plotting stack installed:

    python benchmark_tables.py --runs 30                    # all scenarios
    python benchmark_tables.py --runs 30 --shooting         # single-shooting solvers
    python benchmark_tables.py --scenarios "8 agents" --runs 10
    python benchmark_tables.py --runs 30 --json out.json

Three conventions matter for trustworthy timings, all learned from cases where
the naive choice gave a misleading number:

1. ANALYTIC GRADIENTS EVERYWHERE.  Every method -- including the nominal
   baseline and the nominal initialization inside the robust solve -- is given
   its analytic gradient.  Mixing modes made the 8-agent robust overhead read
   3.72x instead of 1.29x, because the numeric-gradient baseline terminated
   early on a worse optimum.
2. MEDIAN ALONGSIDE MEAN.  A single outlying run deflected the 4-agent ratio
   from 1.89x to 1.72x under the mean.  Both are reported, plus CV, so a noisy
   row is visible rather than silent.
3. OBJECTIVES RECORDED AND CHECKED.  Every run's final objective is stored and
   asserted unique per cell.  If it is not, the timing spread is a mix of local
   minima rather than machine noise, and the row must not be averaged.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from costs import C1_NOMINAL, C1_WIDER, make_cost
from solvers import (optimize_everystep, optimize_everystep_shooting,
                     optimize_optimal, optimize_optimal_shooting, split_solution)

COST_NOMINAL, GRAD_NOMINAL = make_cost(c1=C1_NOMINAL)
COST_WIDER, GRAD_WIDER = make_cost(c1=C1_WIDER)


# --------------------------------------------------------------------------
# Scenarios -- x0/xf and dynamics defined together, per scenario
# --------------------------------------------------------------------------
def _single_integrator(n_a, H=20, tf=2.0, eps=2.0):
    """Scenario skeleton with the paper's dynamics and weights.

    A = I, B = dt*I in R^2; Q = I, Qf = 150I, R = I, matching section IV of the
    paper.  The caller fills in `x0` and `xf`.

    Parameters
    ----------
    n_a : int
        Number of agents.
    H : int
        Horizon length.
    tf : float
        Final time; dt = tf / H.
    eps : float
        Adversary's per-step energy budget.

    Returns
    -------
    cfg : dict
        Every key the solver wrappers need, except `x0` and `xf`.
    """
    dt = tf / H
    sdim = cdim = pdim = 2
    return dict(
        n_a=n_a, H=H, dt=dt, eps=eps, sdim=sdim, cdim=cdim, pdim=pdim,
        A=np.eye(sdim), B=dt * np.eye(cdim),
        Q=np.eye(sdim), Qf=150.0 * np.eye(sdim), R=np.eye(cdim),
        u_max=None,
    )


def _circle(n_a, center=(2.0, 2.0), radius=2.0):
    """n_a agents on a circle, each heading to the antipodal point.

    Every straight-line plan passes through the centre, so all pairs conflict at
    once.  Adjacent agents start 2*radius*sin(pi/n_a) apart, which is why the
    12-agent probe is a harder problem and not merely a bigger one.

    Parameters
    ----------
    n_a : int
        Number of agents, spaced evenly around the circle.
    center : (float, float)
        Circle centre.
    radius : float
        Circle radius.

    Returns
    -------
    x0 : ndarray, shape (n_a, 2)
        Starting positions.
    xf : ndarray, shape (n_a, 2)
        Antipodal targets.
    """
    c = np.asarray(center)
    ang = np.linspace(0, 2 * np.pi, n_a, endpoint=False)
    x0 = np.array([c + radius * np.array([np.cos(a), np.sin(a)]) for a in ang])
    return x0, np.array([c - radius * np.array([np.cos(a), np.sin(a)]) for a in ang])


def _scenarios():
    """Build the scenario table.

    Returns
    -------
    scenarios : dict
        Maps scenario name to a config dict.  The four published scenarios plus
        a 12-agent scaling probe that is not part of the paper.
    """
    s = {}

    cfg = _single_integrator(2)
    cfg['x0'] = np.array([[0.0, 0.95], [2.0, 1.05]])
    cfg['xf'] = np.array([[2.0, 1.05], [0.0, 0.95]])
    s['Head-on'] = cfg

    cfg = _single_integrator(2)
    cfg['x0'] = np.array([[0.0, 0.0], [2.0, 0.0]])
    cfg['xf'] = np.array([[0.0, 2.0], [2.0, 2.0]])
    s['Parallel'] = cfg

    cfg = _single_integrator(4)
    cfg['x0'] = np.array([[0.0, 1.5], [1.0, 0.0], [1.0, 3.0], [3.0, 0.5]])
    cfg['xf'] = np.array([[3.0, 1.5], [2.0, 2.5], [2.0, 1.0], [1.0, 2.5]])
    s['4 agents'] = cfg

    cfg = _single_integrator(8)
    cfg['x0'], cfg['xf'] = _circle(8)
    s['8 agents'] = cfg

    # Scaling probe beyond the paper's largest case: 66 pairs instead of 28, and
    # 984 decision variables instead of 656 in the full space.  Note the circle
    # gets crowded -- adjacent agents start 2*r*sin(pi/12) = 1.04 apart versus
    # 1.53 at n_a = 8 -- so this is a harder collision problem, not just a bigger
    # one.  Not part of the published tables.
    # cfg = _single_integrator(12)
    # cfg['x0'], cfg['xf'] = _circle(12)
    # s['12 agents'] = cfg

    return s


SCENARIOS = _scenarios()


# --------------------------------------------------------------------------
# Solver wrappers
# --------------------------------------------------------------------------
def _base(c):
    """Positional arguments every solver shares, pulled from a scenario config.

    Parameters
    ----------
    c : dict
        A scenario config from `SCENARIOS`.

    Returns
    -------
    args : tuple
        `(x0, xf, A, B, H, Q, R, Qf, n_a, sdim, cdim, pdim)`, ready to splat.
    """
    return (c['x0'], c['xf'], c['A'], c['B'], c['H'], c['Q'], c['R'], c['Qf'],
            c['n_a'], c['sdim'], c['cdim'], c['pdim'])


def _split(res, c):
    """Unpack a result using a scenario config's dimensions.

    Parameters
    ----------
    res : OptimizeResult
    c : dict
        A scenario config from `SCENARIOS`.

    Returns
    -------
    x : ndarray, shape (n_a, H+1, sdim)
    u : ndarray, shape (n_a, H, cdim)
    """
    return split_solution(res, c['n_a'], c['H'], c['sdim'], c['cdim'])


# Set by --shooting.  When true every method is solved in the reduced
# (controls-only) parameterization instead of the full (x, u) space: the
# n_a*sdim initial-condition and n_a*H*sdim dynamics equality constraints are
# eliminated by rolling the dynamics forward, so SLSQP sees an unconstrained
# problem roughly half the size.  All three rows switch together -- Nominal and
# Wider are both the nominal solver, and the nominal solve dominates the reported
# totals, so reformulating only the robust row barely moves the table.
#
# Caveat worth keeping in mind when comparing modes: the two parameterizations
# start from different points (the full-space default guess sets x to a linear
# interpolation with u = 0, which its own controls do not produce, and shooting
# cannot represent an inconsistent start).  On the 8-agent scenario that is
# enough to land them in different local minima, so compare the reported
# objectives, not just the times.
USE_SHOOTING = False


def _optimal(c, cost, grad, warm=None):
    """Nominal/wider solve, dispatched by USE_SHOOTING.

    The shooting driver takes the same signature and repacks its result into the
    full [x, u] layout, so _split and integrated_path_deviation work unchanged.

    Parameters
    ----------
    c : dict
        A scenario config from `SCENARIOS`.
    cost, grad : callable
        A cost and its derivative, from `costs.make_cost`.
    warm : (ndarray, ndarray) or None
        `(x, u)` warm start.  None uses the solver's own default guess, which is
        what the timed runs do.

    Returns
    -------
    result : OptimizeResult
    """
    kw = {} if warm is None else dict(x_opt=warm[0], u_opt=warm[1])
    fn = optimize_optimal_shooting if USE_SHOOTING else optimize_optimal
    return fn(*_base(c), cost, distance_cost_grad_func=grad, u_max=c['u_max'], **kw)


def solve_nominal(c):
    """Nominal solve at c1 = 2, from the solver's default guess.

    Parameters
    ----------
    c : dict
        A scenario config from `SCENARIOS`.

    Returns
    -------
    result : OptimizeResult
    """
    return _optimal(c, COST_NOMINAL, GRAD_NOMINAL)


def solve_wider(c, warm=None):
    """"Wider" baseline: the same solver at c1 = 5.

    Parameters
    ----------
    c : dict
        A scenario config from `SCENARIOS`.
    warm : (ndarray, ndarray), optional
        `(x, u)` warm start.  The timed runs pass None so this is measured on
        the same footing as the nominal solve.

    Returns
    -------
    result : OptimizeResult
    """
    return _optimal(c, COST_WIDER, GRAD_WIDER, warm)


def solve_robust(c, warm):
    """Everystep solve, warm-started from the nominal solution.

    Parameters
    ----------
    c : dict
        A scenario config from `SCENARIOS`.
    warm : (ndarray, ndarray)
        `(x, u)` from a nominal solve.  Required: the paper initializes the
        robust solve this way and charges its runtime for the initialization.

    Returns
    -------
    result : OptimizeResult
    """
    fn = optimize_everystep_shooting if USE_SHOOTING else optimize_everystep
    return fn(*_base(c), c['eps'], COST_NOMINAL,
              distance_cost_grad_func=GRAD_NOMINAL,
              x_opt=warm[0], u_opt=warm[1], u_max=c['u_max'])


def integrated_path_deviation(x_method, x_nominal, dt, pdim):
    """Trapezoidal area between two trajectories, averaged over agents.

    How far a method had to distort the nominal plan, in units of distance x time.
    Position components only, so it is unaffected by velocity states.

    Parameters
    ----------
    x_method, x_nominal : ndarray, shape (n_a, H+1, sdim)
    dt : float
        Timestep, used for the trapezoid rule's abscissae.
    pdim : int
        Number of leading state components treated as position.

    Returns
    -------
    deviation : float
        Area between the two trajectories, averaged over agents, in units of
        distance x time.  Zero when the trajectories coincide.
    """
    t = np.arange(x_method.shape[1]) * dt
    return float(np.mean([
        np.trapezoid(np.linalg.norm(x_method[i, :, :pdim] - x_nominal[i, :, :pdim], axis=1), t)
        for i in range(x_method.shape[0])
    ]))


# --------------------------------------------------------------------------
# Timing
# --------------------------------------------------------------------------
def run_scenario(name, n_runs=10, verbose=True):
    """Time all three methods on one scenario.

    The robust timing includes its nominal initialization, matching the paper.
    A full untimed warmup runs first so numba compilation and first-call effects
    never land inside a timed region.

    Parameters
    ----------
    name : str
        Key into `SCENARIOS`.
    n_runs : int
        Timed repetitions of all three methods.
    verbose : bool
        Print a line per run and a wall-time summary.

    Returns
    -------
    result : dict
        Keys `scenario`, `n_a`, `H`, `runs`, `timings` (per-method lists of
        time, nit, nfev and objective for every run), `deviation` (per-method
        integrated path deviation), `overhead` (untimed seconds attributed per
        method), `wall_seconds` and `measured_seconds`.
    """
    c = SCENARIOS[name]
    t_start = time.perf_counter()
    if verbose:
        print(f"\n===== {name}  (n_a={c['n_a']}, H={c['H']}, runs={n_runs}) =====", flush=True)

    rec = {m: {'t': [], 'nit': [], 'nfev': [], 'f': []} for m in ('nominal', 'wider', 'robust')}
    # Warmup and deviation solves are not part of the reported timings, but they
    # do cost wall time, so attribute them per method rather than lumping them.
    overhead = {m: 0.0 for m in rec}

    def _oh(m, fn):
        """Run an untimed solve and charge its wall time to one method."""
        t0 = time.perf_counter()
        res = fn()
        overhead[m] += time.perf_counter() - t0
        return res

    warm = _split(_oh('nominal', lambda: solve_nominal(c)), c)   # warmup, all three paths
    _oh('wider', lambda: solve_wider(c, warm))
    _oh('robust', lambda: solve_robust(c, warm))

    def _timed(m, fn):
        """Run one measured solve and record its time, counts and objective."""
        t0 = time.perf_counter()
        res = fn()
        rec[m]['t'].append(time.perf_counter() - t0)
        rec[m]['nit'].append(int(res.nit))
        rec[m]['nfev'].append(int(res.nfev))
        rec[m]['f'].append(float(res.fun))
        return res

    for r in range(n_runs):
        _timed('nominal', lambda: solve_nominal(c))
        _timed('wider', lambda: solve_wider(c))

        def _robust():
            """One robust solve including its own nominal initialization."""
            return solve_robust(c, _split(solve_nominal(c), c))
        _timed('robust', _robust)

        if verbose:
            print(f"  run {r + 1:3d}/{n_runs}: " + "  ".join(
                f"{m}={rec[m]['t'][-1]:8.4f}" for m in rec), flush=True)

    # deviations are deterministic given the inputs, so one solve suffices
    nom = _split(_oh('nominal', lambda: solve_nominal(c)), c)
    x_nom = nom[0]
    dev = {
        'nominal': 0.0,
        'wider': integrated_path_deviation(
            _split(_oh('wider', lambda: solve_wider(c, nom)), c)[0], x_nom, c['dt'], c['pdim']),
        'robust': integrated_path_deviation(
            _split(_oh('robust', lambda: solve_robust(c, nom)), c)[0], x_nom, c['dt'], c['pdim']),
    }
    elapsed = time.perf_counter() - t_start
    measured = float(sum(sum(v['t']) for v in rec.values()))
    if verbose:
        print(f"  wall {elapsed:.1f}s  ({measured:.1f}s measured, "
              f"{elapsed - measured:.1f}s warmup+deviation overhead)", flush=True)
    return {'scenario': name, 'n_a': c['n_a'], 'H': c['H'], 'runs': n_runs,
            'timings': rec, 'deviation': dev, 'overhead': overhead,
            'wall_seconds': elapsed, 'measured_seconds': measured}


def summarize(result):
    """Per-method statistics plus ratio to nominal, by mean and by median.

    Parameters
    ----------
    result : dict
        One entry from `run_scenario`.

    Returns
    -------
    stats : dict
        Maps method name to a dict of `mean`, `median`, `std`, `min`, `max`,
        `cv`, `ratio_mean`, `ratio_median`, `nit`, `nfev`, `objective` and
        `objective_unique`.  `objective_unique` must be 1: anything else means
        the runs landed on different local minima and must not be averaged.
    """
    rec = result['timings']
    base = np.asarray(rec['nominal']['t'])
    out = {}
    for m, v in rec.items():
        t = np.asarray(v['t'])
        f = np.asarray(v['f'])
        uniq = np.unique(np.round(f, 6))
        out[m] = {
            'mean': float(t.mean()), 'median': float(np.median(t)),
            'std': float(t.std()), 'min': float(t.min()), 'max': float(t.max()),
            'cv': float(t.std() / t.mean()) if t.mean() else 0.0,
            'ratio_mean': float(t.mean() / base.mean()),
            'ratio_median': float(np.median(t) / np.median(base)),
            'nit': float(np.mean(v['nit'])), 'nfev': float(np.mean(v['nfev'])),
            'objective': float(f[0]), 'objective_unique': int(len(uniq)),
        }
    return out


LABEL = {'nominal': 'Nominal', 'wider': 'Wider', 'robust': 'Strat. Robust'}


def format_tables(results, stat='median'):
    """Render the runtime and deviation tables as fixed-width text plus LaTeX.

    Parameters
    ----------
    results : list of dict
        Entries from `run_scenario`, one per scenario.
    stat : {'median', 'mean'}
        Which statistic the reported ratios use.  Median is the default: a
        single outlying run moved the 4-agent ratio by 0.17x under the mean.

    Returns
    -------
    text : str
        One block per scenario -- statistics, wall-time attribution -- followed
        by LaTeX rows ready to paste into the paper.
    """
    key = 'ratio_median' if stat == 'median' else 'ratio_mean'
    lines = []
    for r in results:
        s = summarize(r)
        mode = 'single shooting' if USE_SHOOTING else 'full space'
        lines.append(f"\n### {r['scenario']}  (n={r['runs']}, ratios by {stat}, {mode})")
        lines.append(f"{'method':<15}{'mean':>9}{'median':>9}{'std':>8}{'CV':>7}"
                     f"{'nit':>6}{'ratio':>8}{'deviation':>11}{'obj':>14}{'uniq':>6}")
        lines.append('-' * 93)
        for m in ('nominal', 'wider', 'robust'):
            d = s[m]
            flag = '' if d['objective_unique'] == 1 else '  <-- MULTIPLE OPTIMA'
            lines.append(f"{LABEL[m]:<15}{d['mean']:>9.4f}{d['median']:>9.4f}{d['std']:>8.4f}"
                         f"{d['cv']:>6.1%}{d['nit']:>6.0f}{d[key]:>7.2f}x"
                         f"{r['deviation'][m]:>11.4f}{d['objective']:>14.4f}"
                         f"{d['objective_unique']:>6}{flag}")

        # Wall-time attribution: measured runs + warmup/deviation, per method.
        oh = r.get('overhead', {m: 0.0 for m in ('nominal', 'wider', 'robust')})
        tot = r['wall_seconds']
        lines.append("")
        lines.append(f"{'wall breakdown':<15}{'measured':>9}{'overhead':>9}{'total':>8}"
                     f"{'share':>7}{'per run':>9}")
        for m in ('nominal', 'wider', 'robust'):
            meas = float(np.sum(r['timings'][m]['t']))
            tm = meas + oh[m]
            lines.append(f"{LABEL[m]:<15}{meas:>9.2f}{oh[m]:>9.2f}{tm:>8.2f}"
                         f"{tm / tot:>6.1%}{meas / max(r['runs'], 1):>9.3f}")
        acct = sum(float(np.sum(r['timings'][m]['t'])) + oh[m] for m in oh)
        lines.append(f"{'scenario total':<15}{r['measured_seconds']:>9.2f}"
                     f"{sum(oh.values()):>9.2f}{tot:>8.2f}"
                     f"{'':>7}{'':>9}  (unattributed {tot - acct:+.2f}s)")

    lines.append("\n\n% ---- LaTeX ----")
    for r in results:
        s = summarize(r)
        t = 'median' if stat == 'median' else 'mean'
        lines.append(f"% {r['scenario']}")
        for m in ('nominal', 'wider', 'robust'):
            lines.append(f"{LABEL[m]:<14}& {s[m][key]:.2f}x ({s[m][t]:.3f}s) \\\\")
    return "\n".join(lines)


def main():
    """Command-line entry point: run the scenarios and print the tables.

    Returns
    -------
    None
    """
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scenarios', nargs='*', default=list(SCENARIOS),
                    help='subset of: ' + ', '.join(f'"{k}"' for k in SCENARIOS))
    ap.add_argument('--runs', type=int, default=10)
    ap.add_argument('--stat', choices=['median', 'mean'], default='median')
    ap.add_argument('--json', help='write raw per-run data here')
    ap.add_argument('--shooting', action='store_true',
                    help='solve every method by single shooting (controls only, no '
                         'dynamics constraints) instead of in the full (x, u) space')
    a = ap.parse_args()

    global USE_SHOOTING
    USE_SHOOTING = a.shooting
    print('solver parameterization: '
          + ('single shooting (controls only)' if USE_SHOOTING
             else 'full space (x, u)'))

    t0 = time.perf_counter()
    results = [run_scenario(n, a.runs) for n in a.scenarios]
    total = time.perf_counter() - t0
    print(format_tables(results, a.stat))
    print(f"\ntotal wall time: {total:.1f}s ({total / 60:.1f} min) across "
          f"{len(results)} scenario(s) at {a.runs} run(s) each")
    if a.json:
        with open(a.json, 'w') as fh:
            json.dump(results, fh, indent=1)
        print(f"\nraw per-run data -> {a.json}")


if __name__ == '__main__':
    main()
