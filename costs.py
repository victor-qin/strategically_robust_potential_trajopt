"""Inter-agent distance costs and their analytic derivatives.

Every cost here is paired with its exact derivative, and `make_cost` hands the
two back together.  That pairing is deliberate: SLSQP will silently fall back to
finite differences if a solver is handed a cost without a gradient, and the
resulting timings are not merely slower but *wrong* -- the numeric-gradient
baseline terminates early on a worse optimum, which made the 8-agent robust
overhead read 3.72x instead of 1.29x.  See the header of `benchmark_tables.py`.

Costs are written as functions of distance `d` (not `d**2`), are negated so that
"more cost" means "closer together", and are applied to every unordered pair at
every timestep.
"""
from functools import partial

import numpy as np

# Parameters used throughout the paper.  Kept here so the notebook, the
# benchmarks and any new script share one definition rather than three copies.
ALPHA = 1e-8        # log-barrier smoothing
C1_NOMINAL = 2.0    # collision weight, nominal and strategically robust solves
C1_WIDER = 5.0      # collision weight, the "just turn the penalty up" baseline


# ---------------------------------------------------------------------------
# Cost / gradient pairs
# ---------------------------------------------------------------------------
def logarithmic(dist, c1, alpha):
    """Log-barrier cost -c1 * log(d^2 + alpha).  Diverges as d -> 0.

    The paper's cost, with c1 = 2 nominal, c1 = 5 for the "wider" baseline, and
    alpha = 1e-8.

    Parameters
    ----------
    dist : float or ndarray
        Pair separation, not squared.  Vectorized: the solvers pass an array of
        shape (n_pairs, H+1) and expect the same shape back.
    c1 : float
        Collision weight.  Larger values push agents further apart.
    alpha : float
        Barrier smoothing.  Keeps the cost finite at d = 0.

    Returns
    -------
    cost : float or ndarray
        Same shape as `dist`.
    """
    return -c1 * np.log(dist ** 2 + alpha)


def logarithmic_grad(dist, c1, alpha):
    """d/dd of `logarithmic`, equal to -2*c1*d / (d^2 + alpha).

    Parameters
    ----------
    dist : float or ndarray
        Pair separation, not squared.  Vectorized: the solvers pass an array of
        shape (n_pairs, H+1) and expect the same shape back.
    c1 : float
        Collision weight.  Larger values push agents further apart.
    alpha : float
        Barrier smoothing.  Keeps the cost finite at d = 0.

    Returns
    -------
    grad : float or ndarray
        Same shape as `dist`.
    """
    return -2 * c1 * dist / (dist ** 2 + alpha)


def quadratic(dist, c1, alpha):
    """Soft cost -c1 * d^2.  No barrier: agents may pass through one another.

    Parameters
    ----------
    dist : float or ndarray
        Pair separation, not squared.  Vectorized: the solvers pass an array of
        shape (n_pairs, H+1) and expect the same shape back.
    c1 : float
        Collision weight.  Larger values push agents further apart.
    alpha : float
        Barrier smoothing.  Keeps the cost finite at d = 0.

    Returns
    -------
    cost : float or ndarray
        Same shape as `dist`.
    """
    return -c1 * dist ** 2


def quadratic_grad(dist, c1, alpha):
    """d/dd of `quadratic`, equal to -2*c1*d.

    `alpha` is accepted and ignored, so every gradient here has one signature.

    Parameters
    ----------
    dist : float or ndarray
        Pair separation, not squared.  Vectorized: the solvers pass an array of
        shape (n_pairs, H+1) and expect the same shape back.
    c1 : float
        Collision weight.  Larger values push agents further apart.
    alpha : float
        Barrier smoothing.  Keeps the cost finite at d = 0.

    Returns
    -------
    grad : float or ndarray
        Same shape as `dist`.
    """
    return -2 * c1 * dist


def cauchy(dist, c1, alpha):
    """Lorentzian cost -c1 * log(1 + (d/alpha)^2).  Softer barrier than log.

    Parameters
    ----------
    dist : float or ndarray
        Pair separation, not squared.  Vectorized: the solvers pass an array of
        shape (n_pairs, H+1) and expect the same shape back.
    c1 : float
        Collision weight.  Larger values push agents further apart.
    alpha : float
        Barrier smoothing.  Keeps the cost finite at d = 0.

    Returns
    -------
    cost : float or ndarray
        Same shape as `dist`.
    """
    return -c1 * np.log(1 + (dist / alpha) ** 2)


def cauchy_grad(dist, c1, alpha):
    """d/dd of `cauchy`, equal to -2*c1*d / (alpha^2 + d^2).

    Parameters
    ----------
    dist : float or ndarray
        Pair separation, not squared.  Vectorized: the solvers pass an array of
        shape (n_pairs, H+1) and expect the same shape back.
    c1 : float
        Collision weight.  Larger values push agents further apart.
    alpha : float
        Barrier smoothing.  Keeps the cost finite at d = 0.

    Returns
    -------
    grad : float or ndarray
        Same shape as `dist`.
    """
    return -2 * c1 * dist / (alpha ** 2 + dist ** 2)


COSTS = {
    'logarithmic': (logarithmic, logarithmic_grad),
    'quadratic': (quadratic, quadratic_grad),
    'cauchy': (cauchy, cauchy_grad),
}


def make_cost(kind='logarithmic', c1=C1_NOMINAL, alpha=ALPHA):
    """Bind (cost, gradient) for one cost family and set of parameters.

    Parameters
    ----------
    kind : {'logarithmic', 'quadratic', 'cauchy'}
        Which family to use.  The paper uses the log barrier throughout.
    c1 : float
        Collision weight.  `C1_NOMINAL` reproduces the nominal and robust
        solves; `C1_WIDER` reproduces the "wider" baseline.
    alpha : float
        Barrier smoothing.

    Returns
    -------
    (cost, grad) : tuple of callables
        Each takes distance and returns a value of the same shape.  Pass both to
        a solver; passing only the cost silently selects finite differences.

    Examples
    --------
    >>> cost, grad = make_cost(c1=2.0)
    >>> round(float(cost(2.0)), 6), round(float(grad(2.0)), 6)
    (-2.772589, -2.0)
    """
    if kind not in COSTS:
        raise KeyError(f"unknown cost {kind!r}; choose from {sorted(COSTS)}")
    cost, grad = COSTS[kind]
    return partial(cost, c1=c1, alpha=alpha), partial(grad, c1=c1, alpha=alpha)
