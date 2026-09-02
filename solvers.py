"""Trajectory-optimization entry points: nominal and strategically robust.

Four solvers, all returning a `scipy.optimize.OptimizeResult` whose `.x` is the
flat `[x, u]` decision vector (see `split_solution`):

    optimize_optimal            nominal cost, full (x, u) space
    optimize_everystep          worst-case cost, full (x, u) space
    optimize_optimal_shooting   nominal cost, controls only
    optimize_everystep_shooting worst-case cost, controls only

The `everystep` pair replaces each pair's nominal separation with its worst case
under an adversarial perturbation with a fresh energy budget at every prefix of
the horizon (see `adversary.py`).  The `optimal` pair is the baseline it is
compared against; running it with a larger `c1` gives the "wider" baseline, the
naive way to buy separation.

The two parameterizations solve the same problem differently.  Full space hands
SLSQP n_a*(H+1)*sdim + n_a*H*cdim variables subject to n_a*sdim initial-condition
and n_a*H*sdim dynamics equalities.  Single shooting eliminates all of those by
rolling the dynamics forward from the controls, leaving roughly half the
variables and no equality constraints, with the dynamics exact by construction
rather than satisfied to the solver's constraint tolerance.

They do not start from the same point, and on the harder scenarios that is enough
to reach different local minima: the full-space default guess interpolates x
linearly with u = 0, a pair that its own dynamics do not produce, and shooting
cannot represent an inconsistent start.  Compare objectives, not just runtimes.

Analytic gradients are optional but effectively mandatory.  Without
`distance_cost_grad_func` SLSQP finite-differences the objective, which is both
~100x more function evaluations and, more insidiously, a different answer: the
numeric-gradient run tends to stop early on a worse optimum.  `costs.make_cost`
returns the cost and its gradient together for this reason.
"""
from itertools import combinations

import numpy as np
import scipy as sp

from adversary import compute_selected_worst, precompute_selected, selection_matrix, warm_up


# ---------------------------------------------------------------------------
# Small shared utilities
# ---------------------------------------------------------------------------
def _log(verbose, msg):
    """Print a progress line when a solver is running verbosely.

    Parameters
    ----------
    verbose : bool
        Whether the caller asked for progress output.  False silences the line.
    msg : str
        Text to print.

    Returns
    -------
    None
    """
    if verbose:
        print(msg, flush=True)


def split_solution(result, n_a, H, sdim, cdim):
    """Reshape a flat solution into (x, u) trajectories.

    Accepts an `OptimizeResult` or a raw decision vector, so notebook code and
    benchmark code can share one call instead of repeating the slice arithmetic.

    Parameters
    ----------
    result : OptimizeResult or ndarray
        A solver result, or the flat decision vector `[x, u]` directly.  The
        shooting solvers repack their result into this layout on return, so
        their output unpacks here unchanged.
    n_a, H, sdim, cdim : int
        Number of agents, horizon length, per-agent state dimension, per-agent
        control dimension.

    Returns
    -------
    x : ndarray, shape (n_a, H+1, sdim)
        State trajectory of each agent, t = 0..H.
    u : ndarray, shape (n_a, H, cdim)
        Control trajectory of each agent, t = 0..H-1.
    """
    z = getattr(result, 'x', result)
    n_x = n_a * (H + 1) * sdim
    x = z[:n_x].reshape(n_a, H + 1, sdim)
    u = z[n_x:n_x + n_a * H * cdim].reshape(n_a, H, cdim)
    return x, u


def compute_potential_weights(agent_weights, n_a):
    """Convert per-agent coefficients c_i into potential-game weight factors.

    Given coefficients c_i (how much agent i penalises others uniformly), the
    weighted potential is

        P = sum_i (prod_{j != i} c_j) L_ii + (prod_l c_l) sum_{i<j} L_ij

    Parameters
    ----------
    agent_weights : array-like of shape (n_a,) or None
        Per-agent coefficients c_i.  None means equal weights, i.e. the
        unweighted game, and returns ones and 1.0.
    n_a : int
        Number of agents.

    Returns
    -------
    self_weights : ndarray, shape (n_a,)
        Weight on agent i's own tracking cost, prod_{j != i} c_j.
    collision_weight : float
        Weight applied to every pairwise collision cost, prod_l c_l.
    """
    if agent_weights is None:
        return np.ones(n_a), 1.0
    c = np.asarray(agent_weights, dtype=float)
    total_c = np.prod(c)
    return total_c / c, float(total_c)


def _make_rollout(x0, A, B, H, n_a, sdim):
    """Build the map from controls to the states they imply.

    Used by the single-shooting solvers, where states are not decision variables.
    Because the rollout starts at x0 and applies the dynamics directly, both the
    initial-condition and the dynamics constraints hold to machine precision
    rather than to SLSQP's constraint tolerance.

    Parameters
    ----------
    x0 : ndarray, shape (n_a, sdim)
        Initial state of each agent.
    A, B : ndarray
        Discrete-time dynamics x[t+1] = A x[t] + B u[t].
    H, n_a, sdim : int
        Horizon length, number of agents, per-agent state dimension.

    Returns
    -------
    rollout : callable
        `rollout(u)` takes controls of shape (n_a, H, cdim) and returns states
        of shape (n_a, H+1, sdim), with `x[:, 0] == x0` exactly.
    """
    def rollout(u):
        """States implied by a control sequence."""
        x = np.empty((n_a, H + 1, sdim))
        x[:, 0, :] = x0
        for t in range(H):
            x[:, t + 1, :] = x[:, t, :] @ A.T + u[:, t, :] @ B.T
        return x
    return rollout


def _adjoint_to_controls(grad_x, grad_u, A, B, H):
    """Fold dJ/dx back through a rollout to get the total derivative dJ/du.

    u[s] moves every later state, so dJ/du[s] = grad_u[s] + sum_{t>s}
    (A^(t-1-s) B)^T grad_x[t].  Accumulated backwards in one sweep:
    mu[t] = grad_x[t] + A^T mu[t+1], then dJ/du[s] = grad_u[s] + B^T mu[s+1].

    grad_x[0] is intentionally dropped: x[0] is pinned to x0 and no control
    influences it, so its collision-cost contribution is a constant.

    Parameters
    ----------
    grad_x : ndarray, shape (n_a, H+1, sdim)
        Partial derivative of the objective with respect to the states.
    grad_u : ndarray, shape (n_a, H, cdim)
        Partial derivative with respect to the controls.  Modified in place and
        also returned.
    A, B : ndarray
        The same dynamics used to produce the states.
    H : int
        Horizon length.

    Returns
    -------
    grad_u : ndarray, shape (n_a, H, cdim)
        The total derivative dJ/du, the same array passed in.
    """
    n_a, _, sdim = grad_x.shape
    mu = np.zeros((n_a, H + 2, sdim))
    for t in range(H, 0, -1):
        mu[:, t, :] = grad_x[:, t, :] + mu[:, t + 1, :] @ A
    for s in range(H):
        grad_u[:, s, :] += mu[:, s + 1, :] @ B
    return grad_u


# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------
def _dynamics_constraint(x0, A, B, H, n_a, sdim, cdim):
    """Equality constraint pinning x[:, 0] to x0 and enforcing x[t+1] = Ax + Bu.

    Only the full-space solvers need this; shooting satisfies both by
    construction.  The Jacobian is constant -- the dynamics are linear, so the
    entries are built only from A, B and I -- and could be supplied once instead
    of finite-differenced each iteration -- see the appendix of README.md.  It is
    left to SLSQP here so the constraint set matches the published runs.

    Parameters
    ----------
    x0 : ndarray, shape (n_a, sdim)
        Initial state each agent is pinned to.
    A, B : ndarray
        Discrete-time dynamics x[t+1] = A x[t] + B u[t].
    H, n_a, sdim, cdim : int
        Horizon length, number of agents, state and control dimensions.

    Returns
    -------
    constraint : dict
        A SciPy equality-constraint dict `{"type": "eq", "fun": ...}` whose
        residual has length n_a*sdim + n_a*H*sdim and is zero on feasible z.
    """
    def dynamics(z):
        """Initial-condition and dynamics residual at this iterate."""
        x, u = split_solution(z, n_a, H, sdim, cdim)
        ic = (x[:, 0, :] - x0).flatten()
        predicted = np.einsum('ij,ntj->nti', A, x[:, :H, :]) + np.einsum('ij,ntj->nti', B, u)
        return np.concatenate([ic, (x[:, 1:, :] - predicted).flatten()])
    return {'type': 'eq', 'fun': dynamics}


def _control_norm_constraint(u_max, n_a, H, cdim, n_vars, u_offset):
    """Per-step control-norm limit ||u[i, t]|| <= u_max, with exact Jacobian.

    Pure function of u, so it survives the shooting reduction unchanged -- only
    the column offset moves.

    Parameters
    ----------
    u_max : float
        Largest permitted control norm at any single step.
    n_a, H, cdim : int
        Number of agents, horizon length, per-agent control dimension.
    n_vars : int
        Length of the decision vector, i.e. the Jacobian's column count.
    u_offset : int
        Index at which the control block starts in the decision vector.  Zero
        for the shooting solvers, where u is the whole vector.

    Returns
    -------
    constraint : dict
        A SciPy inequality-constraint dict with an exact `jac`, whose residual
        `u_max**2 - ||u[i, t]||**2` has length n_a*H and must stay >= 0.
    """
    def fun(z):
        """Control-norm slack at this iterate."""
        u = z[u_offset:u_offset + n_a * H * cdim].reshape(n_a, H, cdim)
        return (u_max ** 2 - np.sum(u ** 2, axis=2)).flatten()      # >= 0

    def jac(z):
        """Exact Jacobian of the control-norm constraint."""
        u = z[u_offset:u_offset + n_a * H * cdim].reshape(n_a, H, cdim)
        out = np.zeros((n_a * H, n_vars))
        for i in range(n_a):
            for t in range(H):
                col = u_offset + (i * H + t) * cdim
                out[i * H + t, col:col + cdim] = -2 * u[i, t, :]
        return out

    return {'type': 'ineq', 'fun': fun, 'jac': jac}


# ---------------------------------------------------------------------------
# Cost builders, shared by the full-space and single-shooting drivers
# ---------------------------------------------------------------------------
# Both return closures over trajectory space taking (x, u, key) and returning
# either the scalar objective or (grad_x, grad_u); each driver then maps that
# into whatever it actually optimizes over.  Sharing them is what keeps the two
# parameterizations from drifting apart -- the failure mode that once left a
# numba kernel out of sync with its twin for two commits.
def _make_optimal_costs(xf, H, Q, R, Qf, n_a, sdim, cdim, pdim,
                        distance_cost_func, distance_cost_grad_func,
                        self_weights, collision_weight):
    """Nominal cost: pairwise distances taken at the trajectory itself.

    No adversary, no multiplier, no cache -- the cost is a direct function of x,
    so `key` is accepted only to match the everystep signature.

    Parameters
    ----------
    xf : ndarray, shape (n_a, sdim)
        Desired final state of each agent.
    H : int
        Horizon length.
    Q, R, Qf : ndarray
        Running state, control, and terminal state cost matrices.
    n_a, sdim, cdim : int
        Number of agents, per-agent state and control dimensions.
    pdim : int
        Number of leading state components the collision cost sees.
    distance_cost_func : callable
        Pair cost as a function of distance, parameters already bound.
    distance_cost_grad_func : callable or None
        Its derivative.  Only `gradient` uses it; when None the caller must not
        supply a Jacobian to SLSQP.
    self_weights : ndarray, shape (n_a,)
    collision_weight : float
        Potential-game weights from `compute_potential_weights`.

    Returns
    -------
    objective : callable
        `objective(x, u, key=None) -> float`, the total cost.
    gradient : callable
        `gradient(x, u, key=None) -> (grad_x, grad_u)`, with shapes matching
        `x` and `u`.  The caller maps these into its own decision variables.
    """
    pairs = list(combinations(range(n_a), 2))
    idx_i = [p[0] for p in pairs]
    idx_j = [p[1] for p in pairs]

    def objective(x, u, key=None):
        """Total nominal cost of one trajectory."""
        dx_running = x[:, :H, :] - xf[:, None, :]
        dx_terminal = x[:, H, :] - xf
        total = 0.0
        total += np.einsum('n,nti,ij,ntj->', self_weights, dx_running, Q, dx_running)
        total += np.einsum('n,ni,ij,nj->', self_weights, dx_terminal, Qf, dx_terminal)
        total += np.einsum('n,nti,ij,ntj->', self_weights, u, R, u)

        dp = x[idx_i, :, :pdim] - x[idx_j, :, :pdim]
        dists = np.linalg.norm(dp, axis=2)
        total += collision_weight * np.sum(distance_cost_func(dists))
        return total

    def gradient(x, u, key=None):
        """Analytic gradient of the nominal cost."""
        grad_x = np.zeros_like(x)
        grad_u = np.zeros_like(u)
        grad_x[:, :H, :] = 2 * np.einsum('n,ij,ntj->nti', self_weights, Q,
                                         x[:, :H, :] - xf[:, None, :])
        grad_x[:, H, :] = 2 * np.einsum('n,ij,nj->ni', self_weights, Qf, x[:, H, :] - xf)
        grad_u[:] = 2 * np.einsum('n,ij,ntj->nti', self_weights, R, u)

        # d/dd of the pair cost, converted to a position gradient by the chain
        # rule: df/d(dp) = f'(d) * dp / d, with the clamp guarding d -> 0.
        dp = x[idx_i, :, :pdim] - x[idx_j, :, :pdim]        # (n_pairs, H+1, pdim)
        dists = np.linalg.norm(dp, axis=2)                  # (n_pairs, H+1)
        dcost_over_d = distance_cost_grad_func(dists) / np.maximum(dists, 1e-12)
        dp_grad = collision_weight * dcost_over_d[..., None] * dp
        for k, (i, j) in enumerate(pairs):
            grad_x[i, :, :pdim] += dp_grad[k]
            grad_x[j, :, :pdim] -= dp_grad[k]
        return grad_x, grad_u

    return objective, gradient


def _make_everystep_costs(xf, H, Q, R, Qf, n_a, sdim, cdim, pdim, eps,
                          distance_cost_func, distance_cost_grad_func,
                          self_weights, collision_weight, C, selected):
    """Strategically robust cost: pairwise distances taken at the worst case.

    The pair sweep is memoized on a caller-supplied key with a single-entry
    cache, so a matched objective/gradient pair at the same iterate costs one
    sweep rather than two.  The worst case depends on x alone -- the adversary
    perturbs the relative state, not the controls -- which is why shooting can
    key the cache on its control vector.

    Parameters
    ----------
    xf : ndarray, shape (n_a, sdim)
        Desired final state of each agent.
    H : int
        Horizon length.
    Q, R, Qf : ndarray
        Running state, control, and terminal state cost matrices.
    n_a, sdim, cdim : int
        Number of agents, per-agent state and control dimensions.
    pdim : int
        Number of leading state components the collision cost sees.
    eps : float
        Adversary's per-step energy budget.
    distance_cost_func : callable
        Pair cost as a function of distance, parameters already bound.
    distance_cost_grad_func : callable or None
        Its derivative; used only by `gradient`.
    self_weights : ndarray, shape (n_a,)
    collision_weight : float
        Potential-game weights from `compute_potential_weights`.
    C : ndarray, shape (m, sdim)
        Selection matrix, from `adversary.selection_matrix`.
    selected : tuple of ndarray
        `(sigma_all, P_all)` from `adversary.precompute_selected`.

    Returns
    -------
    objective : callable
        `objective(x, u, key) -> float`, the total worst-case cost.
    gradient : callable
        `gradient(x, u, key) -> (grad_x, grad_u)`.

    Both closures require `key` -- a hashable stand-in for the current iterate,
    such as `z.tobytes()` -- and share one single-entry pair cache keyed on it.

    Notes
    -----
    The gradient is an envelope-theorem gradient: the adversary's response does
    not appear in it.  At the optimal multiplier the adversary's controls are
    stationary, so they may be held fixed when differentiating the outer
    objective.  Held fixed, dz_N no longer depends on z_N, giving
    d(z_worst)/d(z_N) = I -- the Riccati recursion is never differentiated
    through.  The derivation, and the I + P_N version that was wrong by order 10,
    are in the appendix of README.md.
    """
    sigma_all, P_all = selected
    pairs = list(combinations(range(n_a), 2))
    m = C.shape[0]
    _cache = {}

    def pair_data(x, key):
        """Worst case for every pair at the current iterate, memoized."""
        if key not in _cache:
            results = {}
            for agent_i, agent_j in pairs:
                # z_worst is (H+1, m): only the selected components are ever read
                # downstream, so the unselected ones are never formed.
                zC = np.ascontiguousarray((C @ (x[agent_i] - x[agent_j]).T).T)
                z_worst, lamb = compute_selected_worst(zC, sigma_all, P_all, eps, H, m)
                results[(agent_i, agent_j)] = (lamb, z_worst)
            _cache.clear()
            _cache[key] = results
        return _cache[key]

    def objective(x, u, key):
        """Total strategically robust cost of one trajectory."""
        dx_running = x[:, :H, :] - xf[:, None, :]
        dx_terminal = x[:, H, :] - xf
        total = 0.0
        total += np.einsum('n,nti,ij,ntj->', self_weights, dx_running, Q, dx_running)
        total += np.einsum('n,ni,ij,nj->', self_weights, dx_terminal, Qf, dx_terminal)
        total += np.einsum('n,nti,ij,ntj->', self_weights, u, R, u)

        pd = pair_data(x, key)
        for pair in pairs:
            _, z_worst = pd[pair]
            # z_worst is already in selected coordinates -- no [:pdim] slice.
            # Step 0 is excluded: x[:, 0] is pinned to x0, so its cost is constant.
            dists = np.linalg.norm(z_worst[1:], axis=1)
            total += collision_weight * np.sum(distance_cost_func(dists))
        return total

    def gradient(x, u, key):
        """Envelope-theorem gradient: the adversary's response does not appear."""
        grad_x = np.zeros_like(x)
        grad_u = np.zeros_like(u)
        grad_x[:, :H, :] = 2 * np.einsum('n,ij,ntj->nti', self_weights, Q,
                                         x[:, :H, :] - xf[:, None, :])
        grad_x[:, H, :] = 2 * np.einsum('n,ij,nj->ni', self_weights, Qf, x[:, H, :] - xf)
        grad_u[:] = 2 * np.einsum('n,ij,ntj->nti', self_weights, R, u)

        if not pairs:                       # a lone agent has no collision term
            return grad_x, grad_u

        pd = pair_data(x, key)
        # Batched over pairs and prefixes at once.  The per-prefix Python loop
        # this replaces was 78% of the gradient's cost at 4 agents, and it sits on
        # SLSQP's hot path -- the adversary's own numba sweep is the cheap part.
        # Step 0 is dropped: x[:, 0] is pinned to x0, so its cost is constant.
        zw = np.stack([pd[p][1][1:] for p in pairs])            # (n_pairs, H, m)
        d = np.linalg.norm(zw, axis=2)                          # (n_pairs, H)
        safe = np.maximum(d, 1e-12)                             # guards d -> 0
        coef = collision_weight * distance_cost_grad_func(safe)
        # Multiply then divide, matching the scalar form it replaces exactly.
        # The trailing `@ C` pulls the selected-coordinate gradient back through
        # C, one row at a time: (v @ C) equals (C.T @ v).  For C = [I_p 0] this
        # writes into the position components, and stays correct for a weighted C.
        dcost_dz = ((coef[..., None] * zw) / safe[..., None]) @ C   # (n_pairs, H, sdim)
        for q, (agent_i, agent_j) in enumerate(pairs):
            grad_x[agent_i, 1:, :] += dcost_dz[q]
            grad_x[agent_j, 1:, :] -= dcost_dz[q]
        return grad_x, grad_u

    return objective, gradient


# ---------------------------------------------------------------------------
# Single-shooting solvers: decision variables are u alone
# ---------------------------------------------------------------------------
def optimize_optimal_shooting(x0, xf, A, B, H, Q, R, Qf, n_a, sdim, cdim, pdim,
                              distance_cost_func, distance_cost_grad_func=None,
                              x_opt=None, u_opt=None, agent_weights=None,
                              u_max=None, maxiter=1000, verbose=False):
    """Single-shooting form of `optimize_optimal`.

    Optimizes over controls alone and recovers states by rolling the dynamics
    forward, so the n_a*sdim initial-condition and n_a*H*sdim dynamics equalities
    disappear.  At n_a = 8, H = 20 that is 656 variables and 336 equality
    constraints down to 320 variables and none.

    This is the larger of the two shooting wins on the paper's tables: the
    nominal solve dominates them, because the robust timing is charged for its
    own nominal initialization.  Reformulating only the robust half would barely
    move the reported ratios.

    Parameters
    ----------
    x0, xf : ndarray, shape (n_a, sdim)
        Initial and desired final state of each agent.
    A, B : ndarray
        Discrete-time dynamics x[t+1] = A x[t] + B u[t], shared by all agents.
        Agents are dynamically independent; they couple only through the cost.
    H : int
        Horizon length.  States run t = 0..H, controls t = 0..H-1.
    Q, R, Qf : ndarray
        Running state, control, and terminal state cost matrices.
    n_a, sdim, cdim : int
        Number of agents, per-agent state dimension, per-agent control dimension.
    pdim : int
        Number of leading state components treated as position, i.e. the ones
        the collision cost sees.
    distance_cost_func : callable
        Pair cost as a function of distance, parameters already bound.
    distance_cost_grad_func : callable, optional
        Its derivative.  Strongly recommended: without it SLSQP
        finite-differences the objective, costing n_a*H*cdim rollouts per
        gradient.
    x_opt : ignored
        Accepted only so the signature matches `optimize_optimal`.  States here
        are a function of the controls; warm-start via `u_opt`.
    u_opt : ndarray, shape (n_a, H, cdim), optional
        Warm start.  The default guess is the least-squares shadow of the
        full-space linear interpolation, mapped through the dynamics.
    agent_weights : array-like of shape (n_a,), optional
        Per-agent coefficients for the weighted potential game.
    u_max : float, optional
        Per-step control-norm limit.  Survives the reduction unchanged.
    maxiter : int
        SLSQP iteration cap.
    verbose : bool
        Print progress and SLSQP's convergence report.

    Returns
    -------
    result : OptimizeResult
        `.x` is repacked into the full [x, u] layout, so every downstream
        consumer works unchanged.  `.nfev` counts control-space evaluations and
        is not comparable to the full-space figure.
    """
    _log(verbose, "Setting up optimal control single-shooting problem.")
    self_weights, collision_weight = compute_potential_weights(agent_weights, n_a)
    n_u = n_a * H * cdim

    objective_xu, gradient_xu = _make_optimal_costs(
        xf, H, Q, R, Qf, n_a, sdim, cdim, pdim,
        distance_cost_func, distance_cost_grad_func,
        self_weights, collision_weight)
    rollout = _make_rollout(x0, A, B, H, n_a, sdim)

    if u_opt is not None:
        u0 = np.asarray(u_opt).flatten().copy()
    else:
        # Mirror the full-space default guess (linear interpolation in state
        # space) by mapping it through the dynamics: pick the controls that best
        # reproduce that reference.  pinv handles non-square or rank-deficient B;
        # where the reference is unreachable this is its least-squares shadow.
        B_pinv = np.linalg.pinv(B)
        u_init = np.zeros((n_a, H, cdim))
        for i in range(n_a):
            x_lin = np.linspace(x0[i], xf[i], H + 1)
            for t in range(H):
                u_init[i, t] = B_pinv @ (x_lin[t + 1] - A @ x_lin[t])
        u0 = u_init.flatten()

    constraints = []
    if u_max is not None:
        constraints.append(_control_norm_constraint(
            u_max, n_a, H, cdim, n_u, u_offset=0))

    def objective(u_var):
        """Nominal cost of the trajectory these controls produce."""
        u = u_var.reshape(n_a, H, cdim)
        return objective_xu(rollout(u), u)

    def jac(u_var):
        """Total derivative dJ/du, folded back through the rollout."""
        u = u_var.reshape(n_a, H, cdim)
        grad_x, grad_u = gradient_xu(rollout(u), u)
        return _adjoint_to_controls(grad_x, grad_u, A, B, H).flatten()

    _log(verbose, "Starting optimization.")
    result = sp.optimize.minimize(
        objective, u0, method='SLSQP',
        jac=jac if distance_cost_grad_func is not None else None,
        constraints=constraints,
        options={'maxiter': maxiter, 'disp': verbose})

    u_final = result.x.reshape(n_a, H, cdim)
    result.x = np.concatenate([rollout(u_final).flatten(), u_final.flatten()])
    return result


def optimize_everystep_shooting(x0, xf, A, B, H, Q, R, Qf, n_a, sdim, cdim, pdim,
                                eps, distance_cost_func, x_opt=None, u_opt=None,
                                distance_cost_grad_func=None, agent_weights=None,
                                u_max=None, maxiter=1000, verbose=False):
    """Single-shooting form of `optimize_everystep`.

    Same reduction as `optimize_optimal_shooting`: optimize over controls alone
    and recover states by rolling the dynamics forward, removing every equality
    constraint.

    The pair cache keys on the control vector, which is sound because the worst
    case depends on the relative *state* alone and the rollout makes states a
    deterministic function of the controls.

    Parameters
    ----------
    x0, xf : ndarray, shape (n_a, sdim)
        Initial and desired final state of each agent.
    A, B : ndarray
        Discrete-time dynamics x[t+1] = A x[t] + B u[t], shared by all agents.
        Agents are dynamically independent; they couple only through the cost.
    H : int
        Horizon length.  States run t = 0..H, controls t = 0..H-1.
    Q, R, Qf : ndarray
        Running state, control, and terminal state cost matrices.
    n_a, sdim, cdim : int
        Number of agents, per-agent state dimension, per-agent control dimension.
    pdim : int
        Number of leading state components treated as position, i.e. the ones
        the collision cost sees.
    eps : float
        Adversary's per-step energy budget.
    distance_cost_func : callable
        Pair cost as a function of distance, parameters already bound.
    x_opt : ignored
        Accepted only so the signature matches `optimize_everystep`.  States here
        are a function of the controls; warm-start via `u_opt`.
    u_opt : ndarray, shape (n_a, H, cdim), optional
        Warm start.  Defaults to zero controls.
    distance_cost_grad_func : callable, optional
        Its derivative.  Strongly recommended: without it SLSQP
        finite-differences the objective, costing n_a*H*cdim rollouts plus a
        pair sweep per gradient.
    agent_weights : array-like of shape (n_a,), optional
        Per-agent coefficients for the weighted potential game.
    u_max : float, optional
        Per-step control-norm limit.  Survives the reduction unchanged.
    maxiter : int
        SLSQP iteration cap.
    verbose : bool
        Print progress and SLSQP's convergence report.

    Returns
    -------
    result : OptimizeResult
        `.x` is repacked into the full [x, u] layout, so every downstream
        consumer works unchanged.  `.nfev` counts control-space evaluations and
        is not comparable to the full-space figure.
    """
    _log(verbose, "Setting up everystep single-shooting optimization problem.")
    self_weights, collision_weight = compute_potential_weights(agent_weights, n_a)
    n_u = n_a * H * cdim

    u0 = np.zeros(n_u)
    if u_opt is not None:
        u0[:] = u_opt.flatten()

    _log(verbose, "Starting precomputation")
    C = selection_matrix(pdim, sdim)
    selected = precompute_selected(A, B, C, H)
    warm_up(selected, eps, H, pdim)
    _log(verbose, "Precomputation and JIT warmup done.")

    objective_xu, gradient_xu = _make_everystep_costs(
        xf, H, Q, R, Qf, n_a, sdim, cdim, pdim, eps,
        distance_cost_func, distance_cost_grad_func,
        self_weights, collision_weight, C, selected)
    rollout = _make_rollout(x0, A, B, H, n_a, sdim)

    constraints = []
    if u_max is not None:
        constraints.append(_control_norm_constraint(
            u_max, n_a, H, cdim, n_u, u_offset=0))

    def objective(u_var):
        """Worst-case cost of the trajectory these controls produce."""
        u = u_var.reshape(n_a, H, cdim)
        return objective_xu(rollout(u), u, u_var.tobytes())

    def jac(u_var):
        """Total derivative dJ/du, folded back through the rollout."""
        u = u_var.reshape(n_a, H, cdim)
        grad_x, grad_u = gradient_xu(rollout(u), u, u_var.tobytes())
        return _adjoint_to_controls(grad_x, grad_u, A, B, H).flatten()

    _log(verbose, "Starting optimization.")
    result = sp.optimize.minimize(
        objective, u0, method='SLSQP',
        jac=jac if distance_cost_grad_func is not None else None,
        bounds=[(None, None)] * n_u,
        constraints=constraints,
        options={'maxiter': maxiter, 'disp': verbose})

    u_final = result.x.reshape(n_a, H, cdim)
    result.x = np.concatenate([rollout(u_final).flatten(), u_final.flatten()])
    return result


# ---------------------------------------------------------------------------
# Full-space solvers: decision variables are (x, u), dynamics as constraints
# ---------------------------------------------------------------------------
def optimize_optimal(x0, xf, A, B, H, Q, R, Qf, n_a, sdim, cdim, pdim,
                     distance_cost_func, distance_cost_grad_func=None,
                     x_opt=None, u_opt=None, agent_weights=None, u_max=None,
                     maxiter=1000, verbose=False):
    """Nominal solve: collision cost at the trajectory itself, no adversary.

    This is the baseline the robust solve is measured against, and -- run again
    with a larger c1 -- the "wider" baseline as well.

    Parameters
    ----------
    x0, xf : ndarray, shape (n_a, sdim)
        Initial and desired final state of each agent.
    A, B : ndarray
        Discrete-time dynamics x[t+1] = A x[t] + B u[t], shared by all agents.
        Agents are dynamically independent; they couple only through the cost.
    H : int
        Horizon length.  States run t = 0..H, controls t = 0..H-1.
    Q, R, Qf : ndarray
        Running state, control, and terminal state cost matrices.
    n_a, sdim, cdim : int
        Number of agents, per-agent state dimension, per-agent control dimension.
    pdim : int
        Number of leading state components treated as position, i.e. the ones the
        collision cost sees.
    distance_cost_func : callable
        Pair cost as a function of distance, parameters already bound.  See
        `costs.make_cost`.
    distance_cost_grad_func : callable, optional
        Its derivative.  Strongly recommended: without it SLSQP finite-differences
        the objective, costing ~100x the function evaluations and often converging
        somewhere worse.
    x_opt, u_opt : ndarray, optional
        Warm start.  Both must be given, or neither; the default guess
        interpolates x linearly from x0 to xf and sets u = 0.
    agent_weights : array-like of shape (n_a,), optional
        Per-agent coefficients for the weighted potential game.  None is the
        unweighted case.
    u_max : float, optional
        Per-step control-norm limit.  None leaves controls unconstrained.
    maxiter : int
        SLSQP iteration cap.
    verbose : bool
        Print progress and SLSQP's convergence report.

    Returns
    -------
    result : OptimizeResult
        `.x` is the flat [x, u] vector; unpack it with `split_solution`.
    """
    _log(verbose, "Setting up optimal control optimization problem.")
    self_weights, collision_weight = compute_potential_weights(agent_weights, n_a)

    objective_xu, gradient_xu = _make_optimal_costs(
        xf, H, Q, R, Qf, n_a, sdim, cdim, pdim,
        distance_cost_func, distance_cost_grad_func,
        self_weights, collision_weight)

    n_vars = n_a * (H + 1) * sdim + n_a * H * cdim
    if x_opt is not None and u_opt is not None:
        z0 = np.concatenate([x_opt.flatten(), u_opt.flatten()])
    else:
        z0 = np.zeros(n_vars)
        for i in range(n_a):
            start = i * (H + 1) * sdim
            z0[start:start + (H + 1) * sdim] = np.linspace(x0[i], xf[i], H + 1).flatten()

    constraints = [_dynamics_constraint(x0, A, B, H, n_a, sdim, cdim)]
    if u_max is not None:
        constraints.append(_control_norm_constraint(
            u_max, n_a, H, cdim, n_vars, u_offset=n_a * (H + 1) * sdim))

    def objective(z):
        """Nominal cost at this iterate."""
        return objective_xu(*split_solution(z, n_a, H, sdim, cdim))

    def jac(z):
        """Analytic gradient at this iterate, flattened to match z."""
        grad_x, grad_u = gradient_xu(*split_solution(z, n_a, H, sdim, cdim))
        return np.concatenate([grad_x.flatten(), grad_u.flatten()])

    _log(verbose, "Starting optimization.")
    return sp.optimize.minimize(
        objective, z0, method='SLSQP',
        jac=jac if distance_cost_grad_func is not None else None,
        constraints=constraints,
        options={'maxiter': maxiter, 'disp': verbose})


def optimize_everystep(x0, xf, A, B, H, Q, R, Qf, n_a, sdim, cdim, pdim, eps,
                       distance_cost_func, x_opt=None, u_opt=None,
                       distance_cost_grad_func=None, agent_weights=None,
                       u_max=None, maxiter=1000, verbose=False):
    """Strategically robust solve: collision cost at the worst case, every step.

    Each pair's separation is evaluated after an adversary, given a fresh energy
    budget at every prefix of the horizon, has moved them as close together as it
    can.  `adversary.py` holds the closed form; the appendix of README.md derives
    why the adversary's response never appears in this solve's gradient.

    Warm-starting from a nominal solution is the intended use, and is what the
    paper's timings measure -- the robust runtime there includes its own nominal
    initialization.

    Parameters
    ----------
    x0, xf : ndarray, shape (n_a, sdim)
        Initial and desired final state of each agent.
    A, B : ndarray
        Discrete-time dynamics x[t+1] = A x[t] + B u[t], shared by all agents.
        Agents are dynamically independent; they couple only through the cost.
    H : int
        Horizon length.  States run t = 0..H, controls t = 0..H-1.
    Q, R, Qf : ndarray
        Running state, control, and terminal state cost matrices.
    n_a, sdim, cdim : int
        Number of agents, per-agent state dimension, per-agent control dimension.
    pdim : int
        Number of leading state components treated as position, i.e. the ones
        the collision cost sees.
    eps : float
        Adversary's per-step energy budget.  eps = 0 recovers the nominal cost.
    distance_cost_func : callable
        Pair cost as a function of distance, parameters already bound.  See
        `costs.make_cost`.
    x_opt, u_opt : ndarray, optional
        Warm start, shapes (n_a, H+1, sdim) and (n_a, H, cdim).  Unlike
        `optimize_optimal`, these may be given independently; whichever is
        supplied seeds its block of the initial guess and the rest stays zero.
    distance_cost_grad_func : callable, optional
        Its derivative.  Strongly recommended: without it SLSQP
        finite-differences the objective, costing ~100x the function evaluations
        and often converging somewhere worse.
    agent_weights : array-like of shape (n_a,), optional
        Per-agent coefficients for the weighted potential game.  None is the
        unweighted case.
    u_max : float, optional
        Per-step control-norm limit.  None leaves controls unconstrained.
    maxiter : int
        SLSQP iteration cap.
    verbose : bool
        Print progress and SLSQP's convergence report.

    Returns
    -------
    result : OptimizeResult
        `.x` is the flat [x, u] vector; unpack it with `split_solution`.
    """
    _log(verbose, "Setting up everystep optimization problem.")
    self_weights, collision_weight = compute_potential_weights(agent_weights, n_a)

    n_vars = n_a * (H + 1) * sdim + n_a * H * cdim
    n_x = n_a * (H + 1) * sdim
    z0 = np.zeros(n_vars)
    if x_opt is not None:
        z0[:n_x] = x_opt.flatten()
    if u_opt is not None:
        z0[n_x:n_x + n_a * H * cdim] = u_opt.flatten()

    _log(verbose, "Starting precomputation")
    C = selection_matrix(pdim, sdim)
    selected = precompute_selected(A, B, C, H)
    warm_up(selected, eps, H, pdim)
    _log(verbose, "Precomputation and JIT warmup done.")

    objective_xu, gradient_xu = _make_everystep_costs(
        xf, H, Q, R, Qf, n_a, sdim, cdim, pdim, eps,
        distance_cost_func, distance_cost_grad_func,
        self_weights, collision_weight, C, selected)

    constraints = [_dynamics_constraint(x0, A, B, H, n_a, sdim, cdim)]
    if u_max is not None:
        constraints.append(_control_norm_constraint(
            u_max, n_a, H, cdim, n_vars, u_offset=n_x))

    def objective(z):
        """Worst-case cost at this iterate."""
        x, u = split_solution(z, n_a, H, sdim, cdim)
        return objective_xu(x, u, z.tobytes())

    def jac(z):
        """Analytic gradient at this iterate, flattened to match z."""
        x, u = split_solution(z, n_a, H, sdim, cdim)
        grad_x, grad_u = gradient_xu(x, u, z.tobytes())
        return np.concatenate([grad_x.flatten(), grad_u.flatten()])

    _log(verbose, "Starting optimization.")
    return sp.optimize.minimize(
        objective, z0, method='SLSQP',
        jac=jac if distance_cost_grad_func is not None else None,
        constraints=constraints,
        bounds=[(None, None)] * n_vars,
        options={'maxiter': maxiter, 'disp': verbose})