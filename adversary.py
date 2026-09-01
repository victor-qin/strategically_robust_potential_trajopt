"""The worst-case oracle: what the adversary can do to one pair of agents.

For a pair of agents with relative state z_N = x_i[N] - x_j[N], the adversary
picks a perturbation of the relative dynamics, with energy budget eps, that
brings the pair as close together as possible at step N.  It gets a fresh budget
for every prefix length N = 1..H -- hence "everystep": the guarantee holds at
each point of the horizon, not just at the end.

The solve is available in closed form.  With the reachability Gramian

    G_N = sum_{l<N} A^l B B^T (A^T)^l

and a selection matrix C picking the collision-relevant components of the state,
the worst-case selected relative state is

    C z_worst_N = C z_N - Gamma_N (Gamma_N + lambda_N I)^-1 C z_N,
    Gamma_N = C G_N C^T,

with lambda_N >= 0 the multiplier on the energy constraint, fixed by the secular
equation sum_l sigma_l w_l^2 / (lambda + sigma_l)^2 = eps^2 in the eigenbasis of
Gamma_N.  Two consequences drive the runtime numbers:

* Everything reduces to eigendecompositions of Gamma_N, which is (m x m) with
  m = pdim -- typically 2 -- rather than (sdim x sdim).  There are H of them,
  computed once offline, and A is never inverted.
* Online, the worst case costs two matrix-vector products and a scalar Newton
  solve per pair per step.  No forward pass over the adversary's controls is
  needed.

The gradient of the outer objective does not differentiate through this map at
all: by the envelope theorem the adversary's response is held fixed, giving
d(z_worst_N)/d(z_N) = I.  The derivation, and the version of it that was wrong by
order 10, are in the appendix of README.md.

This module is numerics only: it imports numpy and numba and nothing else, so
the solver can run headless with no plotting stack installed.
"""
import numpy as np
from numba import njit


def selection_matrix(pdim, sdim):
    """C = [I_p  0]: keep the first `pdim` state components, drop the rest.

    A general C is supported throughout -- C = S^(1/2) reproduces a cost weighted
    by ||z||_S -- but the position-only case is what the paper uses.  Gamma_N is
    symmetric for any C, so its eigenvectors are orthogonal and neither a
    generalized eigenproblem nor an inverse of S is ever formed.

    Parameters
    ----------
    pdim : int
        Number of leading state components the collision cost sees.
    sdim : int
        Per-agent state dimension.  Must be >= pdim.

    Returns
    -------
    C : ndarray, shape (pdim, sdim)
        The selection matrix [I_p  0].
    """
    return np.hstack([np.eye(pdim), np.zeros((pdim, sdim - pdim))])


def precompute_selected(A, B, C, H):
    """Eigendecompose the selected Gramian Gamma_N = C G_N C^T for N = 1..H.

    G_N is accumulated incrementally as G_N = B B^T + A G_{N-1} A^T, so A need
    not be invertible.  Called once per problem; the result is reused at every
    SLSQP iteration.

    Parameters
    ----------
    A, B : ndarray
        Discrete-time dynamics of a single agent.  The relative state of a pair
        obeys the same dynamics, which is what makes one precompute serve every
        pair.
    C : ndarray, shape (m, sdim)
        Selects/weights the collision-relevant components of the state; see
        `selection_matrix`.
    H : int
        Horizon length.  One eigendecomposition is produced per prefix N = 1..H.

    Returns
    -------
    sigma_all : ndarray, shape (H, m)      eigenvalues of Gamma_N
    P_all     : ndarray, shape (H, m, m)   eigenvectors of Gamma_N
    """
    sdim, m = A.shape[0], C.shape[0]
    sigma_all = np.zeros((H, m))
    P_all = np.zeros((H, m, m))
    G = np.zeros((sdim, sdim))
    BBT = B @ B.T
    for N in range(1, H + 1):
        G = BBT + A @ G @ A.T
        sigma, P = np.linalg.eigh(C @ G @ C.T)
        sigma_all[N - 1] = np.maximum(sigma, 0.0)   # clip eigh round-off on PSD
        P_all[N - 1] = P
    return sigma_all, P_all


@njit(cache=True)
def compute_selected_worst(zC, sigma_all, P_all, eps, H, m):
    """Worst-case selected relative state and its multiplier, for every prefix.

    Evaluates C z_worst_N = C z_N - Gamma_N (Gamma_N + lambda_N I)^-1 C z_N as two
    matrix-vector products in the eigenbasis of Gamma_N, then solves the secular
    equation for lambda_N by Newton's method reusing the same w = P_N^T C z_N.

    Parameters
    ----------
    zC : ndarray, shape (H+1, m)
        Selected relative state; row N is C z_N for the horizon-N subproblem.
    sigma_all, P_all : ndarray
        Output of `precompute_selected`.
    eps : float
        Adversary's per-step energy budget.
    H, m : int
        Horizon length and number of selected components.

    Returns
    -------
    zw   : ndarray, shape (H+1, m)   zw[N] = C (z_N + dz_N); row 0 is unused.
    lamb : ndarray, shape (H,)       lamb[N-1] = lambda_N
    """
    zw = np.zeros((H + 1, m))
    lamb = np.zeros(H)
    eps2 = eps * eps

    for N in range(1, H + 1):
        sigma = sigma_all[N - 1]
        P = P_all[N - 1]
        y = zC[N]
        w = P.T @ y
        num = sigma * w * w

        # A root lambda > 0 exists iff sum_l w_l^2 / sigma_l > eps^2.  Otherwise
        # lambda = 0 and the adversary reaches contact exactly: C dz_N = -C z_N,
        # so zw[N] stays at zero and the pair registers as touching.
        f0 = 0.0
        blocked = False
        for l in range(m):
            if sigma[l] > 1e-14:
                f0 += w[l] * w[l] / sigma[l]
            elif abs(w[l]) > 1e-14:
                blocked = True              # uncontrollable direction with content
                break
        if not blocked and f0 <= eps2:
            lamb[N - 1] = 0.0
            continue                        # zw[N] stays 0

        # Newton on  sum_l sigma_l w_l^2 / (lambda + sigma_l)^2 = eps^2
        lam = np.mean(sigma) * 1.5
        for _ in range(50):
            d = lam + sigma
            t = num / (d * d)
            fv = np.sum(t) - eps2
            if abs(fv) < 1e-12:
                break
            df = -2.0 * np.sum(t / d)
            if abs(df) < 1e-14:
                break
            lam -= fv / df
            if lam < 0.0:
                lam = 0.0
        lamb[N - 1] = lam
        zw[N] = y - P @ ((sigma / (lam + sigma)) * w)
    return zw, lamb


def worst_case_relative(x_i, x_j, C, selected, eps, H):
    """Worst case for one pair, from their full state trajectories.

    Convenience wrapper over `compute_selected_worst` that applies C to the
    relative trajectory.  The solvers inline this in their hot loop; use it in
    analysis code and figures so the worst case shown is the same quantity the
    solver optimized against.

    Parameters
    ----------
    x_i, x_j : ndarray, shape (H+1, sdim)
        Full state trajectories of the two agents.

    Returns
    -------
    zw   : ndarray, shape (H+1, m)
        Row 0 is filled with the unperturbed C z_0, which the kernel leaves at
        zero: a length-0 prefix gives the adversary no budget, so the worst case
        at t = 0 is the nominal separation.  The solvers slice row 0 away (x[:, 0]
        is pinned to x0, so its cost is constant), but analysis code and figures
        want the true value there rather than a spurious zero.
    lamb : ndarray, shape (H,)
    """
    sigma_all, P_all = selected
    zC = np.ascontiguousarray((C @ (x_i - x_j).T).T)
    zw, lamb = compute_selected_worst(zC, sigma_all, P_all, eps, H, C.shape[0])
    zw[0] = zC[0]
    return zw, lamb


def warm_up(selected, eps, H, m):
    """Trigger numba compilation of the kernel outside any timed region.

    Uses non-zero dummy data so the Newton loop actually runs; a zero start would
    take the lambda = 0 shortcut and leave part of the kernel uncompiled.

    Call this before any timed region: on a cold cache the first call to
    `compute_selected_worst` pays one to two seconds of LLVM compilation, which
    would otherwise land inside the first measured solve.

    Parameters
    ----------
    selected : tuple of ndarray
        `(sigma_all, P_all)` from `precompute_selected`.
    eps : float
        Adversary's per-step energy budget.
    H : int
        Horizon length.
    m : int
        Number of selected components, i.e. `C.shape[0]`.

    Returns
    -------
    None
    """
    dummy = np.ascontiguousarray(np.ones((H + 1, m)))
    compute_selected_worst(dummy, selected[0], selected[1], eps, H, m)
