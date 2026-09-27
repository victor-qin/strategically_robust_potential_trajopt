# Strategically Robust Potential Trajectory Optimization

Multi-agent trajectory optimization in which each agent's collision cost is
evaluated against the **worst-case perturbation** of the other agents, at every
step of the horizon.

<p align="center">
  <img src="figures/8agent_trajectories.png" width="380"
       alt="Eight agents swapping to antipodal points, nominal vs strategically robust">
</p>

Strategic robustness is not the same as a larger collision penalty. Both buy
separation, but in different places and at a different cost to the plan.

## Results

Three methods at disturbance budget `eps = 2`. **Nominal** and **strategically
robust** use collision weight `c1 = 2`. **Wider** is the nominal solver at
`c1 = 5`, the naive way to buy margin. Path deviation is the area between a
method's trajectory and the nominal one, averaged over agents. All numbers here
come from the full-space solver.

**Head-on, 2 agents.** Wider buys more worst-case margin, at twice the path
deviation.

| method | min separation | min separation, worst case | path deviation |
|---|---|---|---|
| nominal | 0.706 | 0.258 | — |
| wider | 1.162 | **0.762** | 0.516 |
| strategically robust | 0.965 | 0.530 | **0.252** |

**Four agents.** With six pairs the ordering reverses: robust gets the larger
worst-case margin with less than half the path deviation.

| method | min separation | min separation, worst case | path deviation |
|---|---|---|---|
| nominal | 0.965 | 0.147 | — |
| wider | 1.115 | 0.221 | 0.953 |
| strategically robust | 1.075 | **0.279** | **0.405** |

Wider pushes every pair apart, including pairs that were never in danger.
Robust spends its deviation where the adversary would attack. On *mean*
worst-case separation wider still looks better, because slack in five pairs
hides a tight sixth. The [notebook](notebooks/collision_example.ipynb) shows
both statistics.

**Eight agents** on a circle, each heading for the opposite point, so all 28
pairs conflict at once. Under the worst case the nominal plan leaves the closest
pair 0.023 apart, effectively a contact. The robust plan keeps 0.186.

| method | min separation | min separation, worst case | path deviation |
|---|---|---|---|
| nominal | 0.432 | 0.023 | — |
| strategically robust | 0.469 | **0.186** | 0.421 |

### Runtime

Full-space solver, median of 5 runs (3 at eight agents). Every cell's runs
converged to a single optimum. The robust time includes its nominal
initialization, so each ratio is the end-to-end cost of switching methods.

| scenario | agents | nominal | wider | strategically robust |
|---|---|---|---|---|
| Head-on | 2 | 0.224 s | 0.251 s (1.12x) | 0.319 s (**1.43x**) |
| Parallel | 2 | 0.067 s | 0.081 s (1.20x) | 0.108 s (**1.61x**) |
| 4 agents | 4 | 0.903 s | 1.068 s (1.18x) | 1.731 s (**1.92x**) |
| 8 agents | 8 | 42.24 s | 27.34 s (0.65x) | 52.53 s (**1.24x**) |

The robust overhead *shrinks* as problems grow, from 1.92x at four agents to
1.24x at eight. The ratio tracks `1 + robust_iterations / nominal_iterations`,
so it is highest where the nominal problem is easy: parallel takes 10 nominal
iterations and 6 robust, while eight agents takes 131 and 32. At eight agents
wider is even faster than nominal, because the stronger barrier converges in 85
iterations instead of 131.

### Single shooting

Optimizing over the controls alone removes every equality constraint (336 at
eight agents) and halves the number of variables.

| solve, 8 agents | full space | single shooting | speedup |
|---|---|---|---|
| nominal | 41.84 s | 2.10 s | 19.9x |
| strategically robust | 10.22 s | 0.94 s | 10.8x |

The two start from different points. On two and four agents they reach the same
optima. At eight agents shooting finds a better one, with nominal objective
36.69 against 48.36, and its robust worst-case margin rises to 0.235.

Robust *ratios* rise under shooting on parallel and eight agents. In the full
space the constrained QP, a cost both methods share, takes nearly all of every
iteration and hides the robust evaluation. Shooting removes that shared cost.

## Layout

| file | contents |
|---|---|
| `costs.py` | Pairwise distance costs, each with its analytic derivative. |
| `adversary.py` | The worst-case oracle: closed form, Newton solve for the multiplier, numba kernel. |
| `solvers.py` | Nominal and robust solvers, each in full-space and single-shooting form. |
| `benchmark_tables.py` | Runtime and path-deviation tables. Needs only the solver dependencies. |
| `plotting.py`, `plot_style.py` | Figures, animations and their shared style. The solver never imports them. |
| `notebooks/collision_example.ipynb` | The worked example, with every figure and table for 2, 4 and 8 agents. |
| `notebooks/collision_example.py` | Its source in `# %%` cell format, and the file of record. `notebooks/build.py` rebuilds the notebook from it. |

## Install and reproduce

Python 3.10 to 3.12. numba 0.63 has no wheels past 3.12.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt                  # solver and benchmarks
pip install -r requirements-notebook.txt         # figures and notebook

python benchmark_tables.py --runs 30             # runtime tables, full space
python benchmark_tables.py --runs 30 --shooting  # single-shooting variant
python benchmark_tables.py --scenarios "8 agents" --runs 3
python notebooks/build.py --execute              # notebook and every figure
```

## License

MIT. See `LICENSE`.

---

## Appendix: the outer gradient

The paper gives the adversary's closed form in Appendix B, but not the gradient
of the outer objective, which is what SLSQP consumes. That gradient is easy to
get wrong, so the derivation is recorded here.

### The inner problem

For a pair $(i,j)$ and prefix length $N$, the adversary perturbs the relative dynamics with energy budget $\varepsilon$ to close the distance at step $N$. Write $z_N = x_{i,N} - x_{j,N}$ for the relative state, $G_N = \sum_{l<N} A^l B B^\top (A^\top)^l$ for the reachability Gramian, and $C$ for the selection matrix picking the collision-relevant components. Then

$$
C z^{\mathrm{worst}}_N = C z_N - \Gamma_N \left( \Gamma_N + \lambda_N I \right)^{-1} C z_N,
\qquad \Gamma_N = C G_N C^\top,
$$

with $\lambda_N \ge 0$ fixed by the secular equation

$$
\sum_l \frac{\sigma_l w_l^2}{(\lambda_N + \sigma_l)^2} = \varepsilon^2,
$$

where $\sigma_l$ are the eigenvalues of $\Gamma_N$ and $w$ holds the coordinates of $C z_N$ in its eigenbasis. A root $\lambda_N > 0$ exists iff $\sum_l w_l^2 / \sigma_l > \varepsilon^2$; otherwise $\lambda_N = 0$ and the adversary reaches contact exactly. Implemented in `adversary.compute_selected_worst`.

### The gradient

The robust cost for that pair and prefix is $f(\lVert C z^{\mathrm{worst}}_N \rVert)$, where $z^{\mathrm{worst}}_N$ depends on $z_N$ through the inner optimization. By the envelope theorem, the adversary's optimal controls $\delta u^\star$ can be held fixed when differentiating. The perturbation $\delta z_N$ then comes only from propagating $\delta u^\star$ through the dynamics, so it does not depend on $z_N$, and $\mathrm{d} z^{\mathrm{worst}}_N / \mathrm{d} z_N = I$.

The gradient is therefore the nominal formula, evaluated at the worst-case position. With $d_{\mathrm{worst}} = \lVert C z^{\mathrm{worst}}_N \rVert$,

$$
\frac{\partial f}{\partial x_{i,N}} = + f'(d_{\mathrm{worst}}) \frac{C z^{\mathrm{worst}}_N}{d_{\mathrm{worst}}},
\qquad
\frac{\partial f}{\partial x_{j,N}} = - f'(d_{\mathrm{worst}}) \frac{C z^{\mathrm{worst}}_N}{d_{\mathrm{worst}}}.
$$

`solvers._make_everystep_costs` pulls this back through $C$, so it fills the position components for $C = [I_p \ \ 0]$ and stays correct for a weighted $C$.

**The pitfall.** An earlier derivation used $\mathrm{d} z^{\mathrm{worst}}_N / \mathrm{d} z_N = I + P_N$, where $P_N$ is the Jacobian of the map from $z_N$ to $\delta z_N$. That is the **total** derivative, which lets $\delta u^\star$ move with $z_N$. The envelope theorem calls for the **partial** derivative at fixed $\delta u^\star$, which is $I$. Against finite differences the $I + P_N$ gradient had errors of order 10, while the corrected one agrees to about $10^{-7}$.

### Why every solve gets an analytic gradient

Every solver takes an optional `distance_cost_grad_func`, and every caller here supplies one. `costs.make_cost` returns the cost and its derivative together so the two cannot be separated by accident. Without the derivative SLSQP finite-differences the objective. On the 2-agent head-on scenario, full space:

| solver | finite difference | analytic gradient |
|---|---|---|
| nominal | 5129 function evals | **50** |
| strategically robust | 2155 function evals | **23** |

Both reach the same objective here. On larger scenarios the finite-difference solve tends to stop early at a worse optimum, which distorts any runtime comparison against it.

### Constraint Jacobian

The full-space dynamics constraints are linear, so their Jacobian is constant and block-diagonal per agent:

$$
\frac{\partial c_{\mathrm{IC}}}{\partial x_{i,0}} = I,
\qquad \frac{\partial c_{\mathrm{dyn}}}{\partial x_{i,t+1}} = I,
\qquad \frac{\partial c_{\mathrm{dyn}}}{\partial x_{i,t}} = -A,
\qquad \frac{\partial c_{\mathrm{dyn}}}{\partial u_{i,t}} = -B,
$$

and zero elsewhere. `_dynamics_constraint` leaves it to SLSQP's finite differences to match the published runs. Single shooting removes these constraints entirely.
