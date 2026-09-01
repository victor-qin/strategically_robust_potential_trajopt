# Strategically Robust Potential Trajectory Optimization

Reference implementation for multi-agent trajectory optimization in which each
agent's collision cost is evaluated against a **worst-case adversarial
perturbation** of the others — at every step of the horizon, not just at the end.

<p align="center">
  <img src="figures/8agent_trajectories.png" width="380"
       alt="Eight agents swapping to antipodal points, nominal vs strategically robust">
</p>

The claim the code supports: strategic robustness is not the same as turning up
the collision penalty. Both buy separation, but they buy different separation, in
different places, at different cost to the plan.

## Results

Three methods on the paper's scenarios, at disturbance budget `eps = 2`. The
collision weight is `c1 = 2` except for **wider**, which is the same solver at
`c1 = 5` — the naive way to buy margin. Path deviation is the area between a
method's trajectory and the nominal one, averaged over agents.

**Head-on, 2 agents.** Wider buys more absolute worst-case margin, and pays more
than twice the distortion for it.

| method | min separation | min separation, worst case | path deviation |
|---|---|---|---|
| nominal | 0.706 | 0.258 | — |
| wider | 1.162 | **0.762** | 0.516 |
| strategically robust | 0.965 | 0.530 | **0.252** |

**Four agents.** With six interacting pairs the ordering reverses, and
robustness wins on both axes at once — a larger worst-case margin for less than
half the distortion.

| method | min separation | min separation, worst case | path deviation |
|---|---|---|---|
| nominal | 0.965 | 0.147 | — |
| wider | 1.115 | 0.221 | 0.953 |
| strategically robust | 1.075 | **0.279** | **0.405** |

A wider penalty pushes every pair apart uniformly, including pairs that were
never in danger; the robust solve spends its distortion where the adversary would
actually attack. Note that on *mean* worst-case separation wider still looks
better at four agents — averaging over six pairs lets slack in five cover a tight
sixth. `notebooks/collision_example.ipynb` shows both statistics side by side.

**Eight agents**, antipodal swap on a circle — 28 pairs, all conflicting at once.
This is where the gap is widest: the nominal plan leaves the tightest pair 0.023
apart under the worst case, effectively a contact, and the robust plan holds
0.186.

| method | min separation | min separation, worst case | path deviation |
|---|---|---|---|
| nominal | 0.432 | 0.023 | — |
| strategically robust | 0.469 | **0.186** — 8.2x | 0.421 |

### Runtime

Median over 5 runs (3 at eight agents); CV ≤ 3.4% on every cell, and every cell's
runs converged to a single optimum. The robust timing **includes its own nominal
initialization**, so the ratio is the end-to-end cost of switching methods, not
the marginal cost of the robust solve.

| scenario | agents | nominal | wider | strategically robust |
|---|---|---|---|---|
| Head-on | 2 | 0.213 s | 0.239 s (1.12x) | 0.305 s (**1.43x**) |
| Parallel | 2 | 0.061 s | 0.088 s (1.45x) | 0.103 s (**1.69x**) |
| 4 agents | 4 | 0.902 s | 1.066 s (1.18x) | 1.699 s (**1.88x**) |
| 8 agents | 8 | 41.66 s | 26.99 s (0.65x) | 52.00 s (**1.25x**) |

Two things in that table are worth not glossing over. The robust overhead
*shrinks* as the problem grows — 1.88x at four agents, 1.25x at eight — because
the worst-case oracle costs O(pairs) while the nominal solve's own difficulty
grows faster. And at eight agents the wider penalty is **faster than nominal**
(0.65x): the stronger barrier conditions the problem better, converging in 85
iterations against 131.

### Single shooting

Solving over the controls alone eliminates every equality constraint — 336 of
them at eight agents — and roughly halves the variable count:

| solve | full space | single shooting | |
|---|---|---|---|
| nominal, 8 agents | 41.73 s | 2.11 s | 19.8x |
| strategically robust, 8 agents | 10.31 s | 1.04 s | 9.9x |

The two parameterizations start from different points and reach different local
minima here (nominal objective 48.36 full-space against 36.69 shooting — shooting
found the better one), so compare the objectives alongside the times.

## Layout

| File | Contents |
|---|---|
| `costs.py` | Inter-agent distance costs, each paired with its analytic derivative. |
| `adversary.py` | The worst-case oracle: selected-state closed form, Newton solve for the multiplier, numba kernel. numpy and numba only. |
| `solvers.py` | The four solver entry points — nominal / robust, full-space / single-shooting. |
| `plotting.py` | Publication figures and trajectory animations. Nothing else imports it. |
| `plot_style.py` | Two-column conference style and palette. |
| `benchmark_tables.py` | Runtime and path-deviation tables. Runs headless. |
| `notebooks/collision_example.ipynb` | The worked example: 2, 4 and 8 agents, every figure and table. |
| `notebooks/collision_example.py` | Cell source for the above in `# %%` format — the file of record, since it reviews as a normal diff. `notebooks/build.py` regenerates the notebook from it. |

The solver imports only numpy, scipy and numba: `benchmark_tables.py` runs on a
headless box with no plotting stack installed.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt                  # solver + benchmarks
pip install -r requirements-notebook.txt         # figures + notebook
```

Python 3.10–3.12. numba 0.63 has no wheels past 3.12, and its kernel is on the
hot path.

## Reproduce

```bash
python benchmark_tables.py --runs 30                    # the tables above
python benchmark_tables.py --runs 30 --shooting         # single-shooting variant
python benchmark_tables.py --scenarios "8 agents" --runs 10
python notebooks/build.py --execute                     # every figure, from source
jupyter lab notebooks/collision_example.ipynb
```

## A note on gradients

Every solver takes an optional `distance_cost_grad_func`, and every caller here
supplies one. This is not only a speed question. Without it SLSQP finite-
differences the objective (~100x the function evaluations) *and* tends to stop
early on a worse optimum — comparing a robust solve against such a baseline
inflated the measured 8-agent overhead from 1.29x to 3.72x. `costs.make_cost`
returns the cost and its derivative together so the pair cannot be separated by
accident.

The paper derives the adversary's closed form (Appendix B) but not the gradient
of the outer objective, which is the part that is easy to get wrong: the obvious
route differentiates through the Riccati recursion, and the envelope theorem says
not to. That derivation, and the earlier version of it that was wrong by order
10, are in the `adversary.py` module docstring.

## License

MIT. See `LICENSE`.
