# Large-scale workflow placement in serverless computing using integer nonlinear programming
Under revision for IEEE Access
## How to use/Evaluate the results

a) Install the necessary dependencies:

```pip install -r requirements.txt```

b) Simple example:
Run the placement problem

```placement.py --num_run=0```

Calculate the costs through simulation by running

```selection_simulation.py --num_run=0```

c) Scaling experiments:

For the first comparison against the heuristic run:

```scaling_experiments.py```

In the command line run

```run_time_case_study_heuristic.bat```

To visualize or show the data run

```visualization_time_heuristic.py num_run=123```

Change "heuristic" in paramters_1.json to false

For the second comparison against the centralized approach run:

```scaling_experiments.py```

In the command line run

```run_time_case_study.bat```

To visualize or show the data run

```visualization_time.py num_run=126```

d) Sensitivity to weights

To generate the different parameter files run

```generate_different_weights.py```

In the command line run

```run_case_study.bat```

To generate the images of the Paerto fronts run

```visualization.py --num_run=2```

e) Sensitivity to uncertain parameters (demand, running time, latency, RAM)

The placement is optimized against the assumed parameter values. To test how
robust a computed placement is when the actual values differ, first run the
placement for a run as usual:

```placement.py --num_run=N```

Then configure the `sensitivity_config` block in `settings/parameters_N.json`.
Each uncertain parameter (`request_rate`, `time`, `speedup`, `latency`, `ram`)
can independently be set to `"mode": "noise"` (relative noise of `noise_pct`
around the baseline) or `"mode": "scale"` (a fixed `factor`), or `"off"`. Set
`num_scenarios` and a `seed` for reproducibility.

Generate the scenario input files and evaluate the fixed placement against each:

```generate_sensitivity_scenarios.py --num_run=N```

```run_sensitivity.py --num_run=N```

The scenario inputs are written to `sensitivity/run_N/scenario_<i>/` (only the
perturbed cloud/workflow requirement files; everything else is inherited from
the baseline run). The placement solution files (`placement/run_N/opt_*.json`)
are never modified. Latency is perturbed via the `L` matrix that `placement.py`
writes into the cloud requirement files after solving, so the placement
decision stays fixed. Aggregated results are written to
`results/run_N/sensitivity/summary.json` and `summary.csv`.

To attribute the effect of each individual uncertainty, run a one-factor-at-a-
time scale sweep (each parameter is varied alone across a set of scale factors):

```run_scale_sweep.py --num_run=N```

This generates its own scenarios under `sensitivity/run_N/scale_sweep/` and
writes `results/run_N/sensitivity/scale_sweep_summary.json` (optional
`--factors=0.5,1.0,2.0` and `--params=latency,time`).

The sweep evaluates the *fixed nominal* placement under wrong parameters. To
also see how it compares against re-planning, add `--adapt`:

```run_scale_sweep.py --num_run=N --adapt```

For every sweep point this additionally re-solves the placement (and the
centralized/heuristic comparison, per the run's settings) with the parameters
adapted to the perturbed values, inside a self-contained sandbox
`sensitivity/run_N/scale_sweep/<param>/factor_<tag>/adapted/` (mirroring the
usual `settings/cloud/workflows/placement/results` layout, so the baseline run
is never touched). The adapted costs are added to the summary as
`adapted_overall_cost` / `adapted_overall_cost_comparison`. Note: this runs one
full placement solve per sweep point (with a centralized comparison that is a
full-architecture MIQP each time) and stores one full run folder per point, so
use it with a small `--factors`/`--params` selection on large runs. The
adapted run at factor 1.0 reproduces the baseline exactly (sanity check).

To plot the study run

```visualization_sensitivity.py --num_run=N```

which saves a tornado chart, per-parameter response curves (with the adapted
placement overlaid as dashed lines when `--adapt` data exists), an
adaptation-offset chart (absolute and relative regret of keeping the nominal
placement instead of re-planning), a distribution overview of the combined
study, and a regression-based estimate of the individual effects to
`results/run_N/`. Each figure covers both the decomposed and the comparison
solution, and the titles state whether the comparison is the centralized
optimization or the heuristic (read from `settings/parameters_N.json`).

## Optional model flags

These keys in `settings/parameters_N.json` switch on model variants. **Every one
defaults to the behaviour of the original model, so a parameter file that omits
them reproduces the published runs.** Placements are not bit-comparable across
different settings.

| key | values | default | what it does |
|---|---|---|---|
| `stochastic_selection` | `true` / `false` | `false` | Relaxes the selection `H_{k,n}` from a one-hot choice to a stochastic routing policy in `[0,1]`, and samples each request's deployment from it during simulation. Full description in [`stochastic_selection.md`](stochastic_selection.md). |
| `ram_eligibility` | `true` / `false` | `false` | Forbids placing a function on a node whose subtree contains no physical node large enough to host it. A necessary condition for any physical realisation, so the decomposition stays a relaxation. See [`ram_capacity_model.md`](ram_capacity_model.md). |
| `ram_capacity_norm` | `"off"` / `"l1"` / `"l2"` | `"off"` | Capacity used to normalise the node utilization `M_i`, restricted per function to the subtree nodes that can host it. `"off"` keeps the aggregate `RAM_max,i`; `"l2"` is the value that makes an aggregated node's squared utilization match the smallest total its children can achieve. Only affects levels above the physical one. See [`ram_capacity_model.md`](ram_capacity_model.md). |

Note on comparing runs: the placement problem is a nonconvex MIQCQP solved to
`MIPGap 0`, and when several placements attain the same optimum the one Gurobi
returns is arbitrary. Re-solving the *identical* model under different Gurobi
`Seed` values moved the simulated cost by up to 236 on `run_141` and 1347 (9.3%)
on `run_140`. Differences below that scale between two model variants are not
evidence; compare objective values or seed-replicated ranges instead.

## Changelog / reproducibility notes

**2026-08 — RAM capacity model (`ram_eligibility`, `ram_capacity_norm`).** Above
the last level a "node" is a whole region, and `generate_cloud` Dirichlet-splits
a parent's RAM over its children, so the aggregated capacity is the *sum* of the
subtree while a function must fit on *one* physical node. On `run_141` region 2
sums to 265.1 but its largest node is 52.3, so it can host only 22 of 48
functions while the top level scores it as 265.1 of headroom; 16–18% of the
simulated waiting time came from functions placed where no node could ever hold
them. Two independently gated options address this — a hard per-function
eligibility constraint and a per-function capacity norm for `M_i` — both
defaulting to the previous behaviour. Each node's cloud config gains a
`ram_leaf_profile` field. Rationale, derivation and limits in
[`ram_capacity_model.md`](ram_capacity_model.md).

Fixed unconditionally in the same change: `M_i` used the bare execution time
`Time[m]` while the function-utilization penalty `u_{n,m}` used
`Speed[m][i]*Time[m]`, so `M_i` overstated the RAM held on cloud nodes by
`1/speedup`. Objective-neutral in practice (cloud `RAM_max` is 1e8; measured
worst relative objective change 3.7e-9 over 50 subproblems), but it does change
which of several equal-optimum placements the solver returns.

**2026-08 — simulator and accumulator fixes.** Six pre-existing defects were
corrected: (A) the node-occupancy test ignored booking start times, so
executions scheduled in the future blocked the present; (B) `update_cloud_config`
added `M[i]` to the utilization although `M[i]` already contains it, doubling the
baseline with every workflow; (C)/(D) branch-successor lookups compared a
function id against a list length; (E) the `lazy_start` branch recorded absolute
completion times instead of response times; (F) branch traversal used ascending
function ids from the membership matrix `b` rather than the execution order in
the workflow config's `paths`, which is non-monotone for half the generated
workflows. Results from before these fixes are not comparable.

**2026-07 — objective index fixes in `solve_opt` (optimization.py).** Two
indexing errors in the objective were fixed: (1) the quadratic function-
utilization penalty used function 0's placement row for every function of a
deployment (now `P[n*num_functions+m, i]`, matching the documented
u_{n,m} formula); (2) the money cost of returning the result to the user
indexed the data-transfer tensor `D` with the leaf *counter* instead of the
leaf's node id (now `D[m, i, leave_nodes[k]]`; only full/centralized and
top-level subproblems were affected). Placements solved after these fixes are
NOT bit-comparable with runs produced before them; regenerate baselines and
`--adapt` sweeps before comparing against new results.

**2026-07-16 — cached JSON reads in `selection_simulation.py`.** The cost
loops re-read the same placement/cloud/workflow files once per request,
function and hierarchy level (millions of `open()` calls on multi-level runs
— prohibitively slow on network drives). Input files are now served from an
in-process cache (`load_json_cached`). Correctness: every access re-validates
the entry against the file's `(st_mtime_ns, st_size, st_ino)`, and — because
the Windows file-time clock ticks coarsely — a matching stamp is only trusted
once the file's mtime is older than a 5 s grace window; younger or changed
files are always re-parsed. Verified bit-identical `results.json` and
`detailed_simulation_run_N.json` on run 135 (baseline and scenario sweep
point); JSON opens dropped from 18,054 to 85 there.
