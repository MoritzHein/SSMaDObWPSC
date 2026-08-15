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

f) RAM capacity model x utilization weight x selection policy

Compares the deterministic and the stochastic selection policy across the
utilization weight `w_3`, once with the RAM capacity model switched off and once
with it on, on the standard case-study architecture (2 levels, 10 workflows,
10-11 nodes per region, i.e. 40 physical nodes and 20 leaves).

Generate the parameter files:

```ram_w3_experiments.py```

This writes **`settings/parameters_300.json` ... `settings/parameters_323.json`**
(24 runs) plus the manifest `settings/ram_w3_campaign.json`, from the template
`settings/parameters_1.json`. The grid is

| axis | values |
|---|---|
| RAM capacity model | `off` (`ram_eligibility=false`, `ram_capacity_norm="off"`) and `l2` (`true`, `"l2"`) |
| `w_3` | 0.5, 1.0, 1.5, 2.0, 2.5, 3.0 |
| selection policy | deterministic, stochastic (`stochastic_selection`) |

The horizon is a knob of the study rather than a template inheritance:
`SIMULATION_TIME` in `ram_w3_experiments.py` (currently **400**) is written into
every generated parameter file and into the manifest. It matters more than it
looks. The queue does not reach a steady state at this arrival rate — waiting per
request keeps rising to the end of the arrival stream, about 2.75x faster under
the deterministic policy — so the *cost gap between the policies is a function of
the horizon* and does not converge: on the `l2`, `w_3=1.5` cell it is −0.9% at
`simulation_time=200`, −1.6% at 400, −3.4% at 800 and −8.5% at 1600. Quote the
horizon whenever quoting a cost difference, and prefer the horizon-independent
backlog slope from `visualization_waiting_over_time.py` (below) for claims about
which policy is better. The old value of 200 was additionally drain-dominated:
the last request finished around t=370, so 44% of requests completed after
arrivals had stopped.

Because `simulation_time` enters only `selection_simulation.py` and
`evaluate_comparison_objective.py` — the optimization objective uses the rates
`lambda`, not `lambda * T` — changing it needs **no re-solve**. Regenerate the
parameter files and re-run `resimulate_seeds.py` over the existing placements.

with the run number of a cell given by
`300 + ram_index*12 + w3_index*2 + policy_index` — runs 300-311 are the
RAM-model-off half and 312-323 the `l2` half. Nothing downstream hard-codes that
arithmetic; the manifest carries the mapping.

In the command line run

```run_ram_w3_study.bat```

which solves and simulates all 24 runs and then calls

```resimulate_seeds.py --seeds=10```

to re-simulate every placement on ten independent request streams *without*
re-solving (a single simulation at `simulation_time=200` carries roughly 10%
Monte-Carlo noise, which is larger than several of the effects being compared).
The per-stream costs land in `results/run_N/results_seeds.json` and the
per-request response and waiting times of the same streams in
`results/run_N/latency_seeds.json`.

The seeds are what makes the policy comparison work. Both policies are simulated
on the *same* request streams — `selection_simulation.py` draws the Poisson
arrivals from `rng` and the per-request deployment from a separate `rng_sel`
precisely so the arrival stream is unaffected by the policy — so the comparison
is paired and the per-stream *difference* is the statistic with the small
variance. Its confidence interval is roughly four times narrower than the
interval on either policy alone, which matters because the policy effect
(2-3% of the overall cost) is smaller than the Monte-Carlo spread of a single
configuration (~6%).

No solver parameter needs to be touched: the generated files inherit
`own_gap = 0` and `comparison_gap = 0` from the template and set no time limit,
so every subproblem is solved to proven optimality. Budget hours rather than
minutes for the twelve stochastic runs — relaxing `H` turns the subproblems into
nonconvex MIQCQPs, and a stochastic run takes upwards of ten minutes against
about twenty seconds for a deterministic one. The achieved gap of every
subproblem is recorded in `results/run_N/time.json` under `own_mipgap`; with the
default settings these are all zero. Setting `OWN_TIME_LIMIT` in
`ram_w3_experiments.py` trades that guarantee for speed and makes the solutions
incumbents rather than optima — use it for scanning only.

To generate the figure run

```visualization_ram_w3.py```

which saves two figures to `results/run_300/`.

`ram_w3_comparison.pdf/.png` has four rows, an absolute-cost row and a
difference row per RAM capacity model (off on top, eligibility + l2 below), with
`w_3` on the x-axis. In an absolute row each `w_3` carries a deterministic and a
stochastic bar, stacked into `w_1*M + w_2*B + w_2*W` (monetary, basic time,
waiting) so its height is the overall cost; the measured `overall_cost` is drawn
as a separate marker with a 95% confidence interval over the request streams.
The difference row below plots the *paired* difference stochastic − deterministic
for the same cells, decomposed into the same three contributions, with the paired
95% confidence interval of the overall cost and each cell's relative effect and
seeds-won count. Error bars therefore mean different things in the two kinds of
row — unpaired CI of one configuration above, paired CI of the difference below —
and only the lower one resolves the policy effect.

`ram_w3_latency.pdf/.png` shows the mechanism: the response-time CDF of one
representative cell (`--cell=l2,1.5` by default), deterministic against
stochastic, pooled over all request streams, plus its p50/p90/p95/p99 tail.

The script also prints the numbers behind both figures.

g) Waiting time over the course of a run

```visualization_waiting_over_time.py --runs=316,317```

Buckets every request by its **arrival** time and plots the mean waiting time of
each bucket against elapsed simulation time, pooled over all request streams with
a 95% confidence band. It answers what the cost bars cannot: a flat line means the
queue has settled and the cost numbers measure a steady state; a rising line means
the backlog is still building when the simulation stops, so total waiting grows
faster than linearly in `simulation_time` and every cost number depends on the
horizon.

The fitted slope printed in the legend — seconds of waiting added per 1000 s of
elapsed time — is the horizon-independent way to compare two policies, since it
does not depend on where the simulation was stopped. On the standard architecture
the deterministic policy adds about 40 s per 1000 s against about 15 s for the
stochastic one.

`--runs` takes a single run or any comma-separated list, so it draws one curve or
overlays several; the intended use is the deterministic and stochastic run of the
same cell. Labels come from `settings/ram_w3_campaign.json` when the runs belong
to that campaign. `--window` sets the bucket width (default `simulation_time/8`)
and `--export` the output folder (default `results/run_<first run>`). Without a
`latency_seeds.json` it falls back to the single baseline stream, which is noisy
enough to show a dip that looks like the queue recovering — run
`resimulate_seeds.py` first.

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
| `selection_policy` | block, see below | absent | Blocking-aware selection. Restricts each request's draw to the deployments that could start it now. **Simulation only** — it never changes the placement, so a variant can reuse a solved run. |
| `comparison_mode` | `"centralized"` or one of `optimization.HEURISTIC_MODES` | derived from `heuristic` | What the comparison run evaluates. `heuristic: true` maps to `"all_cloud"` and `false` to `"centralized"`, so older parameter files are unchanged. |

### `selection_policy`

```json
"selection_policy": {
    "candidates": "support_then_placed",
    "probe":      "entry",
    "rule":       "free_sample",
    "instrument": false
}
```

`candidates` is the option set: `"support"` = `{n : H_{k,n} > 0}`, `"placed"` =
every deployment whose functions are all placed, `"support_then_placed"` widens
beyond the support only when nothing in it is free. `rule` is one of
`open_loop`, `free_sample`, `free_argmax_h`, `free_uniform`, `least_loaded`,
`power_of_two`, `deficit_free`. `instrument` records how often a rule had
anything to decide into `results.json` under `selection_stats` without changing
the draw, so an instrumented open-loop run reproduces its plain counterpart
bit-for-bit.

Two things are worth knowing before using this:

* Under a **deterministic** placement the support of `H` holds a single
  deployment — selection is not recomputed below the top level, so a whole
  region shares one — and every rule over `"support"` is a no-op. Runtime
  recourse on a deterministic placement has to use `"placed"`.
* The rules draw from `rng_sel` only, never from `rng`, so the arrival stream
  and the paired comparison across policies are unaffected.

Campaign: `policy_experiments.py` writes **runs 500–607** (2 memory models ×
6 values of w₃ × 9 selection variants) plus **runs 620–625** (the model-free
reference placements), and `settings/policy_campaign.json`. The policy runs clone
`cloud/`, `workflows/` and `placement/` from the RAM × w₃ study — **nothing is
re-solved**, which also rules out the solver returning a different optimum of
equal objective. They set `comparison: false`, because the model-free baselines
depend on neither w₃ nor the memory model nor the selection rule, and are
therefore simulated once each instead of in all 108 cells.

`run_policy_study.sh [MAXJOBS]` drives the whole study: generate, solve only
620–625, simulate every run once, `resimulate_seeds.py` for the ten paired
streams, then `visualization_policy.py --paper --latex` for the tables and the
two figures (`policy_bars.pdf`, `policy_w3.pdf`).

### Model-free comparison baselines

`all_cloud` (the previous default) places every function on node 0 and sends
every leaf to deployment 0, so the whole architecture's traffic for a workflow
serialises through one function instance. `all_cloud_rr` keeps the placement but
spreads the leaves; `local_first`, `best_fit_edge`, `greedy_cost` and
`random_feasible` are the edge-first, bin-packing, greedy and random references.
All of them honour the eligibility condition `RAM_m <= RAM^node_i` and none
solves anything.

Note on comparing runs: the placement problem is a nonconvex MIQCQP solved to
`MIPGap 0`, and when several placements attain the same optimum the one Gurobi
returns is arbitrary. Re-solving the *identical* model under different Gurobi
`Seed` values moved the simulated cost by up to 236 on `run_141` and 1347 (9.3%)
on `run_140`. However, this is mostly important when w_3=0.
Differences below that scale between two model variants are not
evidence; compare objective values or seed-replicated ranges instead.

## Changelog / reproducibility notes

**2026-08 — the head node's outside-demand placeholder (Eq. (8)).** Two defects
made the placeholder that `generate_cloud` writes into every region's head node
much larger than the demand that actually reaches the region from outside.
(A) `request_overall = sum(request_list[-2])` summed the whole previous level and
was **never reduced by the region's own demand**, so it was identical for every
region: on `run_316` all four regions carry the placeholder 0.1205211843 although
their own rates are 0, 0.03509, 0.04733 and 0.03810, and the three-level
`run_332` shows that one value across every region of every level. (B) that sum
also counted rates parked on nodes that are not user leaves — `request_list[0]`
was `[0, 0.03509 (google), 0.04733, 0.03810]` = 0.12052, but only tinyfaas leaves
originate simulated requests, so the demand that exists is 0.08543. The latter is
the value that reproduces the paper's own figure, 0.08543 × 400 s × 10 workflows
= 342 requests. Together the head of region `1_2` was weighted 0.12071 in a
deterministic subproblem where the correct weight is 0.00019 + 0.03810 = 0.03829,
a factor 3.15.

Only the deterministic path was affected. The stochastic decomposition already
used the right quantity, since `own_head` is `request_rate - virtual_request_rate`
and the external stream is weighted by `outside_rate` from
`_stochastic_selection_targets`, and both are invariant under the fix. The
flattened (centralized) config carries no placeholder, and the top-level problem
has none either.

Fixed by three edits in `cloud_workflow_setup.py`: a request rate is drawn only
for tinyfaas nodes at level 0; the level-0 root head node is given rate 0 to agree
with `request_list[0][0]`; and `request_overall` becomes
`sum(request_list[-2]) - request_list[-2][m]`. All `np.random.uniform` draws are
kept, in their original evaluation order, so the RNG stream and the rest of the
generated architecture are unchanged — a field-by-field config diff shows only
`request_rate`, `virtual_request_rate` and `utilization` moving.

Verified on scratch runs at two and three levels. Every region's placeholder now
equals both Λ − (own demand) and the recursion used by the stochastic path, 4
regions on two levels and 44 on three, zero mismatches; Λ = 0.08543276 is
identical at every level and equal to the total rate of the leaves the simulator
samples; the head weight of region `1_2` is 0.03829309 = 0.00019330 + 0.03809979;
and no `flow conservation skipped` warning appears. The self-check to keep is that
`virtual_request_rate` must equal `outside_rate` for every region, because the
recursion telescopes to Λ − λ_{l-1,m}.

Impact on stored results is small but non-zero. The stochastic runs are
bit-identical. Of the deterministic cells re-solved, the aggregate-capacity
w_3 = 0.5 cell reproduces `run_300` exactly and the eligibility w_3 = 1.5 cell
moves from 33108.05 to 33109.05 (+0.003%, entirely in `time_cost`, money and
waiting unchanged). Every deterministic result and every det-vs-stoch or
dec-vs-cen comparison should nevertheless be regenerated. Eq. (8) of the paper was
restated accordingly, with the originating demand and the outside rate as separate
symbols so that the placeholder can no longer be confused with a node's own rate.

**2026-08 — the `seed` in parameters_N.json now reaches the architecture.**
`generate_cloud` accepted a `seed` argument but opened with a hard-coded
`np.random.seed(42)`, so **every run produced the byte-identical architecture no
matter what `seed` said** — a multi-seed study varied only the workflows. Now
fixed. Verified: seeds 42-47 previously gave 1 distinct architecture, now 6
(total edge RAM 443.6 - 1710.2 on the 9-node example). Consequence: any stored
run whose `seed` is not 42 was solved on the seed-42 architecture; re-running it
now produces a different instance and its stored results are not comparable.

**2026-08 — the `seed` now reaches the workflows as well.** The companion half of
the same problem: `generate_random_workflows` built each workflow's DAG with
`generate_connected_graph_with_paths(..., seed=i)`, and that function opened with
`np.random.seed(seed)`. It never used the global stream itself (it builds its own
`default_rng`), so the call was dead code inside the function and pure collateral
damage outside — it reset the *caller's* stream to the workflow index. Only the
two draws before the first graph call responded to the run seed; workflows 3-5
were identical across all seeds and seeds 45/47 produced byte-identical workflow
sets. Fixed by removing the global reseed and drawing the DAG seed from the
generator's own stream. Verified: same seed twice is byte-identical; all 10 pairs
of seeds 43-47 now differ in **all five** workflows, and every workflow has a
distinct DAG topology per seed.

Together with the architecture fix above, `seed` now varies both the
infrastructure and the workload, so different seeds are genuinely independent
instances. Every generated workflow changes, so results produced before this are
not comparable.

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
