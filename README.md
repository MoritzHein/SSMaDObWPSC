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

`visualization_ram_w3.py --paper` instead writes the paper's compact version to
`UCC_MINLP/Figures/`: one column wide, two panels, all four combinations of
memory model and selection policy on a single pair of axes. Panel (b) reports
the paired difference **in per cent of the deterministic cost of the same
cell**, because in absolute terms the aggregate model at `w_3 = 0.5` is an order
of magnitude larger than the other eleven cells and flattens them.
`--paper-style markers` writes the alternative `ram_w3_paper_markers.pdf`, which
drops the bars for markers with error bars. A marker needs no zero baseline, so
that variant cuts the y-axis to the band the modes occupy and names the two
aggregate cells at `w_3 = 0.5` that fall outside it, which makes the few per
cent between the modes resolvable in the absolute panel as well.

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
| `arrival_trace` | block, see below | absent | Replaces the stationary Poisson arrival stream with per-minute intensities from the Azure Functions Trace 2019. **Simulation only** — the trace supplies the temporal shape and λ keeps the level, so no placement is re-solved. |

### `selection_policy`

```json
"selection_policy": {
    "candidates": "support_then_placed",
    "probe":      "entry",
    "rule":       "free_sample",
    "instrument": false
}
```

Full description of every candidate set, every rule and every model-free
baseline, with the measured numbers, in
[`selection_policies.md`](selection_policies.md).

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
figures. `policy_bars.pdf` and `policy_w3.pdf` are the diagnostic ones with all
eleven policies and all six references. `--paper` additionally writes two
column-wide alternatives, both restricted to the five policies of `PAPER_SERIES`
and the two references of `PAPER_HEURISTICS`:

* `policy_paper.pdf` — cost against `w_3` plus the paired difference to the
  deterministic open loop, eligibility model only (`PAPER_RAM`). Shows that the
  ordering does not depend on `w_3`.
* `policy_bars_paper.pdf` — one panel at a single `w_3` (`--w3`, default 1.5),
  horizontal bars stacked into `w_1*Z + w_2*B + w_2*W`, the aggregate model on
  top and eligibility + l2 at the bottom, each policy bar labelled with its
  paired difference to the open loop of its own block. Shows the cost
  composition and both memory models, at the price of a single `w_3`.

  `--heuristic-style` chooses how the model-free references appear.
  `row` (default) gives each one a bar of its own **between the two blocks**, so
  both blocks are adjacent to it and it carries the same cost split and the same
  interval as any other row. `rule` draws it as a vertical dashed line across
  all bars instead, which is denser and lets every bar be compared to it at a
  glance, but shows no split and no interval; the line is named in the legend,
  since it has no y tick of its own. Either way a reference too large for the
  axis is omitted rather than allowed to compress every other bar, so `edge
  first` at 53353 appears in neither variant and has to be quoted in the text.

  This needs the comparison cost split per stream
  (`money_cost_comparison` and friends in `results_seeds.json`), which
  `resimulate_seeds.py` records only for runs re-simulated after that field was
  added. Without it the reference is drawn as one plain grey bar, so a missing
  decomposition is visible rather than invented.

**The plotted basic time is not `time_cost`.** `time_cost` in `results.json` is
the *analytic* basic time — the critical path of the workflow times the number
of requests of that (workflow, leaf) pair, accumulated in the reporting block of
`selection_simulation.py`. It is about 3.7 % below the basic time the realized
cost is actually made of, so stacking it leaves the bar 0.89 % to 1.44 % short
of `overall_cost`. Money and waiting, by contrast, enter
`overall_cost = w_1*Z + w_2*sum(realized response times)` exactly as reported,
so the whole discrepancy sits in the basic time and the realized one is simply
the remainder:

```
w_2*B = overall_cost - w_1*Z - w_2*W
```

This is exact, not an approximation. The simulator records
`wait = max(0, response - base_T)`, so `sum(response) - sum(wait) = sum(base_T)`
unless a request finishes faster than its own base time — which happens in none
of the 138 campaign runs (it does in the old runs 123–137, at up to 0.3 %). All
stacked bars therefore use the remainder, sum exactly to the measured cost, and
are directly comparable to the model-free reference rules, which are also
`overall_cost`. Getting this wrong is not cosmetic: with the analytic basic
time the aggregate/`det_ol` bar at `w_3 = 1.5` stops at 30761 while its measured
cost is 31171 and the `all_cloud_rr` rule sits at 31052 between them, so the bar
and the marker disagree about which side of the baseline that configuration is
on.

### Model-free comparison baselines

`all_cloud` (the previous default) places every function on node 0 and sends
every leaf to deployment 0, so the whole architecture's traffic for a workflow
serialises through one function instance. `all_cloud_rr` keeps the placement but
spreads the leaves; `local_first`, `best_fit_edge`, `greedy_cost` and
`random_feasible` are the edge-first, bin-packing, greedy and random references.
All of them honour the eligibility condition `RAM_m <= RAM^node_i` and none
solves anything.

### `arrival_trace` — Azure Functions arrivals instead of stationary Poisson

```json
"arrival_trace": {
    "path": "settings/arrival_trace_azure_d01_hi.json",
    "window_minutes": 7,
    "normalize": "window",
    "scope": "leaf",
    "load_factor": 1.0
}
```

Absent or `null` keeps the stationary Poisson stream. When set, the per-second
draw becomes `rng.poisson(request_rates * shape[t])`, where `shape` is a
unit-mean per-minute intensity from the Azure Functions Trace 2019. **The trace
supplies the temporal shape and λ keeps the level**, so nothing is re-solved and
the results stay comparable with the Poisson runs. Build the shape file with
`build_arrival_trace.py --data_dir=<dir>` after extracting
`invocations_per_function_md.anon.d01.csv` from the release tarball;
`--measure_only` prints the burstiness evidence and writes nothing.

| key | values | meaning |
|---|---|---|
| `normalize` | `"window"` / `"day"` | `"window"` rescales the selected window to unit mean, so every stream realises the design load and burstiness is the only factor that changes against the Poisson control. `"day"` leaves each window at its own level. |
| `scope` | `"leaf"` / `"shared"` / `"workflow_leaf"` | how far the pattern is correlated across streams — see the gotcha below. |
| `window_minutes` | int | length of the slice normalized to unit mean. Must cover `simulation_time` and should barely exceed it — see the gotcha below. |
| `load_factor` | float | scales the simulated stream only; it does **not** re-solve, so anything but 1.0 is a deliberate load-mismatch experiment. |

Four things are easy to get wrong here.

*The dataset has no arrival timestamps* — only a count per function per minute.
Nothing below 60 s comes from the data, so no sub-minute claim is supportable.

*Correlation decides whether the trace changes anything, not the window length.*
A single (workflow, leaf) stream runs at ~0.26 invocations/minute, where the
trace's modulation (shape CV 0.22 over 30 minutes, 0.67 over a whole day) is far
below the stream's own Poisson noise (~1.96) at **every** window length.
Blocking, though, is driven by the aggregate load on a node, where Poisson noise
averages down as `1/sqrt(n)` and a shared pattern does not. Measured per-minute
CV of the architecture-wide arrival count: at `simulation_time = 1800` and a
30-minute window, Poisson 0.150, `"leaf"` 0.175, `"shared"` 0.593; at 400 s and
a 7-minute window, Poisson 0.135, `"leaf"` 0.143 (1.06x), `"shared"` 0.300
(2.22x). The mechanism is visible directly in the shapes: the twenty `"leaf"`
applications are mutually uncorrelated (mean pairwise correlation -0.02 to
+0.04 over the ten seeds), so their modulations partly cancel in the sum -
aggregate intensity CV 0.094 against 0.132 per leaf - while `"shared"` copies
one application to every leaf and cannot cancel against itself, leaving the
aggregate CV at the per-leaf 0.262. The cancellation is weaker than
`1/sqrt(20)` because the per-leaf rates are skewed: one leaf carries 24% of the
load and the effective number of independent leaves is **8.76, not 20**. Note
also that `"shared"` picks its application from the *seed* while `"leaf"` picks
by leaf index, so across ten streams the shared arm draws ten different
applications and its stream-to-stream spread is far larger.

*`window_minutes` must match the horizon, not the dataset.* `build_shape_series`
normalizes the selected window to unit mean and **then** truncates the
per-second series to `simulation_time`, so every minute that is normalized over
but never simulated decouples the realized load from the design lambda. At
`simulation_time = 400` a 30-minute window would leave the ten campaign streams
running at 0.113x-2.103x lambda under `"shared"` and 0.924x-1.223x under
`"leaf"` - a load-level lottery, not a burstiness experiment. `window_minutes =
7` (420 s, the shortest window still covering the horizon) restores it to
0.992-1.026 and 1.008-1.016 and costs almost no burstiness: the mean intensity
CV over the simulated 400 s is 0.262 either way for `"shared"` and 0.159 ->
0.132 for `"leaf"`. `trace_experiments.py` warns when more than a minute of the
window goes unsimulated.

*An application at our own rate has no shape to borrow.* At ~0.37
invocations/minute a 30-minute window holds about 11 invocations and many hold
none, so the series admits no unit-mean shape and the simulator raises. Summing
several quiet applications to fix the sparsity just reproduces a busy one —
sampling quality and rate are inseparable, which is why the shape must come from
a well-sampled application.

Campaign: `trace_experiments.py` writes **runs 660–674** (3 arrival arms × 5
policies) and `settings/trace_campaign.json`, cloning `cloud/`, `workflows/` and
`placement/` from the $w_3 = 1.5$, eligibility + $\ell_2$ cell — run **316**
(deterministic) and run **317** (stochastic). The memory model is the same on
every row, so the grid isolates the placement and the runtime rule instead of
folding a third factor into every percentage.
Nothing is re-solved. `run_trace_study.sh [MAXJOBS]` drives it end to end and
finishes with `trace_table.py --latex`. It runs at `simulation_time = 400`, the
horizon of the RAM × $w_3$ and policy campaigns, so **its numbers are directly
comparable with those figures**; the Poisson arm reproduces runs 316 and 317
bit-for-bit on all ten streams, which is the regression gate on the trace code
path. There is still no steady state at this arrival rate, so `trace_table.py`
reports the backlog slope next to every percentage.

Measured at $w_3 = 1.5$ with the memory model held at eligibility + $\ell_2$ on
**every** row, ten paired streams, each policy paired against the deterministic
open loop of the *same* arm. Percentages carry the t-based 95 % half-width of
the *paired* difference, which is the only interval that is a statement about
the policy — the absolute CI on the first row is dominated by how much the ten
streams differ from one another, which the pairing cancels. Every cell wins
10/10.

| policy | Poisson | trace, high volume | trace, shared |
|---|---:|---:|---:|
| det., open loop | 31578 ± 1434 | 32608 ± 1772 | 31965 ± 2399 |
| det., least loaded | −6.25 ± 1.43 % | −6.42 ± 1.45 % | −4.93 ± 1.26 % |
| stoch., open loop | −3.79 ± 1.30 % | −4.91 ± 2.32 % | −4.41 ± 1.01 % |
| stoch., free sample wide | −8.85 ± 1.77 % | −10.37 ± 2.83 % | −8.34 ± 1.60 % |
| stoch., least loaded | −9.24 ± 1.92 % | −10.26 ± 2.82 % | −8.26 ± 1.63 % |

The paired difference is the **ratio of means**, $100\,\overline{a-b}/\overline{b}$,
with the t interval of the absolute differences rescaled by the same
$\overline{b}$ — the convention `visualization_policy.py` and
`visualization_ram_w3.py` already used. It is *not* the mean of the per-stream
ratios: the two differ by 0.05–0.08 points here, because the per-stream saving
is correlated with how expensive the stream is (the five cheap streams save
2.95 %, the five expensive ones 4.53 %) and the ratio of means is algebraically
a cost-weighted mean of the per-stream ratios. That is enough to move a
one-decimal label, so all three scripts now use the same definition.

Backlog slope (s of added waiting per 1000 s) down the same five rows: Poisson
75.0 → 47.1 → 53.8 → 28.0 → 26.6; high volume 77.3 → 45.5 → 43.7 → 23.5 → 22.0;
shared 55.9 → 35.6 → 31.9 → 16.3 → 18.9. Four readings matter.

**The advantage survives burstiness, and grows slightly with it.** Every policy
is at least as good in the high-volume arm as under Poisson — the full stack
goes −9.2 % → −10.3 %, the relaxation alone −3.8 % → −4.9 %.

**The Poisson arm reproduces the policy campaign to two decimals**, as it must:
it is runs 316/317 with the same rules and now the same estimator, and
−3.79 / −6.25 / −8.85 / −9.24 % match `visualization_policy.py`'s
−3.7916 / −6.2494 / −8.8525 / −9.2406 % on the same cells. That is the
cross-check that the trace machinery changed nothing it should not have.

**Runtime recourse on the deterministic placement is still the honest
competitor.** `det., least loaded` (−6.25 %) beats `stoch., open loop`
(−3.79 %) in all three arms — §6's "uncomfortable one", and stated cleanly here
because the memory model is the same on both sides. The comparison that
isolates the *placement* is `stoch., least loaded` against `det., least loaded`,
same rule and same candidate set on both: about three percentage points, holding
across all three arrival arms.

**The rule still does not matter, the candidate set does**: `free sample, wide`
and `least loaded` are within each other's intervals in every arm, so $H$ can
stay authoritative at no measured cost. Absolute CIs in the shared arm are
±7.5 % of the level against ±4.5 % under Poisson, so **only the paired numbers
are reportable there** — and those are the tightest of the three arms
(±0.7–1.2 %).

**Why the memory model is fixed rather than varied.** An earlier version of this
grid paired a deterministic *aggregate* baseline against the stochastic
$\ell_2$ arms, which folded the memory model into every percentage. Measured on
these ten streams, the deterministic $\ell_2$ placement (run 316) costs
**+1.30 % against the deterministic aggregate one (run 304), on 10/10 streams**,
and that penalty is **purely monetary**: $w_1Z$ 14801 → 15335 (+3.6 %), $w_2B$
11442 → 11313 (−1.1 %), and $w_2W$ **4928.85 in both, bit-identical on every
stream**. On a deterministic placement the eligibility condition only relocates
functions to pricier, marginally faster nodes and changes realized congestion
not at all — the memory model starts paying for itself once $H$ is relaxed.
Holding it fixed removes that term from every number above.

`visualization_trace.py` draws the figure: `results/trace/trace_bars.pdf`, six
horizontal stacked bars — `det., open loop`, `stoch., open loop` and
`stoch., least loaded` under Poisson and under the high-volume trace — split
into $w_1Z$ / $w_2B$ / $w_2W$, with the absolute interval as the error bar and
the paired difference as the annotation. The split is what makes the mechanism
visible: **$w_1Z$ is identical across the three bars of an arm and $w_2B$ moves
by under 0.2 %, so the whole effect is the waiting segment** — 4929 → 3747 →
2027 under Poisson. `--arms` and `--policies` select any other cells of the
3 × 5 grid. It prints every plotted number and asserts that each stack sums to
the measured cost, so the figure cannot drift from `trace_table.py`.

Note on comparing runs: the placement problem is a nonconvex MIQCQP solved to
`MIPGap 0`, and when several placements attain the same optimum the one Gurobi
returns is arbitrary. Re-solving the *identical* model under different Gurobi
`Seed` values moved the simulated cost by up to 236 on `run_141` and 1347 (9.3%)
on `run_140`. However, this is mostly important when w_3=0.
Differences below that scale between two model variants are not
evidence; compare objective values or seed-replicated ranges instead.


