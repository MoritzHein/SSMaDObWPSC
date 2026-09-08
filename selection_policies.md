# Selection policies and model-free placements

This note documents the **runtime selection policy** (`selection_policy` in
`settings/parameters_<N>.json`) and the **model-free comparison placements**
(`comparison_mode`). It is the reference behind Section IV‑B of the UCC paper,
and the companion of [`stochastic_selection.md`](stochastic_selection.md), which
covers the placement‑time relaxation, and
[`ram_capacity_model.md`](ram_capacity_model.md), which covers the memory model.

Every number quoted here was measured on the current tree, on the campaign of
`settings/policy_campaign.json` (runs 500–607 and 620–625), ten paired request
streams of $t_\text{sim}=400$ s each.

---

## 1. Two different things are called a "policy"

The repository overloads the word, and the two meanings live in different
places and different phases.

| | placement‑time | runtime |
|---|---|---|
| flag | `stochastic_selection` | `selection_policy` |
| what it decides | whether $H_{k,n}$ is binary or a distribution on the simplex | which of the deployments a request is actually sent to |
| when | while solving, once | per request, during simulation |
| touched code | `optimization.py`, `placement.py`, `cloud_workflow_setup.py` | `selection_simulation.py` only |
| re‑solve needed | yes | **no** — a variant reuses a solved run |
| campaign axis named "policy" | `ram_w3_experiments.py`, `centralized_experiments.py`, `scaling_levels_experiments.py` | `policy_experiments.py`, `trace_experiments.py` |

The two compose. The placement fixes $P$ and $H$; the runtime policy decides
what to do with $H$ when the deployment it names is busy. **A runtime rule is
only as useful as the option set the placement leaves it**, which is the point
the paper makes with these experiments.

---

## 2. Notation

| symbol | meaning |
|---|---|
| $k$ | user leaf that a request originates at |
| $n$ | deployment index of the workflow |
| $m$ | function index |
| $H_{k,n}$ | routing probability of leaf $k$ to deployment $n$ (one‑hot under a deterministic placement) |
| $\operatorname{supp}H_{k,\cdot}$ | $\{n : H_{k,n}>0\}$ |
| $\mathcal P$ | deployments whose functions are all placed and not pruned |
| $\mathcal C$ | the candidate set of the policy, $\mathcal C\subseteq\mathcal P$ |
| $\mathcal F$ | the free candidates, $\{n\in\mathcal C : \delta_n=0\}$ |
| $\delta_n$ | delay estimate of deployment $n$, $0$ if it can start now |
| $E$ | entry functions of the workflow, one per branch |
| $R_m$ | memory demand of function $m$ |
| $c_{k,n}$ | how often leaf $k$ has already been served by deployment $n$ |

---

## 3. What the simulator knows at request time

A runtime rule is meaningful only because the simulator has state the optimizer
did not. Two blocking rules produce that state.

1. A deployed function instance serves **one request at a time**.
2. No function starts on a node whose **memory is exhausted**.

`deployment_delay(plan, entries, node_active, node_ram_max, w, d, cur_t)`
(`selection_simulation.py:287`) turns that state into a scalar per deployment.
For each entry function $m\in E$ it looks at the node the plan put $m$ on and
asks whether the request could start there right now. If some other request
already occupies that function instance, or the node's free memory is below
$R_m$, the deployment is not free and the delay is the time until the earliest
running execution on that node ends. The result is

$$
\delta_n=\max_{m\in E}\;\bigl(\text{time until } m \text{ of deployment } n \text{ can start}\bigr),
\qquad
\delta_n=0 \iff \text{deployment } n \text{ can start the request now.}
$$

The **entry functions** come from `entry_functions` (`:255`): the first function
of every branch of the workflow, deduplicated. This is the `probe` axis of the
policy and `"entry"` is its only value. Probing only the entries is a
deliberate approximation — a deployment that is free at its entry may still
block later in the workflow — chosen because the later functions' occupancy at
the time the request will reach them is not known when the request arrives.

---

## 4. Candidate sets

`choose_deployment` (`:357`) builds the candidate set first, then applies the
rule. The three sets differ in how much authority they leave to $H$.

### `support` — keep $H$ authoritative

$$\mathcal C=\operatorname{supp}H_{k,\cdot}\cap\mathcal P$$

The rule may only reorder the deployments the optimizer already selected. The
realized routing is therefore a conditioning of $H$ on availability, and the
rates $u_{n,m}$ and the node utilizations $M_i$ the placement was priced with
keep describing what happens at runtime.

> **A deterministic placement makes this set a singleton.** Under the
> decomposition, selection is solved once per region, so all leaves of a region
> share one deployment. Measured over the physical-level subproblems of runs 316
> and 317, ten workflows each: the deterministic placement leaves **10**
> non-empty subproblems carrying **one** selection row each, while the relaxed
> one leaves **40** subproblems carrying **220** rows, twenty of them with one
> row and twenty with ten. A rule restricted to `support` is therefore a
> **no-op** on a deterministic placement. This is why the paper calls the
> relaxation the enabling condition for runtime recourse rather than an
> adaptation in itself.

### `placed` — ignore $H$ at runtime

$$\mathcal C=\mathcal P$$

Every deployment of the workflow is a candidate, whatever $H$ says. This is the
only set that gives a deterministic placement something to decide, so it is
what the fair "same recourse on both placements" comparison uses. The price is
that the realized routing has no relation to $H$ any more.

### `support_then_placed` — widen only when forced

Start from `support`. **Only if nothing in it is free**, add the remaining
deployments of $\mathcal P$ and re‑evaluate. $H$ stays authoritative whenever
anything it selected is available, and the widening is a fallback rather than a
policy.

### Measured, at the ℓ₂ memory model and $w_3=1.5$

`selection_stats` of runs 572–580, 341 requests per run:

| candidate set | requests with a choice | something free | nothing free |
|---|---|---|---|
| `support` | 26.1 % | 50.1 % | 49.9 % |
| `support_then_placed` | 65.7–68.0 % | 63.6–65.1 % | 34.9–36.4 % |
| `placed`, deterministic placement | 100 % | 55.4 % | 44.6 % |
| `placed`, relaxed placement | 100 % | 63.9 % | 36.1 % |

Two things to read out of this. Requests concentrate on the leaves that do have
several deployments, so the support gives a quarter of all requests a choice
even though only a small minority of selection rows carry more than one
deployment. And widening finds a free deployment for a further 14 % of all
requests, roughly a third of everything the support alone leaves blocked —
which is exactly the traffic that ends up on deployments $H$ never selected.

---

## 5. Rules

All rules are in `choose_deployment` (`:427`–`:511`). Two guards apply to all of
them.

* **Fallback** (`:445`). If $\mathcal F=\varnothing$ the rule is skipped and the
  deployment is drawn from $H_{k,\cdot}$ as usual. `fallback` is a policy key
  but `open_loop` is its only accepted value.
* **Placement guard** (`:507`). If the chosen deployment has no built plan the
  draw is repeated from $H_{k,\cdot}$.

### `open_loop`

The unmodified policy: $n\sim H_{k,\cdot}$. Kept as an explicit rule so that a
run with `instrument: true` reproduces its uninstrumented counterpart
bit‑for‑bit — the draw is identical and only `selection_stats` is written.
`normalize_selection_policy` returns `None` for an uninstrumented open loop, so
the whole policy code path is bypassed in that case.

### `free_sample`

$$\Pr[n]=\frac{H_{k,n}}{\sum_{n'\in\mathcal F}H_{k,n'}},\qquad n\in\mathcal F$$

$H$ conditioned on availability. Over `support` this is the rule that preserves
the model's own routing most closely, which is why it is the one the paper
reports. If the free candidates carry no probability mass at all — possible only
with a widened set — it falls back to a uniform draw over $\mathcal F$.

### `free_uniform`

Uniform over $\mathcal F$. Discards the magnitudes of $H$ and keeps only its
support. It exists to test whether the optimizer's probability *values* buy
anything at runtime beyond the option set they define.

### `free_argmax_h`

$\arg\max_{n\in\mathcal F}H_{k,n}$, ties broken uniformly with a tolerance of
$10^{-12}$. The greedy counterpart of `free_sample`.

### `least_loaded`

$$n=\arg\min_{n'\in\mathcal C}\delta_{n'}$$

ties broken uniformly. Note it minimises over $\mathcal C$, **not** over
$\mathcal F$ — identical whenever something is free, since then the minimum is
zero, but it also gives a sensible answer when nothing is. It uses no
information from $H$ at all, so combined with `placed` it is the upper bound of
what runtime recourse alone can achieve, and the only rule that is meaningful on
a deterministic placement.

### `deficit_free`

$$n=\arg\max_{n'\in\mathcal F}\Bigl(H_{k,n'}\cdot\bigl(1+\textstyle\sum_j c_{k,j}\bigr)-c_{k,n'}\Bigr)$$

The free deployment with the largest shortfall of realized service against the
share $H$ prescribes. Intended to hold the empirical routing frequencies on $H$
more tightly than sampling does. **It does not preserve the marginals once the
candidate set widens** — no rule does, because the widening itself is what
moves traffic off the support.

### `power_of_two`

Two independent draws from $H_{k,\cdot}$, take the one with the smaller known
delay. Implemented and validated, but **used by no campaign** — no entry in
`policy_experiments.VARIANTS` and no parameter file references it.

---

## 6. The campaign variants

`policy_experiments.py:58` defines the nine combinations that are actually run,
plus the two open‑loop references, which have no `selection_policy` block at all
and are read from their source runs 300–323.

| key | placement | candidates | rule | role |
|---|---|---|---|---|
| `det_ol` | det. | — | open loop | the baseline every difference is paired against |
| `stoch_ol` | stoch. | — | open loop | the relaxation without recourse |
| `det_ll` | det. | `placed` | `least_loaded` | recourse without the relaxation |
| `det_free` | det. | `placed` | `free_sample` | same, sampling instead of greedy |
| `free_sup` | stoch. | `support` | `free_sample` | recourse with $H$ authoritative |
| `deficit_sup` | stoch. | `support` | `deficit_free` | same, marginal‑preserving attempt |
| `uniform_sup` | stoch. | `support` | `free_uniform` | same, values of $H$ discarded |
| `free_wide` | stoch. | `support_then_placed` | `free_sample` | widen when forced |
| `argmaxh` | stoch. | `support_then_placed` | `free_argmax_h` | same, greedy |
| `deficit_wide` | stoch. | `support_then_placed` | `deficit_free` | same, deficit |
| `stoch_ll` | stoch. | `placed` | `least_loaded` | ceiling of runtime recourse |

Run numbering (`policy_experiments.py:135`) is
`500 + 54*ram_index + 9*w3_index + variant_index`, with the indices taken in
manifest order, and the model-free references are 620–625.
The runs **clone** `cloud/`, `workflows/` and `placement/` from runs 300–323
instead of re‑solving, so the solver cannot return a different optimum of equal
objective and confound the comparison.

### Measured, eligibility + ℓ₂, paired against `det_ol`

Flat in $w_3$, and every cell wins on all ten streams.

| variant | range over $w_3\in[0.5,3]$ |
|---|---|
| `stoch_ol` | −3.7 … −4.5 % |
| `free_sup`, `deficit_sup`, `uniform_sup` | −5.6 … −6.9 % |
| `det_free` | −6.1 … −6.2 % |
| `det_ll` | −6.2 … −6.4 % |
| `free_wide`, `argmaxh`, `deficit_wide` | −8.8 … −9.4 % |
| `stoch_ll` | −9.0 … −9.5 % |

The three support‑only rules are indistinguishable from one another, and so are
the three widened ones. **The rule does not matter, the candidate set does.**
Whether the optimizer's probability values are sampled, taken greedily or
discarded for a uniform draw changes nothing once blocked deployments are
skipped; what the relaxation contributes at runtime is the option set.

At $w_3=1.5$ the cost split is unchanged except for waiting. Every one of the
five arms the paper reports pays a monetary cost of $15335.415$, identical to
three decimals, and a realized basic time of $11313$ on a deterministic
placement or $11298$ on a relaxed one, a spread of $16$ over all five. Waiting
falls $4929\to3747\to3159\to2955\to2027$ along `det_ol`, `stoch_ol`,
`free_sup`, `det_ll`, `stoch_ll`. The runtime rule cannot move money or base
time by construction — it changes which deployment serves a request, not where
the functions live or how fast they run.

**The trade‑off to report.** Keep $H$ authoritative for about 6 %, with the
model's own pricing still describing runtime, or widen for about 9 % and accept
that a part of the traffic is routed where $H$ never sent it.

**The uncomfortable one.** `det_ll` beats `free_sup` at four of the six weights.
Runtime recourse on the old placement is a real competitor to the narrow rule on
the new one, and it is cheaper than `stoch_ol` everywhere. The comparison that
isolates the relaxation is `det_ll` against `stoch_ll` — the same rule and the
same candidate set on both placements — where the relaxed placement is still
about three percentage points cheaper.

---

## 7. Model-free comparison placements

Selected with `comparison_mode`; the tuple lives in `optimization.py:33` and the
dispatcher is `heuristic_placement` (`:176`). None of them solves anything and
all of them honour the eligibility condition through `_eligible_nodes` (`:37`).
Overflow always goes to `_cheapest_cloud` (`:43`), the non‑edge node with the
smallest monetary cost for that function.

| mode | label | placement rule | selection $H$ |
|---|---|---|---|
| `all_cloud` | all on cloud | every function of every deployment on node 0 | every leaf to deployment 0 |
| `all_cloud_rr` | all on cloud, spread | same | leaf $k$ to deployment $k \bmod n_\text{dep}$ |
| `local_first` | edge first | one edge home per deployment, round robin; fill it while its RAM budget lasts, spill the rest to the cloud | round robin |
| `best_fit_edge` | best fit edge | best‑fit‑decreasing bin packing over all edge devices, largest function first, tightest fitting device wins | round robin |
| `greedy_cost` | greedy cost | walk each branch in execution order, take the locally cheapest eligible node under $w_1C+w_2(\sigma t+L_\text{prev})$ | leaf takes the deployment with the shortest entry‑exit round trip |
| `random_feasible` | random | uniform over the eligible nodes, seeded with 42 | uniform |

Measured over the ten streams of runs 620–625, independent of $w_3$, of the
memory model and of the selection rule, which is why the campaign simulates them
once each rather than in all 108 cells:

| mode | $w_1Z$ | $w_2B$ | $w_2W$ | $C$ |
|---|---:|---:|---:|---:|
| `all_cloud` | 15 335 | 11 298 | 24 539 | 51 172 |
| `all_cloud_rr` | 15 335 | 11 298 | 4 419 | **31 052** |
| `local_first` | 15 202 | 11 563 | 26 589 | 53 353 |
| `best_fit_edge` | 6 128 | 15 106 | 183 674 | 204 908 |
| `greedy_cost` | **2 928** | 15 249 | **911 938** | 930 115 |
| `random_feasible` | 14 979 | 12 679 | 25 616 | 53 274 |

Three of these deserve a comment.

* **`all_cloud` is a straw man.** Every leaf takes deployment 0, so one function
  instance serialises the whole architecture's traffic for a workflow. It was
  the historical default baseline. `all_cloud_rr` is the *same placement* with
  the leaves spread over the deployments and costs 40 % less. Comparing against
  `all_cloud` measures the selection, not the placement.
* **`all_cloud_rr` is the reference worth beating**, and it is genuinely hard to
  beat. Paired at $w_3=1.5$ under the ℓ₂ model, the deterministic open loop is
  **+1.7 %** against it and cheaper on only 3 of 10 streams. Only the relaxed
  placement with recourse clears it convincingly, at **−7.7 %** on 10 of 10.
* **`greedy_cost` is the argument for the utilization penalty.** The split above
  makes the mechanism explicit. Its monetary cost is 2 928, by far the *lowest*
  of all six and a fifth of the cloud baseline's, because it puts most functions
  on unpriced edge devices. Its waiting is 911 938, two hundred times the cloud
  baseline's. A myopic cost-greedy rule finds the free edge devices, piles every
  deployment onto the same ones and destroys itself on congestion. This is
  precisely the failure mode the $w_3$ term in the objective exists to prevent,
  and no runtime rule can undo it, because the placement is already made.
* **Edge placement buys very little money here.** `local_first` saves 0.9 % of
  the monetary cost against all-on-cloud and pays for it with six times the
  waiting. The edge devices are too small to host most functions, so an
  edge-first rule spills nearly everything to the cloud anyway and only
  concentrates what does fit.

---

## 8. Flags

```json
"selection_policy": {
    "candidates": "support_then_placed",
    "probe":      "entry",
    "rule":       "free_sample",
    "fallback":   "open_loop",
    "instrument": false
}
```

| key | values | default |
|---|---|---|
| `candidates` | `"support"`, `"placed"`, `"support_then_placed"` | `"support"` |
| `probe` | `"entry"` | `"entry"` |
| `rule` | `"open_loop"`, `"free_sample"`, `"free_argmax_h"`, `"free_uniform"`, `"least_loaded"`, `"power_of_two"`, `"deficit_free"` | `"open_loop"` |
| `fallback` | `"open_loop"` | `"open_loop"` |
| `instrument` | `true` / `false` | `false` |

Absent or `null` keeps the open loop and bypasses the entire code path. Unknown
keys raise. The rules draw from `rng_sel` only, never from `rng`, so the arrival
stream is untouched and the comparison across policies stays paired by
construction. `selection_policy` requires the default scheduler and raises when
combined with `lazy_start`.

`comparison_mode` takes `"centralized"` or one of the six modes of §7. It
defaults to `"all_cloud"` when the legacy `heuristic` flag is true and to
`"centralized"` otherwise, so older parameter files are unaffected.

---

## 9. Files and tooling

| file | role |
|---|---|
| `selection_simulation.py` | the only consumer of `selection_policy`: `SELECTION_POLICY_DEFAULTS` `:139`, `normalize_selection_policy` `:167`, `new_selection_stats` `:233`, `entry_functions` `:255`, `deployment_delay` `:287`, `choose_deployment` `:357`, call sites `:1557` (own run) and `:2562` (comparison run) |
| `optimization.py` | `HEURISTIC_MODES` `:33`, `heuristic_placement` `:176` and the six `_place_*` helpers |
| `placement.py` | resolves `comparison_mode` from the legacy `heuristic` flag, `:235` |
| `policy_experiments.py` | writes runs 500–607 and 620–625 plus `settings/policy_campaign.json` |
| `run_policy_study.sh` | drives generate → solve 620–625 → simulate → `resimulate_seeds.py` → `visualization_policy.py --paper --latex` |
| `visualization_policy.py` | `policy_bars.pdf`, `policy_w3.pdf` (diagnostic, all eleven policies and all six references); `--paper` adds `policy_paper.pdf` (five policies against $w_3$, ℓ₂ only) and `policy_bars_paper.pdf` (the same five at one $w_3$, both memory models as blocks, references either as their own bars between the blocks or as vertical rules — `--heuristic-style row|rule`) |

**`time_cost` is not the basic time the cost is made of.** It is the *analytic*
one, the critical path of the workflow times the number of requests of that
(workflow, leaf) pair, accumulated in the reporting block, and it runs about
3.7 % below the realized basic time. `money_cost` and `complete_time_cost`, by
contrast, enter

$$\texttt{overall\_cost}=w_1Z+w_2\textstyle\sum(\text{realized response times})$$

exactly as reported, so the entire discrepancy sits in the basic time and the
realized one is the remainder,

$$w_2B=\texttt{overall\_cost}-w_1Z-w_2W .$$

The identity is exact rather than approximate. The simulator records
$\text{wait}=\max(0,\ \text{response}-\text{base}_T)$, so
$\sum\text{response}-\sum\text{wait}=\sum\text{base}_T$ unless a request
finishes faster than its own base time, which happens in **none of the 138
campaign runs** (it does in the old runs 123–137, at up to 0.3 %). Every stacked
figure uses the remainder and therefore sums exactly to the measured cost.

Two consequences. **Never stack the three reported fields directly** — the bar
lands 0.89 % to 1.44 % short of `overall_cost`, and since the model-free
references are `overall_cost_comparison`, a reference rule can fall between a
bar's top and its own marker. It does exactly that for the aggregate model's
`det_ol` at $w_3=1.5$, where the naive stack is 30761, the measured cost 31171
and `all_cloud_rr` 31052. And **quote `overall_cost`** for any level or
difference, never a sum of components.

Results land in `results/run_<N>/`. `results.json` carries `selection_stats` and
`selection_counts` in addition to the cost split, `results_seeds.json` the ten
paired streams that every reported number is computed from, and
`results_comparison.json` the model-free placement's cost together with the
`comparison_mode` that produced it — but only for the last stream simulated, so
never read a level out of it.

`results_seeds.json` also carries the comparison run's **split** per stream, as
`money_cost_comparison`, `time_cost_comparison` and
`complete_time_cost_comparison`, which is what lets a model-free placement be
drawn as a stacked bar rather than a single total. `resimulate_seeds.py` records
these only for runs re-simulated after the field was added, and only when the
run's own `comparison` flag is set, so a stale `results_comparison.json` from an
earlier run cannot leak in. Runs 620–625 were re-simulated for it; the policy
runs 500–607 set `comparison: false` and have no comparison split at all.

**Regression gate.** With `selection_policy` absent, a run has to reproduce its
`results.json` *and* its `detailed_simulation_run_<N>.json` bit‑for‑bit against
the pre‑policy backup in `_backup_3xx_20260811_230808/`.

---

## 10. Known issues

* **`apply_to_comparison` is unreachable.** `selection_simulation.py:1966` gates
  the comparison run on `selection_policy.get("apply_to_comparison")`, but that
  key is not in `SELECTION_POLICY_DEFAULTS` and `normalize_selection_policy`
  raises on unknown keys, so it can never be set to `true`. The effective
  behaviour is the intended default — the comparison run is always open loop, so
  a model-free baseline is never silently credited with the rule under test —
  but the flag as written is dead. Adding `"apply_to_comparison": False` to the
  defaults would make it real and change nothing else.
* **`policy_table.py` is stale** and is not called by `run_policy_study.sh`. It
  reads `man["source_runs"]`, while the manifest key is `baselines`, and it
  indexes `info["w_3"]` on the heuristic entries 620–625, which carry only
  `mode` and `label`. Both raise `KeyError`. `visualization_policy.py` is the
  working producer of the tables and figures.
* **`power_of_two` is untested by any campaign** (§5).
* **`probe: "entry"` under‑estimates blocking** for long workflows, by
  construction (§3).
