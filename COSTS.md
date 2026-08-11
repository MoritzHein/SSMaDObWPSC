# Cost quantities in OptiFaaS: definitions, where they are computed, and how they differ

This document describes every cost quantity used in the pipeline, whether it is
computed in the **optimization problem** (`optimization.py`, `solve_opt`), in the
**simulation** (`selection_simulation.py`), or in both — and, where both exist,
how the two computations differ. Line anchors refer to the current code.

There are **three accounting layers**:

| layer | where | weighting | rounding | includes queueing |
|---|---|---|---|---|
| (1) optimization objective | `optimization.py:284-372` | expected rate λ_k | none (continuous) | no (only the w₃ proxy) |
| (2) simulation *analytic* costs | `selection_simulation.py` cost loops (decomposed `:981-1193`, comparison `:1851-1969`) | **sampled** request counts | none (continuous) | no |
| (3) simulation *event* costs | discrete-event loop `selection_simulation.py:632-966` (comparison mirror `:1546-1841`) | actual sampled requests | `int(np.ceil(...))` on durations and latencies | yes |

Notation: deployment *n*, function *m*, node *i*, leaf (user) *k*, branch/path *s*;
weights `w_1` (money), `w_2` (time), `w_3` (utilization) from
`settings/parameters_<N>.json`; λ_k = request rate of leaf *k*.

---

## 1. Money cost M

### In the optimization problem (layer 1)

Objective terms `optimization.py:289-337`; parameter matrices built at `:150-218`.
Per (leaf *k*, selected deployment *n*), weighted `λ_k · H[k,n] · w_1`:

- **Execution cost** `C[m,i]` (`:211-218`): `pricing_RAM · ram_m · time_m` on
  non-tinyfaas nodes, **0 on tinyfaas**. Note: uses the *base* time — **no
  speedup factor** — so money is independent of the `speedup` parameter.
- **Storage/dependency transfer** (also inside `C[m,i]`, `:200-210`):
  `pricing_Storage_Transfer · d_dependencies` for every dependency stored at a
  provider different from node *i*'s provider.
- **Inter-function data transfer** `S ≡ Σ_successors D[m,i,j]` (`:442-447`,
  objective `:294`): `D[m,i,j]` (`:187-199`) = `data_send` priced with the
  sender's **plus** the receiver's `pricing_data_sent` when the two nodes belong
  to different providers (tinyfaas side free). Counted along the **adjacency**
  `a[m,m_s]` (direct successors), once per edge.
- **Input data** `D_in[i] = input_data · pricing_data_sent` for the input
  function(s) `m_start` (`:186`, objective `:326`).
- **Result return** `D[m,i,leaf_k]` for the output function(s) `m_end`
  (objective `:328`).

Every function's cost is counted **exactly once** (per placement row), regardless
of how many branches contain it.

### In the simulation (layer 2, "analytic" re-computation)

`money_cost` in results.json / results_comparison.json. Comparison (flat) loop
`selection_simulation.py:1851-1969`; decomposed loop `:981-1193` traces every
function down the hierarchy but uses the same terms. Differences to layer 1:

- **Weighting**: each (workflow, leaf) term is multiplied by the **sampled**
  Poisson request count `overall_requests[t,k]` (`:1888`), not λ_k. With few
  requests per pair this weighting is noisy (run 138: ~140 requests over 60
  pairs, 14 pairs sampled zero).
- **Per-branch double counting**: `C_func[m]` accumulates once per branch *s*
  containing function *m* (`:1899-1953`). A function shared by several branches
  (branching workflows, e.g. run 138 wf 2/3/6) has its execution + storage cost
  counted once **per branch**, whereas the model counts it once.
- **Inter-function transfer**: `D_func[m] = D[m][place][next_place]` uses the
  *path successor* (and is overwritten, not accumulated, when *m* lies on
  several branches), whereas the model sums `D` over all adjacency successors.
- Execution money uses `pricing_RAM · ram · time` with the same no-speedup
  convention as the model (`:1950-1953`, decomposed `:1145-1146`).

### Not in layer 3

The event simulation does not price anything; money is purely analytic.

---

## 2. Basic time cost B

### In the optimization problem (layer 1)

`w_2`-weighted terms `optimization.py:291, 326-328`, branch-time constraint `:433-440`:

- `T[n] ≥ Σ_{m∈branch s} ( Speed[m,i]·Time[m] + Σ_{m_s: a[m,m_s]=1} L[i_m, i_{m_s}] )`
  for every branch *s* — at the optimum `T[n]` equals the **maximum branch
  time**. `Speed[m,i]` = the function's `speedup` on non-tinyfaas nodes, 1 on
  tinyfaas (`:99-102`; the per-provider cloud speedup is commented out).
- The **user↔node latencies of the input and output functions are added outside
  the maximum**: the objective adds `w_2·L[leaf_k, i]` for every `m_start` *and*
  every `m_end` (`:326-328`; note both use `L[leaf, i]` — row `leaf` — relying on
  a symmetric L).
- All quantities continuous, weighted `λ_k·H[k,n]`.

### In the simulation (layer 2, `time_cost`)

Comparison loop `:1895-1962`, decomposed loop `:1025-1183`:

- Per branch: `T_paths[s] = Σ_{m∈s} ( speedup·time + L[prev_node, node] ) + L[last_node, leaf]`
  where `prev_node` starts at the leaf. So the **leaf→first and last→leaf
  latencies are inside the branch sums**, and `time_cost = max_s T_paths[s]` —
  the branch maximum *includes* them, whereas the model adds them as a **sum
  over all start/end functions outside** the maximum.
- Decomposed variant composes latencies across the hierarchy via
  `upward_trajectory` (Algorithm 1) instead of the flat `L`.
- Weighted by sampled counts, like money.

### In the event simulation (layer 3, `base_T`)

`selection_simulation.py:660-685` (lazy) / `:836-852` (default), plans `:508-556`:
same structure as the layer-2 branch time but with **integer rounding**:
durations `int(np.ceil(speedup·time))` and transfers `int(np.ceil(L[i,j]))`.
`base_T` is the blocking-free response time a request would have; it is the
baseline that waiting is measured against.

---

## 3. Waiting time W

**Event simulation only** (`selection_simulation.py:632-966`; comparison mirror
`:1546-1841`). No optimization-layer counterpart. Two blocking rules:

1. **RAM capacity**: a node cannot start a function if its occupied RAM plus the
   function's RAM exceeds `max_RAM_curr` — enforced only for tinyfaas nodes;
   cloud nodes are unlimited (`:210-214`).
2. **Function serialization**: one function of one deployment can serve only one
   request at a time (`func_busy` check `:898-906`); requests wait until the
   blocking execution ends.

Per request: `wait = max(0, (end − arrival) − base_T)`; accumulated into
`waiting_acc` and stored as `complete_time_cost` (Σ over all requests).
The per-request response times `end − arrival = base_T + wait` are collected in
`req_done_list`.

The **only** term in the optimization that addresses this effect is the `w_3`
proxy (next section). With `w_3 = 0` the solver is completely blind to
congestion, and equal-objective placements can differ by orders of magnitude in
simulated waiting (run 138: baseline decomposed vs centralized differ by 0.5 in
model cost but ~52,000 in overall simulated cost).

---

## 4. Utilization / load proxies

### In the optimization problem

- **Function utilization** `p[n,m] = Σ_k λ_k·H[k,n]·Time[m]·Speed[m,i_m]`
  (expected busy time of one function deployment), penalized `w_3·p²`
  (`optimization.py:340-348`) — a convex proxy for the serialization blocking.
- **Node RAM utilization** `M[i]` = expected RAM-time share of `ram_max`
  (plus pre-existing utilization), penalized `w_3·M²` (`:349-366`).

### In the simulation

`select_cost = Σ (deployment-selection counts)² + Σ (invocations per node)²`
is computed (`:1192-1193`) but **not persisted** and not part of any stored
cost. The measured counterpart of what `w_3` protects against is the waiting
time W itself.

---

## 5. Aggregates and where they are stored

| quantity | formula | file |
|---|---|---|
| optimization objective | `Σ_k Σ_n λ_k H[k,n] (w_1·M_model + w_2·B_model) + w_3·(Σp² + ΣM²)` | not stored (solver internal); reconstructable with `evaluate_comparison_objective.py --model_objective` |
| `overall_cost` | `w_1·money_cost + w_2·Σ req_done` where `Σ req_done = Σ(base_T + wait)` (event layer) and `money_cost` is the analytic layer-2 money | `results.json` (decomposed solution) |
| `money_cost`, `time_cost` | layer-2 analytic, count-weighted | `results.json` |
| `complete_time_cost` | Σ waiting (layer 3) | `results.json` |
| same four for the comparison solution | identical formulas on `opt_<w>_full.json` | `results_comparison.json` (+ `heuristic` flag: centralized vs all-on-cloud) |
| per-request logs, per-node RAM | layer 3 | `detailed_simulation_run_<N>.json` — the **decomposed** solution's logs, snapshotted before the comparison block re-initializes them; the comparison run's own logs go to `detailed_comparison_run_<N>.json` when `comparison` is on |
| per-request arrival, response and waiting times per request stream | layer 3 | `latency_seeds.json`, written by `resimulate_seeds.py` from the trace above |

Note the hybrid nature of `overall_cost`: its money part is analytic (layer 2,
continuous, count-weighted) while its time part is the event simulation's
response times (layer 3, ceil-rounded, including waiting).

---

## 6. Summary of model-vs-simulation discrepancies

1. **Waiting** exists only in the simulation; the model has only the `w_3·p²/M²`
   proxy. This dominates all observed nominal-vs-adapted and
   decomposed-vs-centralized differences in `overall_cost` when `w_3 = 0`.
2. **Weighting**: model uses expected rates λ_k; simulation weights by the
   sampled Poisson counts (large per-pair noise at low rates).
3. **Shared-function money**: simulation's analytic loop bills functions once
   per branch; the model bills once per function. Affects branching workflows
   only.
4. **Start/end latency aggregation**: simulation puts leaf↔node latencies inside
   the branch maximum; the model sums them over all input/output functions
   outside the maximum.
5. **Inter-function transfer money**: model sums `D` over adjacency successors;
   simulation uses the path successor (overwritten across branches).
6. **Rounding**: event layer ceils durations and latencies; model and analytic
   layer are continuous.
7. **Speedup**: affects only time (model `Speed·Time`, simulation
   `speedup·time`), never money (`pricing_RAM·ram·time` in both). Both apply it
   only on non-tinyfaas nodes.

Because of (2)-(5), *ranking two placements by the simulation's analytic
`w_1·money + w_2·time` is **not** equivalent to ranking them by the optimization
objective*: run 138 showed adapted centralized solutions that are strictly
better in the model objective (as an exact solver guarantees) but evaluate
worse in the analytic accounting (ram ×0.75/0.875, speedup ×0.75/0.875).
`evaluate_comparison_objective.py` evaluates the comparison placement under
either formula (`--model_objective`), either weighting (`--count_weights`
reproduces `results_comparison.json` exactly), for any scenario overlay
(`--scenario_dir`) or adapted sandbox (run with `cwd=<point>/adapted/`).
