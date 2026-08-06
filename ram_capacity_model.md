# RAM capacity in the decomposition: eligibility and the $\ell_1$ / $\ell_2$ norm

This note documents the **RAM capacity model** added to the placement problem of
Eq. (1) and, in particular, to its regional decomposition (Sec. III‑C). It
explains what the previous implementation did, why it systematically
mis‑estimates memory at the upper levels of the hierarchy, and what the two new
options change. Everything is gated behind flags whose defaults reproduce the
previous behaviour exactly.

---

## 1. Notation

| symbol | meaning |
|---|---|
| $i,j$ | nodes of the current subproblem; $j$ additionally used for *physical* nodes |
| $m$ | function index, $m=1,\dots,F$ |
| $n$ | deployment index |
| $k$ | leaf (user) index of the current subproblem |
| $\lambda_k$ | request rate of leaf $k$ |
| $H_{k,n}$ | selection: share of leaf $k$'s requests served by deployment $n$ |
| $P_{n,m,i}\in\{0,1\}$ | placement of function $m$ of deployment $n$ on node $i$ |
| $R_m$ | RAM demand of function $m$ (`ram[m]` in the code) |
| $t_m$ | base execution time of function $m$ (`Time[m]`) |
| $\sigma_{m,i}$ | speedup of function $m$ on node $i$ (`Speed[m][i]`; $1$ on edge nodes) |
| $\mathrm{RAM}^{\max}_i$ | capacity attributed to node $i$ (`ram_max[i]`) |
| $\mathcal{L}(i)$ | the **physical** nodes in the subtree rooted at node $i$ |
| $c_j$ | capacity of physical node $j\in\mathcal L(i)$ |
| $u_{n,m}$ | function‑deployment utilization (`p[...]` in the code) |
| $M_i$ | node utilization of node $i$ |
| $w_3$ | weight of the utilization cost $U$ in Eq. (1) |
| $L$ | total expected RAM load assigned to a region |
| $x_j$ | share of $L$ that a child subproblem puts on physical node $j$ |
| $\mu$ | Lagrange multiplier of the load‑conservation constraint $\sum_j x_j=L$ |

The two utilization terms of Eq. (1) are

$$
u_{n,m}=\sum_i\sum_k \lambda_k H_{k,n}\,P_{n,m,i}\,\sigma_{m,i}t_m ,
\qquad
M_i=\frac{\sum_n\sum_k\sum_m \lambda_k H_{k,n}\,P_{n,m,i}\,\sigma_{m,i}t_m R_m}{\mathrm{RAM}^{\max}_i}+\mathrm{util}_i ,
$$

both entering the objective as $w_3\sum_{n,m}u_{n,m}^2$ and $w_3\sum_i M_i^2$.
Both are dimensionless: $u_{n,m}$ is the fraction of *time* a function deployment
is busy, $M_i$ the fraction of *memory* occupied. For an $M/M/1$ queue the
expected total waiting per unit time is $\rho^2/(1-\rho)\approx\rho^2$, so each
square is a first‑order surrogate for the delay caused by its own resource —
which is why they share the weight $w_3$.

> **Note.** $M_i$ previously used the bare $t_m$ while $u_{n,m}$ used
> $\sigma_{m,i}t_m$, so $M_i$ overstated the RAM held on cloud nodes by
> $1/\sigma$. This is fixed unconditionally; it is numerically inert today
> because cloud nodes carry $\mathrm{RAM}^{\max}=10^8$.

---

## 2. The problem

### 2.1 An aggregated node is a region, not a node

`generate_cloud` builds the hierarchy top‑down and **Dirichlet‑splits a parent's
RAM over its children**, so by construction

$$
\mathrm{RAM}^{\max}_i=\sum_{j\in\mathcal L(i)}c_j=\lVert c\rVert_1 .
$$

A function, however, has to fit on **one** physical node. The sum therefore says
nothing about whether a given function can be hosted at all. On `run_141`:

| region | provider | $\lVert c\rVert_1$ | $\max_j c_j$ | functions it can physically host |
|---|---|---|---|---|
| 1 | google | 291.1 | 57.5 | 25/48 |
| 2 | tinyfaas | **265.1** | **52.3** | **22/48** |
| 3 | tinyfaas | 572.3 | 121.3 | 48/48 |

The median function needs 55.0 MB, so **more than half of all functions can never
run anywhere in region 2** — yet the level‑0 problem scores it as 265.1 of free
memory. The consequence is visible in simulation: 16–18 % of all measured
waiting time comes from *impossible fits*, i.e. functions placed on a node whose
entire capacity is smaller than the function itself, which the simulator then
admits anyway because there is no slot that could ever hold them.

### 2.2 The two penalties are level‑inconsistent

$u_{n,m}$ is **level‑invariant**: $P$ is one‑hot, so a function deployment has
exactly one server whatever the level. $M_i$ is **not**: at level 0 a "node" is a
region of ten physical nodes, and dividing by $\lVert c\rVert_1$ treats them as
one large server. Measured on `run_141`, ratio of node penalty to function
penalty, evaluated on one fixed placement:

| | level 0 (regions) | level 1 (physical nodes) | mismatch |
|---|---|---|---|
| previous ($\ell_1$ aggregate) | 0.0236 | 2.1346 | **90.4×** |

So the level that decides **which region** hosts a function is effectively blind
to memory, while the level that decides **which node** is dominated by it.

---

## 3. Eligibility (hard constraint)

For every node $i$ define the largest single hostable node

$$
\mathrm{RAM}^{\text{node}}_i=\max_{j\in\mathcal L(i)}c_j .
$$

Any physical realisation places function $m$ on some $j\in\mathcal L(i)$ with
$c_j\ge R_m$. Hence

$$
\boxed{\,R_m>\mathrm{RAM}^{\text{node}}_i \;\Longrightarrow\; P_{n,m,i}=0\,}
$$

is a **necessary** condition: it removes no realisable solution, so the upper
levels still solve a relaxation, and a child region can always host what its
parent sent it. Note it uses the **maximum**, not the sum — using
$\lVert c\rVert_1$ here would be unsound, since capacity cannot be pooled to host
one function.

Virtual placements pinned to the head node by `function_set` (Eq. (3)) are
exempt: they execute in another region and their RAM is subtracted again in
Eq. (4), so constraining them would conflict with $P_{n,m,0}=1$.

---

## 4. The capacity that normalises $M_i$: $\ell_1$ versus $\ell_2$

### 4.1 Coherence between levels

Level 0 pays $w_3M_i^2$ for putting load $L$ into region $i$. Level 1 then splits
that load and pays $w_3\sum_{j\in\mathcal L(i)}M_j^2$ with $M_j=x_j/c_j$ and
$\sum_j x_j=L$. For the upper level's decision to mean anything, its estimate
should approximate what the lower level will actually be billed.

### 4.2 What the children pay, at best

Minimise $\sum_j (x_j/c_j)^2$ subject to $\sum_j x_j=L$, $x_j\ge0$. With $\mu$
the multiplier of the load‑conservation constraint, the Lagrangian is
$\sum_j (x_j/c_j)^2-\mu\big(\sum_j x_j-L\big)$ and stationarity gives

$$
\frac{2x_j}{c_j^{2}}=\mu
\;\Longrightarrow\;
x_j=\frac{\mu c_j^{2}}{2}
\;\overset{\sum_j x_j=L}{\Longrightarrow}\;
x_j=L\,\frac{c_j^{2}}{\sum_{j'}c_{j'}^{2}},
$$

with optimal value

$$
\min_{x}\ \sum_j\Big(\frac{x_j}{c_j}\Big)^{2}=\frac{L^{2}}{\sum_j c_j^{2}}=\frac{L^{2}}{\lVert c\rVert_2^{2}} .
$$

Setting the parent's term equal to it yields

$$
\boxed{\,M_i=\frac{L}{\lVert c\rVert_2}\quad\text{instead of}\quad M_i=\frac{L}{\lVert c\rVert_1}\,}
$$

### 4.3 Worked example

A region of $c$ leaves of capacity $R$ each:

| | parent estimate | children pay (even split) |
|---|---|---|
| $\ell_1$ | $L^2/(cR)^2$ | $c\cdot\big(L/(cR)\big)^2=L^2/(cR^2)$ — **$c\times$ larger** |
| $\ell_2$ | $L^2/(\sqrt cR)^2=L^2/(cR^2)$ | $L^2/(cR^2)$ — **exact** |

So with $\ell_1$ the top level understates a region's utilization cost by roughly
the number of nodes in it. Measured on `run_141`:

| region | $\lVert c\rVert_1$ | $\lVert c\rVert_2$ | ratio | $M_i^2$ understated by |
|---|---|---|---|---|
| 2 | 265.1 | 95.4 | 2.78 | 7.7× |
| 3 | 572.3 | 218.7 | 2.62 | 6.8× |

$\lVert c\rVert_2$ is the *effective* capacity of a set of servers under a
quadratic penalty. Ten 10 MB nodes give $\sqrt{10}\cdot10=31.6$ where a single
100 MB node gives $100$, although both sum to $100$ — quadratic penalties reward
concentrated **capacity**, and $\ell_2$ quantifies what fragmentation costs.

### 4.4 Restricting to the nodes that can host the function

Combining §3 and §4, define for each node $i$ and each threshold $x$

$$
\mathrm{Cap}_i(x)=\Big\lVert\{\,c_j: j\in\mathcal L(i),\ c_j\ge x\,\}\Big\rVert_p ,
\qquad p\in\{1,2\},
$$

and normalise each function by the capacity that can actually host it:

$$
M_i=\sum_n\sum_k\sum_m \frac{\lambda_k H_{k,n}P_{n,m,i}\,\sigma_{m,i}t_mR_m}{\mathrm{Cap}_i(R_m)}+\mathrm{util}_i .
$$

The division therefore sits **inside** the sum over $m$. At the last level
$\mathcal L(i)=\{i\}$, so $\mathrm{Cap}_i(R_m)=c_i=\mathrm{RAM}^{\max}_i$ for
every hostable $m$ and the expression **reduces exactly to the previous model** —
the physical level, and hence the flattened comparison architecture, is
untouched by this option.

### 4.5 Why no explicit capacity cuts are needed

Write $o_m$ for the expected occupancy of function $m$ on node $i$. A family of
valid inequalities suggests itself: for every threshold $\theta$, functions
larger than $\theta$ can only live on nodes larger than $\theta$, so
$\sum_{m:R_m>\theta}o_m\le \mathrm{Cap}_i(\theta)$. These are **implied** by the
per‑function normalisation and need not be added:

> **Claim.** $M_i\le1$ implies $\sum_{m\in S}o_m\le\mathrm{Cap}_i(\theta)$ for
> every $\theta$, where $S=\{m: R_m>\theta\}$.
>
> *Proof.* $\mathrm{Cap}_i$ is non‑increasing. For $m\in S$ we have $R_m>\theta$,
> hence $\mathrm{Cap}_i(R_m)\le\mathrm{Cap}_i(\theta)$ and
> $$\frac{1}{\mathrm{Cap}_i(\theta)}\sum_{m\in S}o_m\le\sum_{m\in S}\frac{o_m}{\mathrm{Cap}_i(R_m)}\le M_i\le1 .\qquad\blacksquare$$

This also avoids introducing $\theta$ as an enumeration and — more importantly —
avoids imposing a **hard constraint on an expected value**. The simulator
enforces *instantaneous* concurrency, not expected occupancy, so such a
constraint would guarantee nothing while risking infeasibility; a chance
constraint would additionally introduce a violation level as a hyperparameter.
The design therefore keeps exactly one hard constraint — eligibility, which is
purely combinatorial — and one soft penalty.

---

## 5. Honest limits

* The $\ell_2$ identity of §4.2 is **exact only when every function can use every
  node**. With heterogeneous eligibility the per‑function combination of §4.4 is
  a principled but *conservative* extension: two large functions each divide by
  the same small eligible set, so $M_i$ can exceed $1$ even when they could
  time‑share. That conservatism is the pressure against cramming, but it is not
  a tight bound.
* $M_i$ is consequently a **sum of per‑function pressures**, no longer "the
  fraction of RAM used". It is still $1$ at saturation for a single function, and
  still exactly the old quantity at the physical level.
* Integral placement makes the children's true cost $\ge$ the bound, so the
  parent still solves a relaxation — the right direction for a decomposition.
* The penalty bounds a mean; it cannot guarantee anything about instantaneous
  blocking. That trade‑off remains what $w_3$ expresses.
* The correction does not close the level gap completely — measured 90.4× →
  17.4× ($\ell_1$) → 9.5× ($\ell_2$) on `run_141` — because the regions are
  heterogeneous and eligible sets differ per function.

---

## 6. Flags

Set in `settings/parameters_<N>.json`; both defaults reproduce the previous
model exactly.

| flag | values | default | effect |
|---|---|---|---|
| `ram_eligibility` | `true` / `false` | `false` | §3 — forbid a placement no physical node in the subtree could host |
| `ram_capacity_norm` | `"off"` / `"l1"` / `"l2"` | `"off"` | §4 — capacity normalising $M_i$; `"off"` keeps $\mathrm{RAM}^{\max}_i$ |

---

## 7. Files changed

| file | change |
|---|---|
| `cloud_workflow_setup.py` | `annotate_ram_profiles` writes each node's sorted physical‑leaf capacity profile (`ram_leaf_profile`) into every regional cloud config; `generate_full_cloud` writes the trivial single‑entry profile. Built from **model** capacities (tinyfaas only), since `solve_opt` gives a finite budget to tinyfaas nodes alone. |
| `optimization.py` | `ram_max_node` and `ram_denominator(i,m)` derived from the profile; eligibility constraints; the $M_i$ constraints divide per function inside the $m$‑sum and use $\sigma_{m,i}t_m$. |
| `placement.py` | reads and forwards both flags to the decomposed and the comparison solve. |

Nothing new flows **between** levels: each subproblem reads the profile from its
own cloud config, so `update_requirements_config` and the stochastic‑selection
machinery are untouched. The $M_i$ constraint keeps the same variables, term
count and nonlinearity — only the coefficients change — and eligibility removes
variables.

---

## 8. Verification

* **Profiles** match an independent bottom‑up recomputation from
  `general_config` for every node; level‑0 profiles telescope to the total
  physical tinyfaas capacity; last‑level and flattened profiles are single‑entry.
* **Physical level untouched**: the centralized/comparison solutions
  (`opt_<w>_full.json`) are **bit‑identical** across all three
  `ram_capacity_norm` values, for both eligibility settings.
* **Gating is inert**: with both flags at their defaults the per‑subproblem
  Gurobi objective matches the previous model to $2\cdot10^{-15}$ (pure
  floating‑point reassociation from moving the division inside the sum).
* **Eligibility works**: placements exceeding the largest hostable node in their
  subtree drop from 17 / 8 / 12 (norm = off / l1 / l2) to **0** in all cases.
* **Level consistency**, evaluated on one fixed placement so the incentive is
  isolated from the resulting equilibrium: level 1 is identical under all norms
  (2.1346, as the theory requires), level 0 rises 0.0236 → 0.1226 → 0.2251, and
  the level mismatch falls 90.4× → 17.4× → 9.5×.

> **Caveat on simulated cost.** The model is a nonconvex MIQCQP solved to
> `MIPGap 0`; when several placements attain the same optimum, which one Gurobi
> returns is arbitrary. Re‑solving the *identical* model under different Gurobi
> `Seed` values moves the simulated cost by up to **236** on `run_141` and
> **1347** on `run_140` (9.3 %). Simulated‑cost differences below that scale are
> not evidence; comparisons in this note are therefore made on objective values,
> structural counts, or seed‑replicated ranges.
