# From a deterministic selection to a stochastic routing policy

This note documents the changes introduced to turn the **selection variable
$H_{k,n}$ from a deterministic, one‑hot decision into a stochastic routing
policy**, both for the centralized problem (1) and for the regional
decomposition of Sec. III‑C. Notation follows the paper (Table 1): $k$ indexes
leaves (users), $i,j$ nodes, $m$ functions, $n$ deployments, $l$ levels, $r$
regions, $c$ branches; $\lambda_k$ is the request rate of leaf $k$.

Only the **selection** $H$ is relaxed. The **placement** $P_{n,m,i}\in\{0,1\}$
stays binary throughout — every deployment is still placed deterministically;
what becomes stochastic is *which deployment a user request is routed to*.

---

## 1. Motivation

In Sec. II‑A the paper already observes that a user's workflow choice
$H_{k,n}$ is *"in general not known in advance … a runtime decision made during
operation"*, and that modelling it exactly leads to a **two‑stage stochastic
program** which is intractable at scale. The paper therefore assumes a
**deterministic** policy

$$
H_{k,n}\in\{0,1\},\qquad \sum_{n}H_{k,n}=1\quad\forall k ,
$$

i.e. every user of leaf $k$ deterministically picks one deployment. This note
takes the deferred step and relaxes $H$ to a **stochastic policy**

$$
\boxed{\,H_{k,n}\in[0,1],\qquad \sum_{n}H_{k,n}=1\quad\forall k\,}
$$

so that $H_{k,n}$ is the **probability that a request originating at leaf $k$ is
routed to deployment $n$**. Requests are then sampled from this policy during
online operation (Sec. 4).

---

## 2. Centralized problem (1)

### 2.1 The objective is already an expectation over $H$

Every cost term of (1) is weighted by $\lambda_k H_{k,n}$ and summed over
$k,n$. With the relaxed $H$ and $\sum_n H_{k,n}=1$, each such sum becomes a
genuine **expected cost** under the routing policy,

$$
\sum_{n}\lambda_k H_{k,n}\,g_{n} \;=\; \lambda_k\,\mathbb{E}_{n\sim H_{k,\cdot}}\!\big[g_{n}\big],
$$

for any per‑deployment quantity $g_n$ (monetary $M$, basic time $B$). Hence
**the structure of (1) is unchanged** — only the domain $H_{k,n}\in\{0,1\}$ is
relaxed to $H_{k,n}\in[0,1]$.

### 2.2 The utilization term makes the policy non‑degenerate

If only the linear monetary/time terms were present, the relaxed program would
be an LP in $H$ whose optimum sits at a vertex — i.e. it would collapse back to a
one‑hot selection. What keeps the policy genuinely *spread* is the **quadratic
utilization cost** $w_3\big(\sum_{n,m}u_{n,m}^2+\sum_i M_i\big)$ of (1):
concentrating all of a leaf's mass on a single deployment inflates the function
utilization

$$
u_{n,m}=\sum_k\lambda_k H_{k,n}P_{n,m,i}\,t_{m,i},
$$

so the optimizer *balances load by splitting $H_{k,\cdot}$ across deployments*.
This is exactly why sampling from the resulting policy reduces the real blocking
seen in simulation: the $w_3$ penalty, previously an *approximation* of
congestion, now becomes **the mechanism that generates a blocking‑reducing
policy**. The $w_3$ terms are therefore kept unchanged.

### 2.3 Summary of centralized changes

* $H$ declared continuous on $[0,1]$ (was binary); constraint $\sum_n H_{k,n}=1$
  unchanged; the solver keeps the continuous optimum (no rounding on read‑back).
* Objective, $S_{n,m,i}$, $T_n$, $M_i$, $u_{n,m}$ constraints unchanged.

---

## 3. Decomposition (Sec. III‑C)

### 3.1 The coupling that had to be removed

In the decomposition, selection is solved **once per region**. For a lower‑level
subproblem (region $r$ at level $l$), the paper inherits the parent's choice and
imposes *regional uniformity*:

$$
\text{(5)}\quad H_{l,r,k,n}=1 \ \ \forall k
\quad\text{if}\quad H_{l-1,\text{succ\_region}(r),\text{succ\_node}(r),n}=1 ,
$$

$$
\text{(6)}\quad H_{l,r,0,n}=\Gamma\!\Big(\textstyle\sum_{k}H_{l-1,\text{succ\_region}(r),k,n}\Big),
$$

with $\Gamma(\cdot)$ the step function. Equation (5) forces **all users of a
region onto the same deployment** — the paper itself notes this *"coupling …
typically increases overall cost, particularly regarding selection‑related
terms, since regional uniformity restricts flexibility among users"*. Relaxing
$H$ is meaningful only if this coupling is removed so that leaves of one region
may route **differently**.

### 3.2 The head node conflates two demand streams

By (7) the region head node $0$ carries **both** its own users and a placeholder
for the outside/virtual demand routed through it,

$$
\text{(7)}\quad \lambda_{l,r,0}=\underbrace{\lambda^{\mathrm{own}}_{l,r,0}}_{\text{own users}}
+\underbrace{\nu_r}_{\text{virtual}},\qquad
\nu_r=\sum_{k}\lambda_{l-1,k}-\lambda_{l-1,\text{succ\_node}(0)} .
$$

A **single** variable $H_{l,r,0,\cdot}$ cannot represent both streams at once:
the own users must obey demand conservation to the parent, whereas the virtual
demand's routing is fixed by users in *other* regions. This is the crux of the
relaxation and is resolved by **de‑conflating node $0$**.

### 3.3 The three ingredients replacing (5)/(6)

For every lower‑level subproblem we impose:

**(i) Per‑leaf simplex (continuous).** Each leaf is a valid distribution:

$$
\sum_{n}H_{l,r,k,n}=1,\qquad H_{l,r,k,n}\in[0,1]\qquad\forall k .
$$

**(ii) Real‑leaf demand conservation** (replaces (5)). Using *own* rates only,
the region's real users must reproduce, per deployment, the demand the parent
allocated to this region:

$$
\boxed{\;\sum_{k}\lambda^{\mathrm{own}}_{l,r,k}\,H_{l,r,k,n}
=\lambda^{\mathrm{own}}_{l-1,\text{succ\_node}(r)}\;
H_{l-1,\text{succ\_region}(r),\text{succ\_node}(r),n}\qquad\forall n\;}
$$

with the head node de‑conflated,

$$
\lambda^{\mathrm{own}}_{l,r,k}=
\begin{cases}
\lambda_{l,r,0}-\nu_r, & k=0 \text{ and } \text{succ\_node}(r)\text{ is a user leaf},\\[2pt]
0, & k=0 \text{ and } \text{succ\_node}(r)\text{ is }\emph{not}\text{ a user leaf},\\[2pt]
\lambda_{l,r,k}, & k>0 .
\end{cases}
$$

The second case (head refines a non‑leaf cloud node) is essential: there the
head's rate is *pure* placeholder, so its own demand is $0$ and its whole rate
becomes virtual. Without this distinction the constraint is **infeasible**
(a positive own rate must sum to $1$ over $n$, but its target is $0$).

**(iii) Virtual demand as a fixed pseudo‑leaf** (continuous generalization of the
step function $\Gamma$ in (6)). The outside stream is *not* a free variable; it is
pinned to the routing induced by the other regions in the parent solution,

$$
\rho_{r,n}=
\frac{\sum_{k\neq \text{succ\_node}(r)}\lambda^{\mathrm{own}}_{l-1,\text{succ\_region}(r),k}\,H_{l-1,\text{succ\_region}(r),k,n}}
{\sum_{k\neq \text{succ\_node}(r)}\lambda^{\mathrm{own}}_{l-1,\text{succ\_region}(r),k}},
\qquad
\tilde\nu_r=
\begin{cases}\nu_r,&\text{head has own users},\\ \lambda_{l,r,0},&\text{otherwise.}\end{cases}
$$

The subproblem objective then uses $\lambda^{\mathrm{own}}$ on the real leaves and
adds, for each $n$, a **constant** contribution weighted by $\tilde\nu_r\rho_{r,n}$
(no $H$ variable). For example the utilization becomes

$$
u_{n,m}=\underbrace{\sum_{k}\lambda^{\mathrm{own}}_{l,r,k}H_{l,r,k,n}P_{n,m,i}t_{m,i}}_{\text{own users (free }H)}
+\underbrace{\tilde\nu_r\,\rho_{r,n}\,P_{n,m,i}t_{m,i}}_{\text{virtual (fixed)}},
$$

and analogously for the monetary/time terms — where the virtual stream **enters
and exits through the head node $0$**, i.e. it uses $L_{0,i}$ and $D_{m,i,0}$
instead of the user‑specific $L_{k,i},\,D_{m,i,k}$ (all outside communication is
routed through the head).

**Feasibility.** Summing (ii) over $n$ and using the simplex gives
$\sum_n(\text{RHS})=\lambda^{\mathrm{own}}_{l-1,\text{succ\_node}(r)}=\sum_k\lambda^{\mathrm{own}}_{l,r,k}$,
so the conservation and simplex constraints are mutually consistent; a feasible
policy always exists (e.g. every leaf copies the parent distribution,
$H_{l,r,k,n}=H_{l-1,\dots,n}$). This consistency **requires** keeping every
deployment the parent routes to "alive" in the subproblem (Sec. 3.5).

### 3.4 Real vs. virtual placements (unchanged, but now central)

The placement bookkeeping of (3)/(4) is unchanged and interacts with the new
selection as follows.

* **Real placement.** A function $m$ of deployment $n$ solved *inside* this
  region sits on an actual node $i$: $P_{l,r,n,m,i}=1$.
* **Virtual placement (3).** A function placed in *another* region appears here
  only as a placeholder pinned to the head node,
  $P_{l,r,n,m,0}=1\ \forall m\in\mathcal M$, because every inter‑region route
  passes through node $0$. These do not consume this region's RAM — hence the
  correction of the head utilization $M_{l,r,0}$ in (4).
* **Active set / pruning.** Deterministically, a region only keeps the deployment
  it selected. Under the stochastic policy a leaf may route to **any** deployment
  with $H_{l-1,\cdot,n}>0$, so **every such deployment must be kept active** in
  the subproblem — otherwise conservation (ii) cannot be satisfied. Deployments
  with no function solved locally are kept purely through their virtual
  placements (all functions in $\mathcal M$).

### 3.5 What is passed between levels

The parent solution is pushed to each child region. The one‑hot inheritance of
(5)/(6) is replaced by three quantities derived from the parent's $H$ and rates:

| passed down | meaning | replaces |
|---|---|---|
| $\;\lambda^{\mathrm{own}}_{l-1,\text{succ\_node}(r)}\,H_{l-1,\dots,n}\;$ (per $n$) | conservation target for the region's own users, RHS of (ii) | (5) |
| $\rho_{r,n}$ (per $n$) | pinned routing of the virtual/outside stream | (6), i.e. $\Gamma$ |
| head‑has‑own flag | whether $\text{succ\_node}(r)$ is a real user leaf (sets $\lambda^{\mathrm{own}}_{l,r,0}$) | — |

The **placement** $P$ is passed down exactly as before (which functions are
refined here vs. virtual). Selections are **not** recomputed one‑hot; instead the
child freely re‑optimizes per‑leaf distributions subject to the conservation
target — this is precisely what re‑introduces the per‑user flexibility that
regional uniformity had removed.

---

## 4. Simulation

### 4.1 Per‑request sampling and Monte‑Carlo cost

Requests still arrive by a Poisson process of rate $\lambda_k$ at every leaf $k$
over $t_{\mathrm{sim}}$ seconds. Previously all requests of a leaf used the one
selected deployment; **now each incoming request draws its deployment**

$$
n \sim H_{k,\cdot}\quad(\text{categorical}),
$$

executes the functions of deployment $n$ on their placed nodes under the two
blocking rules (one request per function–deployment; node RAM cap), and on a
block simply **waits** (FIFO). The realized cost is Monte‑Carlo: for each request
the sampled deployment contributes its monetary cost, and the response time
(basic time $B$ plus waiting $W$) is measured, giving

$$
C=w_1 M + w_2 (B+W),\qquad
M=\sum_{k}\sum_{n}\,c^{\mathrm{count}}_{k,n}\,M_{k,n},
$$

where $c^{\mathrm{count}}_{k,n}$ is the number of requests of leaf $k$ that
sampled deployment $n$ (so $M$ is an unbiased estimate of
$\sum_k\sum_n\lambda_k H_{k,n}M_{k,n}$). A separate seeded RNG is used for the
deployment draws so the Poisson stream is identical to the deterministic run.

### 4.2 Reassembly into a flat (centralized‑equivalent) solution

The simulator does **not** run on the hierarchy directly. It first *reassembles*
a decomposed solution into the **same flat form as a centralized solution** — a
placement $P^{\mathrm{full}}_{n,m,i}$ on physical nodes and a per‑leaf policy
$H^{\mathrm{full}}_{k,n}$ — via `reconstruct_full_solution`. Two things differ
from the deterministic reconstruction:

* **Placement for *all* deployments.** Because any deployment with $H^{\mathrm{full}}_{k,n}>0$
  may be sampled, $P^{\mathrm{full}}$ is built for every deployment $n$ (not only
  the selected one) by following each $n$ down its placement tree with the
  existing per‑level maps.
* **Per‑leaf distribution from the *lowest* subproblem.** The deterministic code
  reads a single regional selection from the top problem. Under the stochastic
  policy the leaves of a region differ, so each leaf's distribution is read from
  the subproblem at the **last** level $l=n_l-1$ where the leaf physically lives:

$$
H^{\mathrm{full}}_{k,n}=H_{\,n_l-1,\;r(k),\;\kappa(k),\;\tilde n} ,
$$

  where $r(k)$ is $k$'s last‑level region, $\kappa(k)$ its local leaf index, and
  $\tilde n$ is the **local** deployment index that maps to the original index
  $n$ by composing the per‑level active sets up the chain
  $l=n_l-1\!\to\!\dots\!\to\!0$. Reading the top problem would be far too coarse
  (there each row already aggregates a whole level‑1 region).

### 4.3 Is the decomposed simulation the same as the centralized one?

**Yes — the simulation *engine* is identical, and by design.** After
reassembly both solutions are described by the same objects
$(P^{\mathrm{full}}, H^{\mathrm{full}}, L)$ on the physical architecture, and the
**exact same** discrete‑event simulator runs on both: same Poisson arrivals,
same rule $n\sim H^{\mathrm{full}}_{k,\cdot}$ per request, same FIFO blocking,
same Monte‑Carlo cost $C=w_1M+w_2(B+W)$.

The only difference is **where $H^{\mathrm{full}}$ comes from**:

* **Centralized:** $H^{\mathrm{full}}_{k,n}=H_{k,n}$ is read directly — the
  centralized problem already solves one policy per physical leaf.
* **Decomposed:** the decomposition never solves a single global $H$; it solves a
  policy per region and level. $H^{\mathrm{full}}_{k,n}$ is therefore
  *reconstructed* per leaf from the last‑level subproblem, and the
  cross‑level conservation constraints (ii) guarantee that these reconstructed
  per‑leaf distributions **aggregate**, region by region, to what the upper
  levels decided — while still being allowed to differ between leaves of the same
  region.

So the decomposed result does *build up* selections and placements into the
structure of a centralized solution, and the simulation is common to both; the
decomposition only changes how that structure is *produced* (and, for selection,
adds the per‑leaf freedom that the centralized problem has natively but that
regional uniformity used to forbid). In the degenerate case where every
reconstructed $H^{\mathrm{full}}_{k,\cdot}$ is one‑hot, the run reduces exactly to
the previous deterministic simulation.

---

## 5. Files changed

| file | change |
|---|---|
| `optimization.py` | $H$ relaxable to $[0,1]$; own‑rate ($\lambda^{\mathrm{own}}$) objective/utilization; fixed virtual pseudo‑leaf $\tilde\nu_r\rho_{r,n}$; real‑leaf conservation constraint; no rounding on read‑back. |
| `cloud_workflow_setup.py` | store the virtual rate $\nu_r$ on head nodes; `_stochastic_selection_targets` builds the conservation RHS, $\rho_{r,n}$ and the head‑has‑own flag; keep all $H>0$ deployments active. |
| `placement.py` | pass the parent cloud config down; forward `relax_selection` to the decomposed and centralized solves. |
| `selection_simulation.py` | reconstruct $P^{\mathrm{full}}$ for all deployments and per‑leaf $H^{\mathrm{full}}$ from the last‑level subproblem; per‑request sampling with a dedicated RNG; Monte‑Carlo cost. |

Everything is gated on a `stochastic_selection` flag in
`settings/parameters_<N>.json`; absent/false reproduces the original
deterministic behavior exactly.

---

## 6. Verification (case‑study runs)

* **Conservation gate** (2‑level run): for every subproblem
  $\sum_k\lambda^{\mathrm{own}}_{l,r,k}H_{l,r,k,n}=\lambda^{\mathrm{own}}_{l-1,\text{succ\_node}(r)}H_{l-1,\dots,n}$
  holds exactly; all $H$ rows sum to $1$; the top‑level policy is genuinely
  fractional.
* **Per‑user decoupling:** two leaves of the *same* region are reconstructed with
  *different* distributions (e.g. $[0.886,0.114]$ vs. $[1,0]$) — impossible under
  regional uniformity (5).
* **Multi‑level generalization** (3‑level run, $200$ leaves): the reconstruction
  completes with all $H^{\mathrm{full}}_{k,\cdot}$ summing to $1$ over the
  original deployments and no index errors, confirming the active‑set chain
  composition across levels.
* **Backward compatibility:** with the flag off, the deterministic decomposition
  and its simulation are reproduced bit‑for‑bit.

## 7. Modelling caveat

The virtual rate $\nu_r$ (the Eq. 7 placeholder) aggregates rates over nodes that
are not all user leaves; consequently the **real‑leaf** conservation (ii) is
*exact*, while the **virtual** stream's magnitude/routing $(\tilde\nu_r,\rho_{r,n})$
is an approximation of the true outside demand — of the same order as the
original Eq. 6/7 aggregation. This is adequate for the utilization/cost accounting
but is the natural place to refine if finer outside‑demand fidelity is needed.
