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

with the head node de‑conflated. Crucially, "user leaf" here means **physical
user leaf**: the `leave` flag *and* an edge (tinyfaas) provider. Then

$$
\lambda^{\mathrm{own}}_{l,r,k}=
\begin{cases}
\lambda_{l,r,0}-\nu_r, & k=0 \text{ and } \text{succ\_node}(r)\text{ is a \emph{physical} user leaf},\\[2pt]
0, & k=0 \text{ otherwise},\\[2pt]
\lambda_{l,r,k}, & k>0 .
\end{cases}
$$

The zero case covers two situations. (a) The head refines a **non‑leaf cloud
node**: its rate is *pure* placeholder, so its own demand is $0$. (b) The head
refines a **cloud‑region head with `leave=true` but a non‑edge provider**: such
"leaves" are only request *collectors*. Their apparent own rate is **phantom
demand** — an artifact of the architecture generation, which Dirichlet‑splits a
node's rate over *all* child nodes of a region, including non‑leaf ones. In
cloud regions only the head carries the `leave` flag, so the bulk of the split
demand lands on non‑leaf nodes and never reaches a physical user leaf; the
simulator (which samples only physical tinyfaas leaves) never sees any of it.
Counting it as own demand makes (ii) **structurally infeasible** (the parent
allocates $\lambda^{\mathrm{own}}$ that the child's physical leaves cannot
carry — e.g. only $2.4\%$ of it on `run_124`). With the physical‑leaf
definition, own demand telescopes **exactly** through the hierarchy: edge
regions conserve the Dirichlet split ($\sum_k$ child own $=$ parent own to
machine precision), and phantom‑head cloud regions conserve $0=0$. The
feasibility tolerance on (ii) is thus a pure floating‑point guard; a violation
is reported as a warning instead of silently dropping the constraint.

**(iii) Virtual demand as a fixed pseudo‑leaf** (continuous generalization of the
step function $\Gamma$ in (6)). The outside stream is *not* a free variable; it is
pinned to the routing induced by *all* users outside the region, built
**recursively** from the parent solution: the parent region's other physical
leaves *plus the parent's own outside stream* (users beyond the parent region,
routed by the parent's $\rho$),

$$
\tilde\nu_r=\underbrace{\sum_{k\neq \text{succ\_node}(r)}\lambda^{\mathrm{own}}_{l-1,\text{succ\_region}(r),k}}_{\text{siblings in the parent region}}
\;+\;\underbrace{\tilde\nu_{\text{succ\_region}(r)}}_{\text{parent's outside stream}},
$$

$$
\rho_{r,n}=
\frac{\sum_{k\neq \text{succ\_node}(r)}\lambda^{\mathrm{own}}_{l-1,\text{succ\_region}(r),k}\,H_{l-1,\text{succ\_region}(r),k,n}
\;+\;\tilde\nu_{\text{succ\_region}(r)}\,\rho_{\text{succ\_region}(r),n}}
{\tilde\nu_r} .
$$

At level $1$ the parent is the top problem, whose outside stream is $0$, so the
recursion grounds in the previous (non‑recursive) formula there. By induction,
$\tilde\nu_r\rho_{r,n}$ is the **actual rate of *all* users outside region $r$
choosing deployment $n$** — at *every* level: the region's inside and outside
demand always telescope to the total physical user demand,
$\sum_n(\text{flow\_rhs}_n) + \tilde\nu_r = \Lambda^{\mathrm{tot}}$. This is the
quantity that makes the *inside : outside* balance correct (Sec. 7); it is
**not** the coarse Eq. 7 placeholder
$\nu_r=\sum_k\lambda_{l-1,k}-\lambda_{l-1,\text{succ\_node}(r)}$, which additionally
lumps in non‑user (cloud) node rates and is used only to de‑conflate the head's
*own* rate in (ii). (Deployments the outside stream routes to are kept active in
the subproblem — as virtual placements if none of their functions is refined
here — so the virtual load is representable at any depth.)

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
| $\tilde\nu_r$, $\rho_{r,n}$ (per $n$) | rate and pinned routing of the virtual/outside stream; built from the parent's leaves **and the parent's own $(\tilde\nu,\rho)$** (recursion of (iii)) | (6), i.e. $\Gamma$ |
| head‑has‑own flag | whether $\text{succ\_node}(r)$ is a *physical* user leaf (sets $\lambda^{\mathrm{own}}_{l,r,0}$) | — |

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
| `optimization.py` | $H$ relaxable to $[0,1]$; own‑rate ($\lambda^{\mathrm{own}}$, physical leaves only) objective/utilization; fixed virtual pseudo‑leaf $\tilde\nu_r\rho_{r,n}$; real‑leaf conservation constraint (FP‑tolerance guard, warns on violation); no rounding on read‑back. |
| `cloud_workflow_setup.py` | store the virtual rate $\nu_r$ on head nodes; `_stochastic_selection_targets` builds the conservation RHS, the **recursive** $(\tilde\nu_r,\rho_{r,n})$ and the (physical) head‑has‑own flag; keep all $H>0$ *or outside‑routed* deployments active. |
| `placement.py` | pass the parent cloud config *and the parent workflow config* (outside stream) down; forward `relax_selection` to the decomposed and centralized solves. |
| `selection_simulation.py` | reconstruct $P^{\mathrm{full}}$ for all deployments and per‑leaf $H^{\mathrm{full}}$ from the last‑level subproblem; per‑request sampling with a dedicated RNG; Monte‑Carlo cost. |

Everything is gated on a `stochastic_selection` flag in
`settings/parameters_<N>.json`; absent/false reproduces the original
deterministic behavior exactly.

---

## 6. Verification (case‑study runs)

* **Conservation gate** (2‑level `run_126` *and* 3‑level `run_124`): for **every**
  subproblem
  $\sum_k\lambda^{\mathrm{own}}_{l,r,k}H_{l,r,k,n}=\lambda^{\mathrm{own}}_{l-1,\text{succ\_node}(r)}H_{l-1,\dots,n}$
  holds within $10^{-6}$ with the feasibility guard **never** triggered (no
  subproblem falls back to a free simplex); all $H$ rows sum to $1$; the
  top‑level policy is genuinely fractional.
* **Outside‑stream telescoping** (both runs): for every subproblem at every
  level, $\sum_n\text{flow\_rhs}_n+\tilde\nu_r=\Lambda^{\mathrm{tot}}=0.069721$
  exactly, and $\sum_n\rho_{r,n}=1$ — i.e. inside + outside demand always equals
  the total physical user demand (Sec. 7). On `run_124` this fixes, e.g.,
  `opt_0_15` (cloud region, phantom head): before the recursive stream it saw
  $\tilde\nu=0$, inside $=0.0017$ (phantom) with an infeasible conservation
  target that had to be guard‑skipped; now inside $=0$, $\tilde\nu=0.0697$,
  conservation $0=0$.
* **Per‑user decoupling:** two leaves of the *same* region are reconstructed with
  *different* distributions (e.g. $[0.886,0.114]$ vs. $[1,0]$) — impossible under
  regional uniformity (5).
* **Multi‑level generalization** (3‑level run, $200$ leaves): the reconstruction
  completes with all $H^{\mathrm{full}}_{k,\cdot}$ summing to $1$ over the
  original deployments and no index errors, confirming the active‑set chain
  composition across levels.
* **Backward compatibility:** with the flag off, the pipeline reproduces the
  pre‑stochastic deterministic run (`run_122`) **bit‑for‑bit** in all placement
  solutions and workflow configs (cloud configs gain only the additive
  `virtual_request_rate` / persisted `L` bookkeeping fields).

## 7. Modelling caveat

Within a subproblem the **inside : outside weighting is exact — at every
level**. For any deployment $n$ whose functions are hosted in region $r$ —
through which *all* users choosing $n$ must pass — the two loads are

$$
\text{inside}=\lambda^{\mathrm{own}}_{l-1,\text{succ\_node}(r)}H_{l-1,\dots,n}
\ \ (\text{own users choosing }n),\qquad
\text{outside}=\tilde\nu_r\rho_{r,n}
\ \ (\text{all other users choosing }n),
$$

because the virtual rate $\tilde\nu_r$ is built **recursively** (Sec. 3.3) from
the parent's other physical leaves *plus the parent's own outside stream*, so
by induction it equals the true demand of *all* users outside region $r$ and
$\text{inside}+\tilde\nu_r=\Lambda^{\mathrm{tot}}$ at every level. Two earlier
approximations are thereby removed:

* $\tilde\nu_r$ is **not** the coarse Eq. 7 placeholder $\nu_r=$`request_overall`,
  which lumped in non‑user cloud rates and inflated outside by $\sim\!3\times$
  on run_126;
* $\tilde\nu_r$ is **not** truncated to the parent region either (the
  non‑recursive version used at first): that undercounted outside demand at
  levels $\geq 2$ — most severely in deep cloud regions, which have no sibling
  leaves and thus saw $\tilde\nu_r\!=\!0$, i.e. *no* utilization load at all
  for their placement decisions, although in truth *all* users are outside
  such regions.

So the optimizer weighs an inside and an outside request for a locally hosted
function **equally**, and the placement is not biased toward or away from
outside demand at any depth.

The **only** remaining approximation is the **absolute** magnitude *summed across
regions*: the same outside demand is still added to *every* region that could
host $n$'s functions (a deliberate worst‑case placeholder — outside demand is not
conserved across *sibling* regions), so per‑level cost totals are inflated. This is
inconsequential here because the outside stream feeds only the heuristic $w_3$
utilization/cost *penalties* that guide placement — the **simulation never uses
it** (it samples real per‑leaf demand through the reconstructed placement), and
the placement decision only depends on the *relative* inside/outside balance,
which is now correct. The natural refinement, if ever needed, is to also condition
$\tilde\nu_r$ on whether region $r$ actually hosts $n$'s functions, making the
outside stream conserved across sibling regions too.

**A related generation artifact, documented for completeness:** the random
architecture assigns request rates to *non‑leaf* nodes (at level 0) and
Dirichlet‑splits every node's rate over **all** child nodes, including non‑leaf
ones. Demand that lands on a node that never spawns a physical (tinyfaas) leaf
is **phantom**: it inflates the upper‑level rates and the Eq. 7 placeholder,
but no user request is ever simulated for it. The stochastic decomposition
therefore defines own demand on *physical* leaves only (Sec. 3.3 (ii)); the
phantom rates are simply ignored, exactly as the simulator ignores them.
