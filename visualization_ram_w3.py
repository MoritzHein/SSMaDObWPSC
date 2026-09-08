# Visualization of the RAM-model x w_3 x policy study.
#
# Usage: visualization_ram_w3.py [--manifest=settings/ram_w3_campaign.json]
#                                [--export=results/run_<base_run>]
#                                [--cell=l2,1.5]
#
# Two figures are written.
#
# ram_w3_comparison - four stacked rows, one pair (absolute cost / paired
# difference) per RAM capacity model:
#
#     row 1  absolute cost, RAM model OFF (aggregate capacity, previous model)
#     row 2  paired difference stochastic - deterministic, RAM model OFF
#     row 3  absolute cost, eligibility + l2 capacity norm
#     row 4  paired difference stochastic - deterministic, eligibility + l2
#
# The absolute rows stack every bar into the three weighted cost contributions
#
#     w_1 * money_cost   +   w_2 * time_cost   +   w_2 * complete_time_cost
#     (monetary Z)           (basic time B)        (waiting time W)
#
# so a bar's height is the overall cost and all four requested quantities are
# visible at once. The basic time is NOT the reported time_cost, which is the
# ANALYTIC one (the critical path traced through the hierarchy, weighted by the
# sampled request counts) and sits about 3.7% below the realized one. Since
#
#     overall_cost = w_1 * money_cost + w_2 * sum(realized response times)
#
# and money and waiting enter it exactly as reported, the realized basic time is
# the remainder, w_2 B = overall_cost - w_1 Z - w_2 W. See weighted_split.
#
# The difference rows exist because the effect of the selection policy (2-3% of
# the overall cost) is far smaller than the Monte-Carlo spread of a single
# configuration (~6% of it). Both policies are simulated on the SAME request
# streams - selection_simulation.py draws the Poisson arrivals from `rng` and the
# per-request deployment from the separate `rng_sel`, precisely so the streams
# stay identical - so the comparison is paired and the per-stream difference is
# the statistic with the small variance. Its 95% confidence interval is roughly
# four times narrower than the interval on either policy alone, which is what
# makes the effect visible. Error bars therefore mean different things in the two
# kinds of row: unpaired CI of one configuration above, paired CI of the
# difference below.
#
# ram_w3_latency - the mechanism behind the difference: the response-time
# distribution of one representative cell (--cell, default the l2 model at
# w_3=1.5), deterministic against stochastic, pooled over all request streams,
# plus its p50/p90/p95/p99 tail.
#
# Both figures need the per-stream data produced by resimulate_seeds.py
# (results_seeds.json and latency_seeds.json). Without a results_seeds.json the
# cost figure falls back to the single results.json, draws no error bars and no
# difference rows (a single simulation carries roughly 10% Monte-Carlo noise, so
# treat such a figure as indicative only).
import argparse
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import numpy as np

# Reuse the paper plotting style when available, but degrade gracefully so the
# script runs without a LaTeX toolchain.
try:
    import scienceplots  # noqa: F401
    plt.style.use(["science", "no-latex"])
except Exception:
    pass

DEFAULT_MANIFEST = os.path.join("settings", "ram_w3_campaign.json")

# Paul Tol muted, shared with visualization_policy.py so both paper figures use
# one palette. The C0/C1/C2 triple this used to carry fails a colour-vision
# check - orange against green is dE 4.2 under protanopia, against a target 8.
COMPONENTS = [
    ("money_cost", "$w_1 Z$ (monetary)", "#332288"),
    ("time_cost", "$w_2 B$ (basic time)", "#DDCC77"),
    ("complete_time_cost", "$w_2 W$ (waiting)", "#CC6677"),
]
POLICY_HATCH = {"det": "", "stoch": "///"}

# The reported `time_cost` is NOT the basic time the cost is made of. It is the
# ANALYTIC one - the critical path of the workflow times the number of requests
# of the (workflow, leaf) pair, accumulated in the reporting block of
# selection_simulation.py - whereas
#
#     overall_cost = w_1 * money_cost + w_2 * sum(realized response times).
#
# Money and waiting, by contrast, enter overall_cost exactly as reported. So the
# whole discrepancy sits in the basic time, and the realized basic time is the
# remainder, w_2 B = overall_cost - w_1 Z - w_2 W. That identity is exact: the
# simulator sets wait = max(0, response - base_T), so response - wait = base_T
# unless a request finishes faster than its own base time, which happens in none
# of the 138 campaign runs (it does in the old runs 123-137, at up to 0.3 %).
# Taking the remainder therefore makes the stack sum to the measured cost
# without inventing a residual segment, and it plots the realized basic time
# instead of an analytic estimate that is about 3.7 % below it.
# The two helpers that do this are weighted_split and weighted_split_diff,
# defined below next to the statistics they call.

# IEEE two-column width, from config_mpl.py. The paper figure is one column
# wide, so the four modes share an axis instead of sitting in two panels each.
COLUMNWIDTH = 3.44488

# The four modes of the compact paper figure, in the order they are drawn. The
# bar style separates them by hatch, because colour is spent on the three cost
# contributions; the marker style separates them by hue (memory model) and fill
# (selection policy), because it has no stack to colour.
#
# The hatches lean forward for the aggregate model and backward or crossed for
# the corrected one, so the memory model is readable before the legend is. Dots
# were the fourth pattern here and disappeared at a bar width of a fifth of a
# tick, hence the backslashes. "\\\\\\" is three backslash characters.
PAPER_MODES = [
    ("off", "det", "aggregate, det.", "", "#4477AA", "o", "none"),
    ("off", "stoch", "aggregate, stoch.", "///", "#4477AA", "o", "full"),
    ("l2", "det", "elig. + $\\ell_2$, det.", "\\\\\\", "#AA3377", "s", "none"),
    ("l2", "stoch", "elig. + $\\ell_2$, stoch.", "xxx", "#AA3377", "s", "full"),
]

# Panel (b) shows one paired difference per memory model, not per mode. It takes
# the hatch of that model's STOCHASTIC bar in panel (a), because the difference
# is what the stochastic policy adds to that model.
PAPER_DIFF = [
    ("off", "aggregate", "#4477AA", "o", "///"),
    ("l2", "elig. + $\\ell_2$", "#AA3377", "s", "xxx"),
]

# Hatch strokes at their default 1.0 pt fill a bar of this width into a solid
# block. Thinner strokes let the pattern read at both bar and legend size.
plt.rcParams["hatch.linewidth"] = 0.5

# t quantiles at n-1 degrees of freedom. visualization_policy.py reports the
# same ten paired streams, so both scripts have to use the same quantile; 1.96
# understates the interval at n = 10 by 15 per cent.
T_QUANTILE = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
              7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
              14: 2.145, 19: 2.093, 29: 2.045}


def t_quantile(df):
    if df in T_QUANTILE:
        return T_QUANTILE[df]
    return T_QUANTILE[min(T_QUANTILE, key=lambda k: abs(k - df))]


QUANTILES = [0.5, 0.9, 0.95, 0.99]


def load_run(num_run):
    """Per-stream cost components of one run; empty if the run has no results."""
    base = os.path.join("results", f"run_{num_run}")
    seeds_path = os.path.join(base, "results_seeds.json")
    single_path = os.path.join(base, "results.json")
    if os.path.exists(seeds_path):
        with open(seeds_path, "r") as f:
            return json.load(f), True
    if os.path.exists(single_path):
        with open(single_path, "r") as f:
            return [json.load(f)], False
    return [], False


def load_latency(num_run):
    """Per-stream response/waiting times of one run, as written by resimulate_seeds.py."""
    path = os.path.join("results", f"run_{num_run}", "latency_seeds.json")
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        return json.load(f)


def stats(rows, key):
    v = [r[key] for r in rows if key in r and r[key] is not None]
    if not v:
        return float("nan"), 0.0
    mean = sum(v) / len(v)
    if len(v) < 2:
        return mean, 0.0
    sd = math.sqrt(sum((x - mean) ** 2 for x in v) / (len(v) - 1))
    return mean, t_quantile(len(v) - 1) * sd / math.sqrt(len(v))


def paired_rows(det_run, stoch_run):
    """(deterministic, stochastic) result rows joined on the request stream.

    Both runs were simulated on the same list of simulation seeds, so the join
    key is the seed. Rows written before the seeds were recorded fall back to
    positional pairing, which is the same thing as long as both runs used the
    manifest's seed order.
    """
    det, det_seeds = load_run(det_run)
    stoch, stoch_seeds = load_run(stoch_run)
    if not (det and stoch and det_seeds and stoch_seeds):
        # A single results.json per run would pair into one difference with a
        # zero-width confidence interval, which reads as certainty it is not.
        return []
    if all("simulation_seed" in r for r in det + stoch):
        d = {r["simulation_seed"]: r for r in det}
        s = {r["simulation_seed"]: r for r in stoch}
        return [(d[k], s[k]) for k in sorted(set(d) & set(s))]
    return list(zip(det, stoch))


def paired_diff(pairs, key):
    """Mean of (stochastic - deterministic) per stream, its 95% CI, n and wins."""
    v = [s[key] - d[key] for d, s in pairs
         if d.get(key) is not None and s.get(key) is not None]
    if not v:
        return float("nan"), 0.0, 0, 0
    mean = sum(v) / len(v)
    wins = sum(1 for x in v if x < 0)
    if len(v) < 2:
        return mean, 0.0, len(v), wins
    sd = math.sqrt(sum((x - mean) ** 2 for x in v) / (len(v) - 1))
    return mean, t_quantile(len(v) - 1) * sd / math.sqrt(len(v)), len(v), wins


def weighted_split(rows, w1, w2):
    """The three weighted contributions, summing exactly to the measured cost."""
    z = w1 * stats(rows, "money_cost")[0]
    wait = w2 * stats(rows, "complete_time_cost")[0]
    return {"money_cost": z,
            "time_cost": stats(rows, "overall_cost")[0] - z - wait,
            "complete_time_cost": wait}


def weighted_split_diff(pairs, w1, w2):
    """Same split for a paired difference; every term is linear in the rows."""
    z = w1 * paired_diff(pairs, "money_cost")[0]
    wait = w2 * paired_diff(pairs, "complete_time_cost")[0]
    return {"money_cost": z,
            "time_cost": paired_diff(pairs, "overall_cost")[0] - z - wait,
            "complete_time_cost": wait}


def stacked_signed(ax, xs, series, width, colors, hatch=None, linewidth=0.4):
    """Stack signed contributions: positives grow up, negatives grow down."""
    up = np.zeros(len(xs))
    down = np.zeros(len(xs))
    for vals, color in zip(series, colors):
        vals = np.asarray(vals, dtype=float)
        pos = np.where(vals > 0, vals, 0.0)
        neg = np.where(vals < 0, vals, 0.0)
        ax.bar(xs, pos, width, bottom=up, color=color, edgecolor="black",
               linewidth=linewidth, hatch=hatch)
        ax.bar(xs, neg, width, bottom=down, color=color, edgecolor="black",
               linewidth=linewidth, hatch=hatch)
        up = up + pos
        down = down + neg


def quantile(values, p):
    if not values:
        return float("nan")
    v = sorted(values)
    k = (len(v) - 1) * p
    lo, hi = math.floor(k), math.ceil(k)
    return v[lo] if lo == hi else v[lo] * (hi - k) + v[hi] * (k - lo)


def save(fig, export_path, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(export_path, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}.pdf / .png")


def cost_figure(man, cell, export_path):
    """Absolute stacked costs plus the paired stochastic-deterministic difference."""
    w1, w2 = man["w_1"], man["w_2"]
    w3_values = man["w3_values"]
    ram_settings = man["ram_settings"]
    policies = man["policies"]
    weight = {"money_cost": w1, "time_cost": w2, "complete_time_cost": w2}

    replicated = True
    missing = []
    nrows = 2 * len(ram_settings)
    fig, axes = plt.subplots(nrows, 1, figsize=(7.15, 4.3 * len(ram_settings)),
                             gridspec_kw={"height_ratios": [3, 1.5] * len(ram_settings)})
    axes = np.atleast_1d(axes)

    x = np.arange(len(w3_values), dtype=float)
    width = 0.38
    offsets = {p["key"]: (i - (len(policies) - 1) / 2) * width
               for i, p in enumerate(policies)}

    abs_axes = [axes[2 * i] for i in range(len(ram_settings))]
    diff_axes = [axes[2 * i + 1] for i in range(len(ram_settings))]

    for ax, dax, ram in zip(abs_axes, diff_axes, ram_settings):
        # ---- absolute costs, one stacked bar per policy
        for pol in policies:
            xs = [x[wi] + offsets[pol["key"]] for wi in range(len(w3_values))]
            heights = {k: [] for k, _, _ in COMPONENTS}
            totals, errs = [], []
            for wi, w3 in enumerate(w3_values):
                num_run = cell.get((ram["key"], w3, pol["key"]))
                rows, has_seeds = load_run(num_run) if num_run else ([], False)
                if not rows:
                    missing.append(num_run)
                    for k, _, _ in COMPONENTS:
                        heights[k].append(0.0)
                    totals.append(float("nan"))
                    errs.append(0.0)
                    continue
                replicated = replicated and has_seeds
                seg = weighted_split(rows, w1, w2)
                for k, _, _ in COMPONENTS:
                    heights[k].append(seg[k])
                m, h = stats(rows, "overall_cost")
                totals.append(m)
                errs.append(h)

            bottom = np.zeros(len(w3_values))
            for k, label, color in COMPONENTS:
                vals = np.array(heights[k])
                ax.bar(xs, vals, width * 0.92, bottom=bottom, color=color,
                       edgecolor="black", linewidth=0.4,
                       hatch=POLICY_HATCH[pol["key"]],
                       label=label if (pol is policies[0] and ax is abs_axes[0]) else None)
                bottom = bottom + vals
            ax.errorbar(xs, totals, yerr=errs, fmt="_", color="black",
                        markersize=11, markeredgewidth=1.4, elinewidth=1.2,
                        capsize=3, linestyle="none", zorder=5)

        ax.set_title(ram["label"], fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{v:g}" for v in w3_values])
        ax.set_ylabel("cost")
        ax.margins(x=0.04)

        # ---- paired difference stochastic - deterministic
        comp_diffs = {k: [] for k, _, _ in COMPONENTS}
        totals, errs, labels = [], [], []
        for w3 in w3_values:
            det_run = cell.get((ram["key"], w3, "det"))
            stoch_run = cell.get((ram["key"], w3, "stoch"))
            pairs = paired_rows(det_run, stoch_run) if det_run and stoch_run else []
            if not pairs:
                for k, _, _ in COMPONENTS:
                    comp_diffs[k].append(0.0)
                totals.append(float("nan"))
                errs.append(0.0)
                labels.append("")
                continue
            seg = weighted_split_diff(pairs, w1, w2)
            for k, _, _ in COMPONENTS:
                comp_diffs[k].append(seg[k])
            m, h, n, wins = paired_diff(pairs, "overall_cost")
            base = sum(d["overall_cost"] for d, _ in pairs) / len(pairs)
            totals.append(m)
            errs.append(h)
            labels.append(f"{100 * m / base:+.1f}%\n{wins}/{n}")

        stacked_signed(dax, x, [comp_diffs[k] for k, _, _ in COMPONENTS],
                       width * 1.5, [c for _, _, c in COMPONENTS])
        dax.errorbar(x, totals, yerr=errs, fmt="D", color="black", markersize=4,
                     elinewidth=1.4, capsize=3, linestyle="none", zorder=6)
        dax.axhline(0.0, color="black", linewidth=0.8)
        dax.set_xticks(x)
        dax.set_xticklabels([f"{v:g}" for v in w3_values])
        dax.set_ylabel("$\\Delta$ cost\n(negative: stochastic better)")
        dax.margins(x=0.04)
        # Annotations go into the empty half of the panel - above the zero line
        # for a negative difference, above the marker for a positive one - so
        # they never reach into the x tick labels.
        for xi, (t, e, lab) in enumerate(zip(totals, errs, labels)):
            if not lab or math.isnan(t):
                continue
            dax.annotate(lab, (x[xi], 0.0 if t < 0 else t + e),
                         textcoords="offset points", xytext=(0, 4),
                         ha="center", va="bottom", fontsize=6.5)

    # common y-ranges: absolute rows share one, difference rows share another
    for group in (abs_axes, diff_axes):
        lo = min(a.get_ylim()[0] for a in group)
        hi = max(a.get_ylim()[1] for a in group)
        for a in group:
            a.set_ylim(lo, hi)
    for dax in diff_axes:
        lo, hi = dax.get_ylim()
        dax.axhspan(lo, 0.0, color="C2", alpha=0.06, zorder=0)
        dax.set_ylim(lo, hi)

    axes[-1].set_xlabel("utilization weight $w_3$")

    handles = [Patch(facecolor=c, edgecolor="black", label=l) for _, l, c in COMPONENTS]
    handles += [Patch(facecolor="white", edgecolor="black",
                      hatch=POLICY_HATCH[p["key"]], label=p["label"]) for p in policies]
    handles += [Line2D([0], [0], color="black", marker="_", linestyle="none",
                       markersize=11, markeredgewidth=1.4,
                       label="measured overall cost" + (" (95% CI)" if replicated else "")),
                Line2D([0], [0], color="black", marker="D", linestyle="none",
                       markersize=4,
                       label=r"$\Delta$ overall cost (paired 95% CI)")]
    n_streams = len(man["simulation_seeds"]) if replicated else 1
    fig.suptitle(f"{man['num_levels']} levels, {man['num_workflows']} workflows, "
                 f"seed {man['seed']}, $t_{{sim}}={man['simulation_time']}$ s, "
                 f"{n_streams} request stream{'s' if n_streams > 1 else ''}",
                 y=0.998, fontsize=9)
    # Legend at figure level: an axes-anchored one collides with the first
    # subplot title once the difference rows push the layout together.
    fig.legend(handles=handles, ncol=3, fontsize=8, loc="upper center",
               bbox_to_anchor=(0.5, 0.982), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.925))
    save(fig, export_path, "ram_w3_comparison")

    if missing:
        print(f"  WARNING: {len(missing)} cells had no results and were drawn "
              f"empty: {sorted(set(missing))}")
    if not replicated:
        print("  WARNING: no results_seeds.json found - bars come from a single "
              "simulation each (~10% Monte-Carlo noise) and carry no error bars. "
              "Run resimulate_seeds.py for confidence intervals.")


def latency_figure(man, cell, ram_key, w3_ref, export_path):
    """Response-time distribution and tail of one representative cell."""
    det_run = cell.get((ram_key, w3_ref, "det"))
    stoch_run = cell.get((ram_key, w3_ref, "stoch"))
    series = {}
    for pol, num_run in (("det", det_run), ("stoch", stoch_run)):
        if not num_run:
            continue
        pooled = []
        for entry in load_latency(num_run):
            pooled.extend(entry.get("response_times", []))
        if pooled:
            series[pol] = pooled
    if len(series) < 2:
        print("  skipping latency figure: no latency_seeds.json for cell "
              f"({ram_key}, w_3={w3_ref}) - run resimulate_seeds.py")
        return

    label = {p["key"]: p["label"] for p in man["policies"]}
    color = {"det": "C3", "stoch": "C0"}

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.15, 2.9))

    for pol in ("det", "stoch"):
        v = np.sort(np.asarray(series[pol], dtype=float))
        cdf = np.arange(1, len(v) + 1) / len(v)
        ax.step(v, cdf, where="post", color=color[pol], linewidth=1.3,
                label=label.get(pol, pol))
    ax.set_xlabel("response time [s]")
    ax.set_ylabel("fraction of requests")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.set_title(f"response-time CDF ({len(series['det'])} requests)", fontsize=9)

    xs = np.arange(len(QUANTILES), dtype=float)
    width = 0.38
    for i, pol in enumerate(("det", "stoch")):
        q = [quantile(series[pol], p) for p in QUANTILES]
        bx.bar(xs + (i - 0.5) * width, q, width * 0.92, color=color[pol],
               edgecolor="black", linewidth=0.4, label=label.get(pol, pol))
    for xi, p in enumerate(QUANTILES):
        qd = quantile(series["det"], p)
        qs = quantile(series["stoch"], p)
        if qd:
            bx.annotate(f"{100 * (qs - qd) / qd:+.0f}%", (xs[xi], max(qd, qs)),
                        textcoords="offset points", xytext=(0, 3), ha="center",
                        fontsize=7)
    bx.set_ylim(0, bx.get_ylim()[1] * 1.12)   # headroom for the p99 annotation
    bx.set_xticks(xs)
    bx.set_xticklabels([f"p{100 * p:g}" for p in QUANTILES])
    bx.set_ylabel("response time [s]")
    bx.set_title("tail of the same distribution", fontsize=9)
    bx.legend(fontsize=8, frameon=False, loc="upper left")

    ram_label = {r["key"]: r["label"] for r in man["ram_settings"]}
    fig.suptitle(f"{ram_label.get(ram_key, ram_key)}, $w_3={w3_ref:g}$, "
                 f"{len(load_latency(det_run))} request streams pooled", fontsize=9)
    fig.tight_layout()
    save(fig, export_path, "ram_w3_latency")


def _paper_cells(man, cell):
    """Everything both paper styles draw, extracted once.

    Returns the absolute stack and total per mode, and the paired difference per
    memory model. The difference is kept in per cent of the deterministic cost
    of the same cell: in absolute terms the aggregate model at w_3 = 0.5 is an
    order of magnitude larger than the other eleven cells and would flatten them
    into invisibility, whereas relative to its own baseline it is +10.6 against
    -4 and both fit on one axis. The paper quotes per cent anyway.
    """
    w1, w2 = man["w_1"], man["w_2"]
    weight = {"money_cost": w1, "time_cost": w2, "complete_time_cost": w2}
    w3_values = man["w3_values"]

    absolute = {}
    for ram_key, pol_key, _, _, _, _, _ in PAPER_MODES:
        comps = {k: [] for k, _, _ in COMPONENTS}
        totals, errs = [], []
        for w3 in w3_values:
            num_run = cell.get((ram_key, w3, pol_key))
            rows, _ = load_run(num_run) if num_run else ([], False)
            seg = (weighted_split(rows, w1, w2) if rows
                   else {k: 0.0 for k, _, _ in COMPONENTS})
            for k, _, _ in COMPONENTS:
                comps[k].append(seg[k])
            m, h = stats(rows, "overall_cost") if rows else (float("nan"), 0.0)
            totals.append(m)
            errs.append(h)
        absolute[(ram_key, pol_key)] = (comps, totals, errs)

    difference = {}
    for ram_key, _, _, _, _ in PAPER_DIFF:
        comps = {k: [] for k, _, _ in COMPONENTS}
        totals, errs, wins = [], [], []
        for w3 in w3_values:
            det_run = cell.get((ram_key, w3, "det"))
            stoch_run = cell.get((ram_key, w3, "stoch"))
            pairs = paired_rows(det_run, stoch_run) if det_run and stoch_run else []
            if not pairs:
                for k, _, _ in COMPONENTS:
                    comps[k].append(0.0)
                totals.append(float("nan"))
                errs.append(0.0)
                wins.append((0, 0))
                continue
            base = sum(d["overall_cost"] for d, _ in pairs) / len(pairs)
            seg = weighted_split_diff(pairs, w1, w2)
            for k, _, _ in COMPONENTS:
                comps[k].append(100 * seg[k] / base)
            m, h, n, w = paired_diff(pairs, "overall_cost")
            totals.append(100 * m / base)
            errs.append(100 * h / base)
            wins.append((w, n))
        difference[ram_key] = (comps, totals, errs, wins)

    return absolute, difference


def _paper_axes():
    """The two stacked panels both styles draw into, at one column width."""
    fig, axes = plt.subplots(2, 1, figsize=(COLUMNWIDTH, 3.4),
                             gridspec_kw={"height_ratios": [1.9, 1.0]},
                             sharex=True)
    axes[1].set_xlabel("utilization weight $w_3$")
    axes[0].set_ylabel("overall cost $C$")
    axes[1].set_ylabel(r"$\Delta C$ [\%]" if plt.rcParams.get("text.usetex")
                       else r"$\Delta C$ [%]")
    for a in axes:
        a.tick_params(labelsize=7)
        a.yaxis.label.set_size(8)
    axes[1].xaxis.label.set_size(8)
    return fig, axes


def _panel_letters(axes):
    axes[0].text(0.015, 0.965, "(a)", transform=axes[0].transAxes,
                 fontsize=8, va="top", ha="left")
    axes[1].text(0.015, 0.05, "(b)", transform=axes[1].transAxes,
                 fontsize=8, va="bottom", ha="left")


def _zero_band(dax):
    """Shade the half of the difference panel in which the relaxation wins."""
    dax.axhline(0.0, color="black", linewidth=0.7)
    lo, hi = dax.get_ylim()
    dax.axhspan(lo, 0.0, color="0.85", alpha=0.35, zorder=0)
    dax.set_ylim(lo, hi)


def paper_cost_figure_bars(man, cell, export_path):
    """Compact paper figure, bar style: four stacked bars per utilization weight.

    The four modes are one grouped bar each, separated by hatch, so the three
    cost contributions keep the colour channel. Panel (b) carries only two bars
    per weight - one paired difference per memory model - and therefore stays
    readable at a single column width even though panel (a) is dense.
    """
    w3_values = man["w3_values"]
    absolute, difference = _paper_cells(man, cell)
    fig, axes = _paper_axes()
    ax, dax = axes
    x = np.arange(len(w3_values), dtype=float)

    width = 0.20
    for i, (ram_key, pol_key, _, hatch, _, _, _) in enumerate(PAPER_MODES):
        comps, totals, errs = absolute[(ram_key, pol_key)]
        xs = x + (i - (len(PAPER_MODES) - 1) / 2.0) * width
        bottom = np.zeros(len(w3_values))
        for k, _, color in COMPONENTS:
            vals = np.array(comps[k])
            ax.bar(xs, vals, width * 0.9, bottom=bottom, color=color,
                   edgecolor="black", linewidth=0.3, hatch=hatch)
            bottom = bottom + vals
        ax.errorbar(xs, totals, yerr=errs, fmt="_", color="black",
                    markersize=4.0, markeredgewidth=0.8, elinewidth=0.7,
                    capsize=1.0, linestyle="none", zorder=5)

    dwidth = 0.34
    for i, (ram_key, _, _, marker, hatch) in enumerate(PAPER_DIFF):
        comps, totals, errs, _ = difference[ram_key]
        xs = x + (i - (len(PAPER_DIFF) - 1) / 2.0) * dwidth
        stacked_signed(dax, xs, [comps[k] for k, _, _ in COMPONENTS],
                       dwidth * 0.9, [c for _, _, c in COMPONENTS],
                       hatch=hatch)
        dax.errorbar(xs, totals, yerr=errs, fmt=marker, color="black",
                     markersize=2.6, elinewidth=0.8, capsize=1.3,
                     linestyle="none", zorder=6)
    _zero_band(dax)

    ax.set_xticks(x)
    ax.set_xticklabels(["%g" % v for v in w3_values])
    for a in axes:
        a.margins(x=0.04)
    _panel_letters(axes)

    # The four hatches go above the figure, the three colours into the space the
    # bars leave under the aggregate outlier at w_3 = 0.5.
    # Both legends live in panel (a), whose upper right is empty once the
    # aggregate outlier at w_3 = 0.5 is behind them. The four hatches go on top,
    # because they are what the reader needs first, and the cost contributions
    # sit below them rather than beside them - side by side the two would run
    # into the w_3 = 0.5 bars. The swatches stay large, since the hatch pattern
    # was unreadable at matplotlib's default handle size.
    # Headroom for the two stacked legends. Five per cent is not enough - the
    # lower legend then runs into the w_3 >= 2 bars, which reach a bit under
    # two thirds of the panel.
    ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1] * 1.18)
    leg_modes = ax.legend(handles=[Patch(facecolor="white", edgecolor="black",
                                         linewidth=0.3, hatch=h, label=l)
                                   for _, _, l, h, _, _, _ in PAPER_MODES],
                          ncol=2, fontsize=6.2, loc="upper right", frameon=False,
                          handlelength=2.2, handleheight=1.6, columnspacing=1.0,
                          labelspacing=0.35, handletextpad=0.5, borderaxespad=0.4)
    ax.add_artist(leg_modes)
    ax.legend(handles=[Patch(facecolor=c, edgecolor="black", linewidth=0.3, label=l)
                       for _, l, c in COMPONENTS],
              ncol=1, fontsize=6.2, loc="upper right",
              bbox_to_anchor=(1.0, 0.74), frameon=False, handlelength=1.8,
              handleheight=1.3, labelspacing=0.35, handletextpad=0.5,
              borderaxespad=0.4)
    fig.tight_layout(pad=0.3, h_pad=0.4)
    save(fig, export_path, "ram_w3_paper")


def paper_cost_figure_markers(man, cell, export_path):
    """Compact paper figure, marker style: the same data without bars.

    A bar has to start at zero, which puts the few per cent that separate the
    modes into the top fifth of the panel. A marker does not, so this variant
    cuts the axis to the band the modes occupy and names the two aggregate cells
    at w_3 = 0.5 that fall outside it. Panel (b) additionally plots the waiting
    part of each difference as an open marker: it lands on the total, which is
    the claim that the difference is waiting time and nothing else.
    """
    w3_values = man["w3_values"]
    absolute, difference = _paper_cells(man, cell)
    fig, axes = _paper_axes()
    ax, dax = axes
    x = np.arange(len(w3_values), dtype=float)

    dodge = 0.12
    series = []
    for i, mode in enumerate(PAPER_MODES):
        ram_key, pol_key, label, _, color, marker, fill = mode
        _, totals, errs = absolute[(ram_key, pol_key)]
        xs = x + (i - (len(PAPER_MODES) - 1) / 2.0) * dodge
        series.append((label, color, marker, fill, xs, totals, errs))

    # Cut the axis to the band the modes occupy. The unpaired intervals are kept
    # but drawn faintly: they overlap by construction - the Monte-Carlo spread of
    # one configuration is larger than the effect - and panel (b) is where the
    # comparison is actually resolved.
    finite = sorted(t for _, _, _, _, _, ts, _ in series for t in ts if t == t)
    cut = 1.25 * finite[len(finite) // 2]
    inlier = [(t, e) for _, _, _, _, _, ts, es in series
              for t, e in zip(ts, es) if t == t and t <= cut]
    top = max(t + e for t, e in inlier)
    bottom = min(t - e for t, e in inlier)
    pad = 0.10 * (top - bottom)

    off_scale = []
    for label, color, marker, fill, xs, totals, errs in series:
        vis = [t if (t == t and t <= cut) else float("nan") for t in totals]
        ax.errorbar(xs, vis, yerr=errs, marker=marker, color=color,
                    markersize=3.2, markerfacecolor=(color if fill == "full" else "white"),
                    markeredgewidth=0.9, linewidth=0.8, elinewidth=0.6,
                    capsize=1.2, linestyle="-", zorder=4, alpha=0.9)
        for xp, t in zip(xs, totals):
            if t == t and t > cut:
                ax.plot([xp], [1.0], marker="^", color=color, markersize=3.0,
                        transform=ax.get_xaxis_transform(), clip_on=False, zorder=6)
                off_scale.append("%s %.0fk" % (label, t / 1000.0))
    ax.set_ylim(bottom - pad, top + pad)
    if off_scale:
        ax.text(0.015, 0.80, "off scale:\n" + "\n".join(off_scale),
                transform=ax.transAxes, fontsize=5.2, color="0.25",
                va="top", ha="left", linespacing=1.4)

    for ram_key, label, color, marker, _ in PAPER_DIFF:
        comps, totals, errs, _ = difference[ram_key]
        dax.errorbar(x, totals, yerr=errs, marker=marker, color=color,
                     markersize=3.2, linewidth=0.9, elinewidth=0.7, capsize=1.2,
                     linestyle="-", zorder=4)
        dax.plot(x, comps["complete_time_cost"], marker=marker, color=color,
                 markersize=2.4, markerfacecolor="white", markeredgewidth=0.7,
                 linestyle="none", zorder=5)
    _zero_band(dax)

    ax.set_xticks(x)
    ax.set_xticklabels(["%g" % v for v in w3_values])
    for a in axes:
        a.margins(x=0.06)
    _panel_letters(axes)

    handles = [Line2D([0], [0], color=c, marker=mk, linestyle="-", linewidth=0.8,
                      markersize=3.2, markerfacecolor=(c if f == "full" else "white"),
                      markeredgewidth=0.9, label=l)
               for _, _, l, _, c, mk, f in PAPER_MODES]
    handles += [Line2D([0], [0], color="0.35", marker="o", linestyle="none",
                       markersize=2.4, markerfacecolor="white",
                       markeredgewidth=0.7, label="waiting part of the difference")]
    fig.legend(handles=handles, ncol=2, fontsize=5.6, loc="upper center",
               bbox_to_anchor=(0.5, 1.0), frameon=False, handlelength=1.8,
               columnspacing=1.0, labelspacing=0.3, handletextpad=0.5)
    fig.tight_layout(pad=0.3, h_pad=0.4, rect=(0, 0, 1, 0.86))
    save(fig, export_path, "ram_w3_paper_markers")




def readout(man, cell):
    """Numeric table behind the figures."""
    print("\noverall cost (mean over request streams) and paired difference:")
    print("%-5s %-5s %10s %10s %10s %8s %9s %6s"
          % ("ram", "w_3", "det", "stoch", "diff", "diff%", "95% CI", "wins"))
    for ram in man["ram_settings"]:
        for w3 in man["w3_values"]:
            det_run = cell.get((ram["key"], w3, "det"))
            stoch_run = cell.get((ram["key"], w3, "stoch"))
            det_rows, _ = load_run(det_run) if det_run else ([], False)
            stoch_rows, _ = load_run(stoch_run) if stoch_run else ([], False)
            md = stats(det_rows, "overall_cost")[0] if det_rows else float("nan")
            ms = stats(stoch_rows, "overall_cost")[0] if stoch_rows else float("nan")
            pairs = paired_rows(det_run, stoch_run) if det_run and stoch_run else []
            m, h, n, wins = paired_diff(pairs, "overall_cost") if pairs \
                else (float("nan"), 0.0, 0, 0)
            print("%-5s %-5g %10.1f %10.1f %10.1f %7.2f%% %9s %4d/%d"
                  % (ram["key"], w3, md, ms, m, 100 * m / md if md else float("nan"),
                     f"+-{h:.0f}", wins, n))

    have_latency = any(load_latency(cell[(ram["key"], w3, "det")])
                       for ram in man["ram_settings"] for w3 in man["w3_values"]
                       if (ram["key"], w3, "det") in cell)
    if not have_latency:
        return
    print("\nresponse time [s] over all pooled request streams:")
    print("%-5s %-5s %-6s %8s %8s %8s %8s %8s"
          % ("ram", "w_3", "policy", "mean", "p50", "p90", "p95", "p99"))
    for ram in man["ram_settings"]:
        for w3 in man["w3_values"]:
            for pol in man["policies"]:
                num_run = cell.get((ram["key"], w3, pol["key"]))
                pooled = []
                for entry in load_latency(num_run) if num_run else []:
                    pooled.extend(entry.get("response_times", []))
                if not pooled:
                    continue
                print("%-5s %-5g %-6s %8.1f %8.0f %8.0f %8.0f %8.0f"
                      % (ram["key"], w3, pol["key"], sum(pooled) / len(pooled),
                         *[quantile(pooled, p) for p in QUANTILES]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--export", default=None,
                        help="output folder (default results/run_<base_run>)")
    parser.add_argument("--cell", default=None,
                        help="representative cell for the latency figure, "
                             "e.g. l2,1.5 (default: last RAM model, middle w_3)")
    parser.add_argument("--paper", action="store_true",
                        help="draw the compact one-column version of the cost "
                             "figure used in the paper instead of the two "
                             "diagnostic figures")
    parser.add_argument("--paper-style", choices=("bars", "markers"), default="bars",
                        help="bars stacks the three cost contributions and "
                             "separates the four modes by hatch; markers drops "
                             "the stack for a cut axis on which the few per "
                             "cent between the modes are resolvable")
    parser.add_argument("--paper-export", default=os.path.join("UCC_MINLP", "Figures"),
                        help="output folder of the paper figure")
    args = parser.parse_args()

    with open(args.manifest, "r") as f:
        man = json.load(f)

    export_path = args.export or os.path.join("results", f"run_{man['base_run']}")
    os.makedirs(export_path, exist_ok=True)

    # index the manifest by (ram, w_3, policy)
    cell = {}
    for num_run, info in man["runs"].items():
        cell[(info["ram"], info["w_3"], info["policy"])] = int(num_run)

    if args.cell:
        ram_key, w3_ref = args.cell.split(",")
        ram_key, w3_ref = ram_key.strip(), float(w3_ref)
    else:
        ram_key = man["ram_settings"][-1]["key"]
        w3_ref = man["w3_values"][len(man["w3_values"]) // 2 - 1]

    if args.paper:
        os.makedirs(args.paper_export, exist_ok=True)
        if args.paper_style == "markers":
            paper_cost_figure_markers(man, cell, args.paper_export)
        else:
            paper_cost_figure_bars(man, cell, args.paper_export)
    else:
        cost_figure(man, cell, export_path)
        latency_figure(man, cell, ram_key, w3_ref, export_path)
    readout(man, cell)
