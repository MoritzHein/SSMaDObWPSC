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
#     (monetary M)           (basic time B)        (waiting time W)
#
# so a bar's height is the overall cost and all four requested quantities are
# visible at once. The stack is only ALMOST the reported overall_cost: the
# simulation computes
#
#     overall_cost = w_1 * money_cost + w_2 * sum(realized response times)
#
# whereas time_cost is the ANALYTIC basic time (traced through the hierarchy and
# weighted by the sampled request counts). The two differ by the analytic-vs-
# realized gap plus per-second rounding - about 2% in practice. The measured
# overall_cost is therefore drawn as a separate marker on each bar, so that
# residual is visible instead of being silently absorbed into the stack.
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

COMPONENTS = [
    ("money_cost", "$w_1 M$ (monetary)", "C0"),
    ("time_cost", "$w_2 B$ (basic time)", "C1"),
    ("complete_time_cost", "$w_2 W$ (waiting)", "C2"),
]
POLICY_HATCH = {"det": "", "stoch": "///"}
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
    return mean, 1.96 * sd / math.sqrt(len(v))


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
    return mean, 1.96 * sd / math.sqrt(len(v)), len(v), wins


def stacked_signed(ax, xs, series, width, colors):
    """Stack signed contributions: positives grow up, negatives grow down."""
    up = np.zeros(len(xs))
    down = np.zeros(len(xs))
    for vals, color in zip(series, colors):
        vals = np.asarray(vals, dtype=float)
        pos = np.where(vals > 0, vals, 0.0)
        neg = np.where(vals < 0, vals, 0.0)
        ax.bar(xs, pos, width, bottom=up, color=color, edgecolor="black",
               linewidth=0.4)
        ax.bar(xs, neg, width, bottom=down, color=color, edgecolor="black",
               linewidth=0.4)
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
                for k, _, _ in COMPONENTS:
                    m, _ = stats(rows, k)
                    heights[k].append(weight[k] * m)
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
            # measured overall cost (the stack is only approximately this, see header)
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
            for k, _, _ in COMPONENTS:
                m, _, _, _ = paired_diff(pairs, k)
                comp_diffs[k].append(weight[k] * m)
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


def paper_cost_figure(man, cell, export_path):
    """The cost figure of the paper: same data as cost_figure, less chrome.

    Two columns, one per capacity model, and two rows: absolute stacked cost on
    top, paired difference below. The four-row variant of cost_figure is 8.6 in
    tall and cannot go on a page; this one is 4.4 in and spans both columns of a
    two-column layout. Titles, the suptitle and the figure-level legend are
    dropped, because a paper figure carries that text in its caption. The legend
    moves inside panel (a), into the space that the tall w_3 = 0.5 bar of the
    aggregate model leaves empty.
    """
    w1, w2 = man["w_1"], man["w_2"]
    w3_values = man["w3_values"]
    ram_settings = man["ram_settings"]
    policies = man["policies"]
    weight = {"money_cost": w1, "time_cost": w2, "complete_time_cost": w2}

    fig, axes = plt.subplots(2, len(ram_settings), figsize=(7.15, 4.35),
                             gridspec_kw={"height_ratios": [2.1, 1.0]},
                             sharex="col")
    axes = np.atleast_2d(axes)
    x = np.arange(len(w3_values), dtype=float)
    width = 0.38
    offsets = {p["key"]: (i - (len(policies) - 1) / 2) * width
               for i, p in enumerate(policies)}
    diff_labels = []

    for ci, ram in enumerate(ram_settings):
        ax, dax = axes[0][ci], axes[1][ci]

        for pol in policies:
            xs = [x[wi] + offsets[pol["key"]] for wi in range(len(w3_values))]
            heights = {k: [] for k, _, _ in COMPONENTS}
            totals, errs = [], []
            for w3 in w3_values:
                num_run = cell.get((ram["key"], w3, pol["key"]))
                rows, _ = load_run(num_run) if num_run else ([], False)
                for k, _, _ in COMPONENTS:
                    heights[k].append(weight[k] * stats(rows, k)[0] if rows else 0.0)
                m, h = stats(rows, "overall_cost") if rows else (float("nan"), 0.0)
                totals.append(m)
                errs.append(h)
            bottom = np.zeros(len(w3_values))
            for k, label, color in COMPONENTS:
                vals = np.array(heights[k])
                ax.bar(xs, vals, width * 0.92, bottom=bottom, color=color,
                       edgecolor="black", linewidth=0.4,
                       hatch=POLICY_HATCH[pol["key"]])
                bottom = bottom + vals
            ax.errorbar(xs, totals, yerr=errs, fmt="_", color="black",
                        markersize=8, markeredgewidth=1.2, elinewidth=1.0,
                        capsize=2, linestyle="none", zorder=5)

        comp_diffs = {k: [] for k, _, _ in COMPONENTS}
        totals, errs, labels = [], [], []
        for w3 in w3_values:
            det_run = cell.get((ram["key"], w3, "det"))
            stoch_run = cell.get((ram["key"], w3, "stoch"))
            pairs = paired_rows(det_run, stoch_run) if det_run and stoch_run else []
            if not pairs:
                for k, _, _ in COMPONENTS:
                    comp_diffs[k].append(0.0)
                totals.append(float("nan")); errs.append(0.0); labels.append("")
                continue
            for k, _, _ in COMPONENTS:
                comp_diffs[k].append(weight[k] * paired_diff(pairs, k)[0])
            m, h, n, wins = paired_diff(pairs, "overall_cost")
            base = sum(d["overall_cost"] for d, _ in pairs) / len(pairs)
            totals.append(m); errs.append(h)
            labels.append(f"{100 * m / base:+.1f}%" + "\n" + f"{wins}/{n}")

        stacked_signed(dax, x, [comp_diffs[k] for k, _, _ in COMPONENTS],
                       width * 1.5, [c for _, _, c in COMPONENTS])
        dax.errorbar(x, totals, yerr=errs, fmt="D", color="black", markersize=3.5,
                     elinewidth=1.2, capsize=2.5, linestyle="none", zorder=6)
        dax.axhline(0.0, color="black", linewidth=0.8)
        dax.set_xticks(x)
        dax.set_xticklabels([f"{v:g}" for v in w3_values])
        dax.set_xlabel("utilization weight $w_3$")
        diff_labels.append(labels)
        for a in (ax, dax):
            a.margins(x=0.05)
            a.tick_params(labelsize=8)

    # The absolute panels share one y-range, so that the collapse of the
    # aggregate model at w_3 = 0.5 shows as a difference in bar height. The
    # difference panels keep their own: one cell of the aggregate model is an
    # order of magnitude larger than every other difference and would flatten
    # the remaining eleven into invisibility. The caption says so.
    lo = min(a.get_ylim()[0] for a in axes[0])
    hi = max(a.get_ylim()[1] for a in axes[0])
    for a in axes[0]:
        a.set_ylim(lo, hi)
    for a in axes[0][1:]:
        a.set_yticklabels([])
    axes[0][0].set_ylabel("overall cost $C$")
    for a in axes[1]:
        a.set_ylabel("$\\Delta C$")

    # Difference annotations sit in the empty half of the panel, above the zero
    # line for a negative difference and above the marker for a positive one.
    for ci, labels in enumerate(diff_labels):
        dax = axes[1][ci]
        lo, hi = dax.get_ylim()
        dax.axhspan(lo, 0.0, color="C2", alpha=0.06, zorder=0)
        # Make room above the zero line for the annotations. Where every
        # difference is negative the zero line is the top of the panel, and the
        # labels would otherwise be written outside it.
        dax.set_ylim(lo, hi + 0.30 * (hi - lo))
        for xi, lab in enumerate(labels):
            if not lab:
                continue
            dax.annotate(lab, (x[xi], 0.0), textcoords="offset points",
                         xytext=(0, 3), ha="center", va="bottom", fontsize=5.6)

    # Panel letters, and the legend inside panel (a).
    for i, a in enumerate([axes[0][0], axes[0][1]]):
        a.text(0.012, 0.965, "(%s)" % "ab"[i], transform=a.transAxes,
               fontsize=9, va="top", ha="left")
    # The difference panels carry their annotations along the top, so their
    # letters go to the bottom left, below the leftmost bar.
    for i, a in enumerate([axes[1][0], axes[1][1]]):
        a.text(0.012, 0.035, "(%s)" % "cd"[i], transform=a.transAxes,
               fontsize=9, va="bottom", ha="left")
    handles = [Patch(facecolor=c, edgecolor="black", label=l) for _, l, c in COMPONENTS]
    handles += [Patch(facecolor="white", edgecolor="black",
                      hatch=POLICY_HATCH[p["key"]], label=p["label"]) for p in policies]
    handles += [Line2D([0], [0], color="black", marker="_", linestyle="none",
                       markersize=8, markeredgewidth=1.2,
                       label="measured $C$ (95\\% CI)")]
    axes[0][0].legend(handles=handles, ncol=2, fontsize=6.6, loc="upper right",
                      frameon=False, handlelength=1.6, columnspacing=1.0,
                      labelspacing=0.35, borderaxespad=0.6)

    fig.tight_layout(pad=0.4, w_pad=0.8, h_pad=0.6)
    save(fig, export_path, "ram_w3_paper")


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
                        help="draw the compact two-by-two version of the cost "
                             "figure used in the paper instead of the two "
                             "diagnostic figures")
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
        paper_cost_figure(man, cell, args.paper_export)
    else:
        cost_figure(man, cell, export_path)
        latency_figure(man, cell, ram_key, w3_ref, export_path)
    readout(man, cell)
