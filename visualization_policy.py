# Visualization of the blocking-aware selection study (settings/policy_campaign.json).
#
# Usage: visualization_policy.py [--w3=1.5] [--paper] [--latex]
#                                [--manifest=settings/policy_campaign.json]
#                                [--export=<dir>]
#
# Produces
#
#   policy_bars.pdf   two panels, one per memory model. Horizontal stacked bars
#                     give the realized cost of every selection policy at one
#                     w_3, split into w_1 Z, w_2 B and w_2 W. Vertical lines mark
#                     the model-free placements, which do not depend on w_3 or on
#                     the memory model and are therefore the same in both panels.
#   policy_w3.pdf     the same comparison against w_3, reduced to the series that
#                     carry the argument, plus the paired difference to the
#                     deterministic open-loop policy.
#
# and prints the tables the paper needs.
#
# Two conventions matter.
#
#   * Every difference is PAIRED on the request stream and its interval uses the
#     t quantile of n-1 degrees of freedom (2.262 at ten streams), not 1.96. The
#     effect here (2-6 %) is smaller than the Monte-Carlo spread of a single
#     configuration, so unpaired intervals would hide it.
#   * Bars are horizontal so the heuristics can be vertical rules across all
#     policies at once. Their cost is read on the same axis as the bars.
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

try:
    import scienceplots  # noqa: F401
    plt.style.use(["science", "no-latex"])
except Exception:
    pass

DEFAULT_MANIFEST = os.path.join("settings", "policy_campaign.json")

# Paul Tol muted. Checked for colour-vision deficiency: the smallest separation
# over all pairs and over protan/deutan/tritan is dE 20.4 (OKLab x100), against
# a target of 8. The C0/C1/C2 triple used elsewhere in the repo fails that check
# badly - orange against green is dE 4.2 under protanopia.
COMPONENTS = [
    ("money_cost", r"$w_1 Z$ (monetary)", "#332288"),
    ("time_cost", r"$w_2 B$ (basic time)", "#DDCC77"),
    ("complete_time_cost", r"$w_2 W$ (waiting)", "#CC6677"),
]

# The reported `time_cost` is NOT the basic time the realized cost is made of.
# It is the ANALYTIC one - selection_simulation.py accumulates the critical path
# of the workflow times the number of requests of the (workflow, leaf) pair -
# whereas overall_cost is w_1 Z + w_2 * sum(realized response times). Money and
# waiting enter overall_cost exactly as reported, so the whole discrepancy sits
# in the basic time and the realized one is the remainder,
# w_2 B = overall_cost - w_1 Z - w_2 W. Exact, because the simulator sets
# wait = max(0, response - base_T) and no request of the campaign finishes
# faster than its own base time. Taking the remainder makes every stack end at
# the measured cost, which is also the quantity the markers and the model-free
# reference rules are in, so a rule can never pass between a bar and its marker.

# Series of the diagnostic w_3 figure, in fixed order. Kept to six so no hue is
# reused.
W3_SERIES = [
    ("det_ol", "det., open loop", "#BBBBBB", "o", "--"),
    ("stoch_ol", "stoch., open loop", "#888888", "s", "--"),
    ("det_ll", "det. + least loaded", "#CC6677", "v", "-"),
    ("free_sup", "stoch. + free sample, supp.", "#DDCC77", "^", "-"),
    ("free_wide", "stoch. + free sample, wide", "#4477AA", "D", "-"),
    ("stoch_ll", "stoch. + least loaded", "#332288", "P", "-"),
]

# The paper compares three rules, on both placements where that is meaningful:
# no recourse at all, recourse confined to the support of H, and recourse over
# every placed deployment. Dashed is open loop, solid is a state-aware rule.
# The remaining six variants stay in the diagnostic figures and the tables.
PAPER_SERIES = [
    ("det_ol", "det., open loop", "#888888", "o", "--"),
    ("stoch_ol", "stoch., open loop", "#4477AA", "s", "--"),
    ("free_sup", "stoch. + free sample, supp.", "#117733", "^", "-"),
    ("det_ll", "det. + least loaded", "#CC6677", "v", "-"),
    ("stoch_ll", "stoch. + least loaded", "#000000", "D", "-"),
]

# Shorter names for the horizontal bar figure, whose y labels eat into a width
# of one column. The placement comes first, the runtime rule second.
PAPER_SHORT = {
    "det_ol": "det., open loop",
    "stoch_ol": "stoch., open loop",
    "free_sup": "stoch., free sample",
    "det_ll": "det., least loaded",
    "stoch_ll": "stoch., least loaded",
}

# The two model-free placements the paper keeps: the cheapest one, which is the
# baseline worth beating, and an edge-first rule, which is the cautionary one.
# The labels override the manifest's, which qualifies the first as "spread" to
# separate it from the all_cloud straw man that this figure does not show.
PAPER_HEURISTICS = (
    ("all_cloud_rr", "all on cloud"),
    ("local_first", "edge first")
)

# The memory model the paper's policy figure uses. The aggregate model is the
# subject of the preceding figure and adding it here would need a second column.
PAPER_RAM = "l2"

# IEEE column width, from config_mpl.py.
COLUMNWIDTH = 3.44488

T_QUANTILE = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
              7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
              14: 2.145, 19: 2.093, 29: 2.045}


def t_quantile(df):
    if df in T_QUANTILE:
        return T_QUANTILE[df]
    return T_QUANTILE[min(T_QUANTILE, key=lambda k: abs(k - df))]


def load_run(num_run):
    """Per-stream cost components of one run; [] if it was never simulated."""
    base = os.path.join("results", "run_%d" % num_run)
    for name in ("results_seeds.json", "results.json"):
        path = os.path.join(base, name)
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            return data if isinstance(data, list) else [data]
    return []


def mean_ci(rows, key):
    """Unpaired mean and t interval of one run; used for absolute levels only."""
    vals = [r[key] for r in rows if r.get(key) is not None]
    n = len(vals)
    if n == 0:
        return float("nan"), 0.0, 0
    mean = sum(vals) / n
    if n < 2:
        return mean, 0.0, n
    var = sum((v - mean) ** 2 for v in vals) / (n - 1)
    return mean, t_quantile(n - 1) * math.sqrt(var / n), n


def paired(rows_a, rows_b, key="overall_cost"):
    """Mean of (a - b) over the shared request streams, with a t interval."""
    if not rows_a or not rows_b:
        return None
    by_seed = {r.get("simulation_seed"): r for r in rows_b}
    diffs = [r[key] - by_seed[r["simulation_seed"]][key]
             for r in rows_a
             if r.get("simulation_seed") in by_seed
             and r.get(key) is not None
             and by_seed[r["simulation_seed"]].get(key) is not None]
    n = len(diffs)
    if n < 2:
        return None
    mean = sum(diffs) / n
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    return mean, t_quantile(n - 1) * math.sqrt(var / n), n, sum(1 for d in diffs if d < 0)


def weighted_split(study, rows):
    """The three weighted contributions, summing exactly to the measured cost."""
    z = study.weight["money_cost"] * mean_ci(rows, "money_cost")[0]
    wait = study.weight["complete_time_cost"] * mean_ci(rows, "complete_time_cost")[0]
    return {"money_cost": z,
            "time_cost": mean_ci(rows, "overall_cost")[0] - z - wait,
            "complete_time_cost": wait}


class Study(object):
    """Everything the figures and tables read, indexed the way they ask for it."""

    def __init__(self, manifest_path):
        self.man = json.load(open(manifest_path))
        self.w3_values = self.man["w3_values"]
        self.ram_models = self.man["ram_models"]
        self.variants = self.man["variants"]
        self.cell = {}
        for num_run, info in self.man["runs"].items():
            if info.get("kind") == "heuristic":     # a reference, not a cell
                continue
            self.cell[(info["ram"], info["w_3"], info["policy"])] = int(num_run)
        self.baseline = {}
        for keystr, num_run in self.man["baselines"].items():
            ram, w3, pol = keystr.split("|")
            self.baseline[(ram, float(w3), pol)] = num_run
        # Objective weights come from a source run rather than the manifest, so
        # the figure cannot drift from the parameter files the runs used.
        any_run = next(k for k, v in self.man["runs"].items()
                       if v.get("kind") != "heuristic")
        cfg = json.load(open(os.path.join("settings", "parameters_%s.json" % any_run)))
        self.weight = {"money_cost": cfg["w_1"], "time_cost": cfg["w_2"],
                       "complete_time_cost": cfg["w_2"]}

    def rows(self, ram, w3, policy):
        """Streams of one configuration; open-loop policies come from the sources."""
        if policy in ("det_ol", "stoch_ol"):
            run = self.baseline.get((ram, w3, policy[:-3]))
        else:
            run = self.cell.get((ram, w3, policy))
        return load_run(run) if run is not None else []

    def heuristics(self, keys=None):
        """Model-free placements: (label, mean cost, half-width, streams).

        `keys` restricts the result to the given comparison_mode names, in the
        manifest's order. The heuristic entries carry `mode` and `label` only,
        so the filter has to go through `mode`.
        """
        out = []
        for num_run, info in sorted(self.man.get("heuristic_runs", {}).items(),
                                    key=lambda kv: int(kv[0])):
            if keys is not None and info.get("mode") not in keys:
                continue
            rows = load_run(int(num_run))
            mean, half, n = mean_ci(rows, "overall_cost_comparison")
            if n:
                out.append((info["label"], mean, half, n))
        return out

    def heuristic_rows(self, key):
        """Per-stream cost of one model-free placement, renamed so that the
        ordinary helpers apply to it.

        The comparison split is present only for runs re-simulated after
        resimulate_seeds.py started recording it. Without it a caller still gets
        the totals and has to fall back to an unsplit bar.
        """
        for num_run, info in self.man.get("heuristic_runs", {}).items():
            if info.get("mode") != key:
                continue
            out = []
            for r in load_run(int(num_run)):
                if r.get("overall_cost_comparison") is None:
                    continue
                row = {"simulation_seed": r.get("simulation_seed"),
                       "overall_cost": r["overall_cost_comparison"]}
                for field in ("money_cost", "time_cost", "complete_time_cost"):
                    if r.get(field + "_comparison") is not None:
                        row[field] = r[field + "_comparison"]
                out.append(row)
            return out
        return []

    def heuristic_label(self, key):
        for info in self.man.get("heuristic_runs", {}).values():
            if info.get("mode") == key:
                return info.get("label", key)
        return key

    def order(self):
        """Bar order: the two open-loop policies, then the variants."""
        return ([("det_ol", "det., open loop"), ("stoch_ol", "stoch., open loop")]
                + [(v["key"], v["label"]) for v in self.variants])


# ---------------------------------------------------------------- figures
def bar_figure(study, w3, export):
    """Cost of every policy at one w_3, one column per memory model.

    Top row: the realized cost split into its three contributions, drawn from
    zero because a stacked bar on a cut axis states a composition that is not
    there. On that scale the policies differ by a few percent and the model-free
    placements are what set it, so the bottom row repeats the comparison as the
    PAIRED difference to the deterministic open-loop policy, which is the only
    form in which an effect of this size is resolvable: both policies see the
    same request streams by construction, so the paired interval is about four
    times narrower than the interval on either policy alone.
    """
    order = study.order()
    heur = study.heuristics()
    fig, axes = plt.subplots(2, len(study.ram_models),
                             figsize=(7.09, 0.30 * len(order) + 4.4),
                             sharey=True)
    axes = np.atleast_2d(axes)
    ypos = np.arange(len(order))[::-1]

    # A reference is drawn as a rule only if it fits next to the bars; the rest
    # are named with their value, so nothing is silently dropped.
    bar_max = max(mean_ci(study.rows(ram["key"], w3, k), "overall_cost")[0]
                  for ram in study.ram_models for k, _ in order
                  if study.rows(ram["key"], w3, k))
    in_range = [h for h in heur if h[1] <= 2.0 * bar_max]
    off_scale = [h for h in heur if h[1] > 2.0 * bar_max]
    xmax = max([bar_max] + [h[1] for h in in_range]) * 1.10
    shades = [str(0.20 + 0.40 * i / max(len(in_range) - 1, 1)) for i in range(len(in_range))]

    for col, ram in enumerate(study.ram_models):
        top, bot = axes[0][col], axes[1][col]
        base_rows = study.rows(ram["key"], w3, "det_ol")
        base_mean, _, _ = mean_ci(base_rows, "overall_cost")

        for y, (key, _) in zip(ypos, order):
            rows = study.rows(ram["key"], w3, key)
            if not rows:
                continue
            left = 0.0
            seg = weighted_split(study, rows)
            for comp, _, color in COMPONENTS:
                top.barh(y, seg[comp], left=left, height=0.62,
                         color=color, edgecolor="white", linewidth=0.8, zorder=3)
                left += seg[comp]

            d = paired(rows, base_rows)
            if d and base_mean == base_mean and key != "det_ol":
                bot.barh(y, 100 * d[0] / base_mean, height=0.62, color="#4477AA",
                         edgecolor="white", linewidth=0.8, zorder=3)
                bot.errorbar(100 * d[0] / base_mean, y, xerr=100 * d[1] / base_mean,
                             fmt="none", ecolor="0.2", elinewidth=0.9, capsize=2, zorder=5)

        # The rules are drawn without inline labels: three of them sit within
        # 2k of each other and rotated text there is unreadable. They are named
        # with their value in one block instead.
        for (label, mean, _, _), shade in zip(in_range, shades):
            top.axvline(mean, color=shade, linestyle=(0, (4, 2)), linewidth=1.1, zorder=4)

        bot.text(0.995, ypos[0], "reference ", fontsize=6, color="0.45",
                 ha="right", va="center", transform=bot.get_yaxis_transform())

        top.set_xlim(0, xmax)
        top.set_title(ram["label"], fontsize=9)
        top.set_xlabel("realized cost $C$")
        bot.set_xlabel("paired difference to det. open loop [\\%]"
                       if plt.rcParams.get("text.usetex") else
                       "paired difference to det. open loop [%]")
        bot.axvline(0, color="0.3", linewidth=0.8, zorder=4)
        for ax in (top, bot):
            ax.set_yticks(ypos)
            ax.set_yticklabels([lab for _, lab in order], fontsize=7)
            ax.grid(axis="x", color="0.92", linewidth=0.5, zorder=0)
            ax.set_axisbelow(True)

    handles = [Patch(facecolor=c, edgecolor="white", label=l) for _, l, c in COMPONENTS]
    handles.append(Line2D([0], [0], color="0.4", linestyle=(0, (4, 2)),
                          label="model-free placement"))
    leg = axes[0][0].legend(handles=handles, fontsize=6, loc="center right", framealpha=1.0)
    leg.set_zorder(10)   # the reference rules are drawn above the bars

    # Every model-free placement is named with its cost, whether or not it fits
    # on the axis, so none of them is silently dropped.
    lines = ["model-free placements"]
    lines += ["%s %.0fk" % (l, m / 1000) for l, m, _, _ in in_range]
    lines += ["%s %.0fk*" % (l, m / 1000) for l, m, _, _ in off_scale]
    if off_scale:
        lines.append("* beyond the axis")
    axes[0][-1].text(0.985, 0.5, "\n".join(lines), transform=axes[0][-1].transAxes,
                     fontsize=5.5, color="0.25", ha="right", va="center",
                     linespacing=1.5, zorder=10,
                     bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                               edgecolor="0.8", linewidth=0.5))
    fig.suptitle("selection policy at $w_3=%s$, ten paired request streams" % w3,
                 fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = os.path.join(export, "policy_bars.pdf")
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote %s" % path)


def w3_figure(study, export):
    """Cost against w_3 (top) and the paired gain over det. open loop (bottom)."""
    fig, axes = plt.subplots(2, len(study.ram_models), figsize=(7.09, 4.2),
                             sharex=True, sharey="row")
    axes = np.atleast_2d(axes)
    for col, ram in enumerate(study.ram_models):
        top, bot = axes[0][col], axes[1][col]
        for key, label, color, marker, ls in W3_SERIES:
            xs, ys, ds, es = [], [], [], []
            for w3 in study.w3_values:
                rows = study.rows(ram["key"], w3, key)
                base = study.rows(ram["key"], w3, "det_ol")
                if not rows:
                    continue
                mean, _, _ = mean_ci(rows, "overall_cost")
                xs.append(w3)
                ys.append(mean)
                d = paired(rows, base)
                bmean, _, _ = mean_ci(base, "overall_cost")
                ds.append(100 * d[0] / bmean if d and bmean == bmean else float("nan"))
                es.append(100 * d[1] / bmean if d and bmean == bmean else 0.0)
            if not xs:
                continue
            top.plot(xs, ys, marker=marker, ls=ls, color=color, label=label,
                     markersize=3.5, linewidth=1.2)
            bot.errorbar(xs, ds, yerr=es, marker=marker, ls=ls, color=color,
                         markersize=3.5, linewidth=1.2, capsize=2)
        top.set_title(ram["label"], fontsize=9)
        top.grid(color="0.92", linewidth=0.5)
        bot.grid(color="0.92", linewidth=0.5)
        bot.axhline(0, color="0.3", linewidth=0.8)
        bot.set_xlabel("$w_3$")
        top.set_axisbelow(True)
        bot.set_axisbelow(True)
    axes[0][0].set_ylabel("realized cost $C$")
    axes[1][0].set_ylabel(r"$\Delta$ vs det. open loop [\%]"
                          if plt.rcParams.get("text.usetex") else
                          "$\\Delta$ vs det. open loop [%]")
    axes[0][0].legend(fontsize=6.5, loc="best", framealpha=0.9)
    fig.tight_layout()
    path = os.path.join(export, "policy_w3.pdf")
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote %s" % path)


def paper_bar_figure(study, w3, export, heuristic_style="row"):
    """One panel, one w_3, the five paper policies under both memory models.

    The two absolute panels of bar_figure merged onto a single axis. The memory
    model becomes a block on the y axis instead of a column, so the ten bars are
    read against one common x scale and the aggregate and the corrected model
    can be compared row by row. The model-free references that fit on the axis
    sit as their own bars BETWEEN the two blocks, which puts them next to both
    without a rule crossing every bar, and lets them carry their cost split and
    their interval like any other row.

    Bars start at zero, because a stacked bar on a cut axis states a
    composition that is not there, and at that scale the policies differ by a
    few per cent. Each policy bar therefore carries its paired difference to the
    open loop of its own block, which is the only form in which an effect of
    this size is resolvable. The references have no open loop of their own and
    carry their measured cost instead.
    """
    # Pass one: the policy rows, which set the scale.
    blocks = []
    bar_max = 0.0
    for ram in study.ram_models:
        entries = []
        for key, _, _, _, _ in PAPER_SERIES:
            rows = study.rows(ram["key"], w3, key)
            if not rows:
                continue
            base_rows = study.rows(ram["key"], w3, "det_ol")
            base_mean, _, _ = mean_ci(base_rows, "overall_cost")
            mean, half, _ = mean_ci(rows, "overall_cost")
            if key == "det_ol":
                note = "reference"
            else:
                d = paired(rows, base_rows)
                note = (("%+.1f\\%%" if plt.rcParams.get("text.usetex") else "%+.1f%%")
                        % (100 * d[0] / base_mean)) if d and base_mean == base_mean else ""
            entries.append((PAPER_SHORT.get(key, key), weighted_split(study, rows),
                            mean, half, note))
            bar_max = max(bar_max, mean)
        blocks.append((ram["label"], entries))

    # Pass two: the references. `heuristic_style` decides how they appear.
    # "row" gives each one a bar of its own between the two blocks, which shows
    # its cost split and its interval and keeps it adjacent to both blocks.
    # "rule" draws it as a vertical line across all bars, which is denser and
    # lets every bar be compared to it at a glance, at the cost of the split.
    # Either way a reference too large for the axis is left out, because putting
    # it on the axis would compress every other bar. Its cost belongs in the
    # text.
    middle, rules = [], []
    for key, label in PAPER_HEURISTICS:
        rows = study.heuristic_rows(key)
        if not rows:
            continue
        mean, half, _ = mean_ci(rows, "overall_cost")
        if mean > 1.10 * bar_max:
            continue
        if heuristic_style == "rule":
            rules.append((label, mean))
            continue
        split = (weighted_split(study, rows)
                 if all("money_cost" in r and "complete_time_cost" in r for r in rows)
                 else None)
        # No annotation: a reference has no open loop of its own, so it would
        # have to carry a raw cost where every other row carries a percentage.
        # Its level is read off the axis like any other bar.
        middle.append((label, split, mean, half, ""))

    order = ([("group", blocks[0][0])] + [("row", e) for e in blocks[0][1]]
             + ([("group", "model-free")] if middle else [])
             + [("row", e) for e in middle]
             + [("group", blocks[1][0])] + [("row", e) for e in blocks[1][1]])
    rows_only = [e for kind, e in order if kind == "row"]

    fig, ax = plt.subplots(figsize=(COLUMNWIDTH, 0.20 * len(rows_only) + 0.95))
    ypos = np.arange(len(rows_only))[::-1]

    for y, (label, split, mean, half, note) in zip(ypos, rows_only):
        left = 0.0
        if split is None:
            # No per-stream split recorded for this run: one plain bar, so the
            # missing decomposition is visible rather than invented.
            ax.barh(y, mean, left=0.0, height=0.68, color="0.75",
                    edgecolor="white", linewidth=0.6, zorder=3)
        else:
            for comp, _, color in COMPONENTS:
                ax.barh(y, split[comp], left=left, height=0.68, color=color,
                        edgecolor="white", linewidth=0.6, zorder=3)
                left += split[comp]
        ax.errorbar(mean, y, xerr=half, fmt="|", color="0.15", markersize=4,
                    markeredgewidth=0.9, elinewidth=0.7, capsize=1.2, zorder=5)
        if note:
            # A reference rule passes straight through these labels, and a digit
            # half covered by a dashed line is misread rather than noticed, so
            # back them with white whenever a rule is drawn.
            ax.annotate(note, xy=(mean + half, y), xytext=(3, 0),
                        textcoords="offset points", fontsize=5.2, color="0.25",
                        va="center", ha="left", zorder=7,
                        bbox=(dict(boxstyle="square,pad=0.12", facecolor="white",
                                   edgecolor="none") if rules else None))

    for _, mean in rules:
        ax.axvline(mean, color="0.35", linestyle=(0, (4, 2)), linewidth=1.0,
                   zorder=4)

    ax.set_xlim(0, 1.20 * bar_max)
    ax.set_ylim(-0.7, len(rows_only) - 0.3)

    # Group separators and the right-hand axis naming the blocks.
    groups, cur, ri = [], None, 0
    for kind, payload in order:
        if kind == "group":
            cur = (payload, [])
            groups.append(cur)
        else:
            cur[1].append(ri)
            ri += 1
    centres, labels = [], []
    for gi, (label, idx) in enumerate(groups):
        if not idx:
            continue
        if gi:
            ax.axhline(ypos[idx[0]] + 0.5, color="0.25", linewidth=0.9, zorder=6)
        centres.append(sum(ypos[i] for i in idx) / float(len(idx)))
        labels.append(label)

    ax.set_yticks(ypos)
    ax.set_yticklabels([lab for lab, _, _, _, _ in rows_only], fontsize=6)
    ax.set_xlabel("realized cost $C$")
    ax.xaxis.label.set_size(8)
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(axis="x", color="0.92", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)

    side = ax.twinx()
    side.set_ylim(ax.get_ylim())
    side.set_yticks(centres)
    side.set_yticklabels(labels, fontsize=6, rotation=90, va="center")
    side.tick_params(axis="y", length=0)

    # A reference too large for the axis is left out of the figure entirely and
    # reported in the text instead. A rule has no y label to name it, so it is
    # named in the legend; a reference row is named by its y tick already.
    handles = [Patch(facecolor=c, edgecolor="white", label=l)
               for _, l, c in COMPONENTS]
    handles += [Line2D([0], [0], color="0.35", linestyle=(0, (4, 2)),
                       linewidth=1.0, label=label) for label, _ in rules]
    fig.legend(handles=handles, ncol=3 if not rules else 2, fontsize=5.6,
               loc="upper center",
               bbox_to_anchor=(0.5, 1.0), frameon=False, handlelength=1.5,
               columnspacing=1.0, labelspacing=0.3, handletextpad=0.5)
    fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.85))
    path = os.path.join(export, "policy_bars_paper.pdf")
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote %s" % path)




def paper_figure(study, export):
    """The paper's policy figure: one column, one memory model, five policies.

    Panel (a) is the realized cost against w_3 with the model-free placements as
    horizontal rules, panel (b) the paired difference to the deterministic open
    loop. The absolute panel exists to place the policies against the reference
    placements; the ordering between the policies is only resolvable in the
    paired panel, whose interval is about four times narrower.
    """
    ram = PAPER_RAM
    fig, axes = plt.subplots(2, 1, figsize=(COLUMNWIDTH, 3.4),
                             gridspec_kw={"height_ratios": [1.5, 1.0]},
                             sharex=True)
    top, bot = axes
    lows, highs = [], []

    for key, label, color, marker, ls in PAPER_SERIES:
        xs, ys, ds, es = [], [], [], []
        for w3 in study.w3_values:
            rows = study.rows(ram, w3, key)
            base = study.rows(ram, w3, "det_ol")
            if not rows:
                continue
            mean, _, _ = mean_ci(rows, "overall_cost")
            bmean, _, _ = mean_ci(base, "overall_cost")
            d = paired(rows, base)
            xs.append(w3)
            ys.append(mean)
            ds.append(100 * d[0] / bmean if d and bmean == bmean else float("nan"))
            es.append(100 * d[1] / bmean if d and bmean == bmean else 0.0)
        if not xs:
            continue
        lows.append(min(ys))
        highs.append(max(ys))
        top.plot(xs, ys, marker=marker, ls=ls, color=color, label=label,
                 markersize=3.0, linewidth=1.0)
        if key != "det_ol":
            bot.errorbar(xs, ds, yerr=es, marker=marker, ls=ls, color=color,
                         markersize=3.0, linewidth=1.0, elinewidth=0.7, capsize=1.5)

    # A reference is a rule only where it fits beside the policies; edge first
    # is 70 per cent above them and would flatten the panel, so it is named
    # instead of dropped.
    lo, hi = min(lows), max(highs)
    span = hi - lo
    named = []
    for key, label in PAPER_HEURISTICS:
        rows = study.heuristic_rows(key)
        if not rows:
            continue
        mean, _, _ = mean_ci(rows, "overall_cost")
        if lo - 1.5 * span <= mean <= hi + 1.5 * span:
            top.axhline(mean, color="0.45", linestyle=(0, (4, 2)), linewidth=0.9,
                        zorder=1)
            top.annotate("%s %.0f" % (label, mean), xy=(0.995, mean),
                         xycoords=top.get_yaxis_transform(),
                         xytext=(0, 2), textcoords="offset points", fontsize=5.4,
                         color="0.35", ha="right", va="bottom")
        else:
            named.append("%s %.0f" % (label, mean))
    if named:
        # Open a strip below the cheapest policy so the note has clear space.
        ylo, yhi = top.get_ylim()
        top.set_ylim(ylo - 0.13 * (yhi - ylo), yhi)
        top.text(0.015, 0.015, "off scale: " + ", ".join(named),
                 transform=top.transAxes, fontsize=5.4, color="0.35",
                 ha="left", va="bottom")

    bot.axhline(0, color="0.3", linewidth=0.8)
    bot.set_xlabel("utilization weight $w_3$")
    top.set_ylabel("realized cost $C$")
    bot.set_ylabel(r"$\Delta$ to det. open loop [\%]"
                   if plt.rcParams.get("text.usetex") else
                   "$\\Delta$ to det. open loop [%]")
    for ax in (top, bot):
        ax.grid(color="0.92", linewidth=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=7)
        ax.yaxis.label.set_size(8)
    bot.xaxis.label.set_size(8)
    top.text(0.015, 0.96, "(a)", transform=top.transAxes, fontsize=8,
             va="top", ha="left")
    # Keep a strip free under the cheapest policy for the panel letter.
    blo, bhi = bot.get_ylim()
    bot.set_ylim(blo - 0.16 * (bhi - blo), bhi)
    bot.text(0.015, 0.03, "(b)", transform=bot.transAxes, fontsize=8,
             va="bottom", ha="left")

    fig.legend(fontsize=5.6, ncol=2, loc="upper center",
               bbox_to_anchor=(0.5, 1.0), frameon=False, handlelength=2.0,
               columnspacing=1.0, labelspacing=0.3, handletextpad=0.5)
    fig.tight_layout(pad=0.3, h_pad=0.4, rect=(0, 0, 1, 0.88))
    path = os.path.join(export, "policy_paper.pdf")
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote %s" % path)


# ---------------------------------------------------------------- tables
def print_tables(study, latex):
    order = study.order()
    for ram in study.ram_models:
        print()
        print("=== memory model: %s ===" % ram["key"])
        head = "%-30s" % "policy" + "".join("%10s" % ("w3=%s" % w) for w in study.w3_values)
        print(head)
        print("-" * len(head))
        for key, label in order:
            cells = []
            for w3 in study.w3_values:
                rows = study.rows(ram["key"], w3, key)
                base = study.rows(ram["key"], w3, "det_ol")
                if not rows:
                    cells.append("%10s" % "-")
                elif key == "det_ol":
                    mean, _, _ = mean_ci(rows, "overall_cost")
                    cells.append("%10.0f" % mean)
                else:
                    d = paired(rows, base)
                    bmean, _, _ = mean_ci(base, "overall_cost")
                    cells.append("%10s" % ("%+.1f%%" % (100 * d[0] / bmean) if d else "-"))
            print("%-30s%s" % (label, "".join(cells)))
        print("(first row absolute cost, the rest paired difference to it; "
              "%d streams)" % len(study.man["simulation_seeds"]))

    heur = study.heuristics()
    if heur:
        print()
        print("=== model-free placements (independent of w_3 and memory model) ===")
        print("%-26s %10s %8s" % ("placement", "C", "+-95%"))
        for label, mean, half, _ in heur:
            print("%-26s %10.0f %8.0f" % (label, mean, half))

    if latex:
        print()
        print("% ---- tab:policy ----")
        for ram in study.ram_models:
            for key, label in order:
                cells = []
                for w3 in study.w3_values:
                    rows = study.rows(ram["key"], w3, key)
                    base = study.rows(ram["key"], w3, "det_ol")
                    if not rows:
                        cells.append("--")
                    elif key == "det_ol":
                        cells.append("$%.0f$" % mean_ci(rows, "overall_cost")[0])
                    else:
                        d = paired(rows, base)
                        bmean, _, _ = mean_ci(base, "overall_cost")
                        cells.append("$%+.1f$" % (100 * d[0] / bmean) if d else "--")
                print("%s&%s&%s\\\\" % (ram["key"], label, "&".join(cells)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--w3", type=float, default=1.5)
    parser.add_argument("--paper", action="store_true",
                        help="export into UCC_MINLP/Figures")
    parser.add_argument("--latex", action="store_true")
    parser.add_argument("--export", default=None)
    parser.add_argument("--heuristic-style", choices=("row", "rule"), default="row",
                        help="how policy_bars_paper.pdf shows the model-free "
                             "references: row gives each one a bar of its own "
                             "between the two memory-model blocks, with its cost "
                             "split and its interval; rule draws it as a vertical "
                             "line across all bars")
    args = parser.parse_args()

    study = Study(args.manifest)
    export = args.export or (os.path.join("UCC_MINLP", "Figures") if args.paper
                             else os.path.join("results", "run_%d" % study.man["base_run"]))
    os.makedirs(export, exist_ok=True)

    print_tables(study, args.latex)
    bar_figure(study, args.w3, export)
    w3_figure(study, export)
    if args.paper:
        paper_figure(study, export)
        paper_bar_figure(study, args.w3, export, args.heuristic_style)
