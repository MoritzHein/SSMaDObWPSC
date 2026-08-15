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

# Series of the w_3 figure, in fixed order. Kept to six so no hue is reused.
W3_SERIES = [
    ("det_ol", "det., open loop", "#BBBBBB", "o", "--"),
    ("stoch_ol", "stoch., open loop", "#888888", "s", "--"),
    ("det_ll", "det. + least loaded", "#CC6677", "v", "-"),
    ("free_sup", "stoch. + free sample, supp.", "#DDCC77", "^", "-"),
    ("free_wide", "stoch. + free sample, wide", "#4477AA", "D", "-"),
    ("stoch_ll", "stoch. + least loaded", "#332288", "P", "-"),
]

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

    def heuristics(self):
        """Model-free placements: (label, mean cost, half-width, streams)."""
        out = []
        for num_run, info in sorted(self.man.get("heuristic_runs", {}).items(),
                                    key=lambda kv: int(kv[0])):
            rows = load_run(int(num_run))
            mean, half, n = mean_ci(rows, "overall_cost_comparison")
            if n:
                out.append((info["label"], mean, half, n))
        return out

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
            for comp, _, color in COMPONENTS:
                val, _, _ = mean_ci(rows, comp)
                # Same weighting as the paper's cost figure: w_1 Z + w_2 B +
                # w_2 W. The stack only approximates overall_cost, which is
                # w_1 Z + w_2 * (realized response times).
                top.barh(y, study.weight[comp] * val, left=left, height=0.62,
                         color=color, edgecolor="white", linewidth=0.8, zorder=3)
                left += study.weight[comp] * val

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
    args = parser.parse_args()

    study = Study(args.manifest)
    export = args.export or (os.path.join("UCC_MINLP", "Figures") if args.paper
                             else os.path.join("results", "run_%d" % study.man["base_run"]))
    os.makedirs(export, exist_ok=True)

    print_tables(study, args.latex)
    bar_figure(study, args.w3, export)
    w3_figure(study, export)
