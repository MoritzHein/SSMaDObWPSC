# Figure for the Poisson-vs-Azure-trace study (settings/trace_campaign.json).
#
# Usage: visualization_trace.py [--manifest=settings/trace_campaign.json]
#                              [--arms=poisson,trace_hi]
#                              [--policies=det_ol,stoch_ol,stoch_ll]
#                              [--export=results/trace]
#
# Produces trace_bars.pdf/.png: one horizontal stacked bar per (arm, policy),
# split into w_1 Z, w_2 B and w_2 W, with the arms as blocks on the y axis so
# every bar is read against one common x scale.
#
# Three conventions carried over from visualization_policy.py, each for a
# reason that has bitten this repository before.
#
#   * THE STACK IS NOT THE THREE REPORTED FIELDS. `time_cost` is the ANALYTIC
#     basic time - the critical path of the workflow times the number of
#     requests - and runs about 1.3 % below the realized one, so stacking the
#     three reported fields lands the bar short of `overall_cost` and a
#     reference line can fall between a bar's top and its own marker. Money and
#     waiting enter overall_cost exactly as reported, so the realized basic time
#     is the remainder, w_2 B = overall_cost - w_1 Z - w_2 W. Exact: verified to
#     0 on this campaign.
#   * BARS START AT ZERO. The policies differ by a few per cent, and a stacked
#     bar on a cut axis states a composition that is not there. The effect is
#     carried by the paired annotation instead, which is the only form in which
#     a difference of this size is resolvable.
#   * THE ANNOTATION IS PAIRED, THE ERROR BAR IS NOT. The error bar is the
#     absolute t interval of the level and is dominated by how much the ten
#     request streams differ from one another; the annotation is the paired
#     difference to the deterministic open loop OF THE SAME ARM, where the
#     stream cancels. They differ by roughly a factor of four here, so the
#     figure says so in as many words rather than letting the wide bar be read
#     as the uncertainty of the comparison.
import argparse
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

try:
    import scienceplots  # noqa: F401
    plt.style.use(["science", "no-latex"])
except Exception:
    pass

DEFAULT_MANIFEST = os.path.join("settings", "trace_campaign.json")
DEFAULT_EXPORT = os.path.join("results", "trace")

# Paul Tol muted, the same triple every cost figure in this repository uses.
# Re-checked for colour-vision deficiency (Machado 2009 severity 1.0, OKLab
# x100): the smallest separation over all three pairs and over
# protan/deutan/tritan is dE 21.7 against a target of 8, and the smallest under
# normal vision is 26.4 against a floor of 15.
COMPONENTS = [
    ("money_cost", r"$w_1 Z$ (monetary)", "#332288"),
    ("time_cost", r"$w_2 B$ (basic time)", "#DDCC77"),
    ("complete_time_cost", r"$w_2 W$ (waiting)", "#CC6677"),
]

# Short y labels: the placement first, the runtime rule second. One IEEE column
# is 3.44 in and the y labels have to fit in it. The memory model is NOT named,
# because the campaign holds it fixed at eligibility + l2 on every row, and
# naming a constant on every bar would suggest it varies. It belongs in the
# caption.
SHORT = {
    "det_ol": "det., open loop",
    "det_ll": "det., least loaded",
    "stoch_ol": "stoch., open loop",
    "stoch_wide": "stoch., free sample",
    "stoch_ll": "stoch., least loaded",
}

COLUMNWIDTH = 3.44488       # IEEE column width, from config_mpl.py

T_QUANTILE = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
              7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179}


def t_quantile(df):
    return T_QUANTILE.get(df, 1.96)


def load_run(num_run):
    path = os.path.join("results", "run_%d" % num_run, "results_seeds.json")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


def mean_ci(rows, key):
    """Unpaired mean and t interval; the ABSOLUTE level of one run."""
    vals = [r[key] for r in rows if r.get(key) is not None]
    n = len(vals)
    if n == 0:
        return float("nan"), 0.0, 0
    mean = sum(vals) / n
    if n < 2:
        return mean, 0.0, n
    var = sum((v - mean) ** 2 for v in vals) / (n - 1)
    return mean, t_quantile(n - 1) * math.sqrt(var / n), n


def paired_rel(rows, base_rows, key="overall_cost"):
    """Paired difference as a percentage, its t interval, and the win count.

    Joined on simulation_seed: the two runs saw the same arrivals and, in a
    trace arm, the same window of the trace, so the stream cancels.

    RATIO OF MEANS, 100 * mean(a-b) / mean(b) with the interval of the absolute
    differences rescaled by the same mean(b) - the convention of
    visualization_policy.py and visualization_ram_w3.py. See the long note in
    trace_table.paired for why this is not the mean of the per-stream ratios
    and what the difference is worth.
    """
    base = {r.get("simulation_seed"): r for r in base_rows}
    pairs = [(r[key], base[r["simulation_seed"]][key])
             for r in rows
             if r.get("simulation_seed") in base
             and r.get(key) is not None
             and base[r["simulation_seed"]].get(key) is not None]
    n = len(pairs)
    if n < 2:
        return None
    diffs = [a - b for a, b in pairs]
    base_mean = sum(b for _, b in pairs) / n
    if not base_mean:
        return None
    mean = sum(diffs) / n
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    half = t_quantile(n - 1) * math.sqrt(var / n)
    return (100.0 * mean / base_mean, 100.0 * half / base_mean, n,
            sum(1 for d in diffs if d < 0))


def weighted_split(rows, w1, w2):
    """The three weighted contributions, summing EXACTLY to the measured cost."""
    z = w1 * mean_ci(rows, "money_cost")[0]
    wait = w2 * mean_ci(rows, "complete_time_cost")[0]
    total = mean_ci(rows, "overall_cost")[0]
    return {"money_cost": z, "time_cost": total - z - wait,
            "complete_time_cost": wait}


def build(manifest, arms, policies):
    """(arm label, [(policy label, split, mean, half, note, wins)]) per arm."""
    runs = manifest["runs"]
    by = {(i["arm"], i["policy"]): int(n) for n, i in runs.items()}
    arm_label = {a["key"]: a["label"] for a in manifest["arms"]}
    pol_label = {p["key"]: p["label"] for p in manifest["policies"]}
    baseline = manifest["policies"][0]["key"]

    any_run = next(iter(runs))
    cfg = json.load(open(os.path.join("settings", "parameters_%s.json" % any_run)))
    w1, w2 = cfg["w_1"], cfg["w_2"]

    blocks = []
    for arm in arms:
        base_rows = load_run(by[(arm, baseline)])
        entries = []
        for pol in policies:
            rows = load_run(by[(arm, pol)])
            if not rows:
                raise SystemExit("no results for arm %s policy %s - run "
                                 "run_trace_study.sh first" % (arm, pol))
            mean, half, _ = mean_ci(rows, "overall_cost")
            if pol == baseline:
                note, wins = "reference", None
            else:
                d = paired_rel(rows, base_rows)
                fmt = ("%+.1f\\%%" if plt.rcParams.get("text.usetex")
                       else "%+.1f%%")
                note = fmt % d[0] if d else ""
                wins = d[3] if d else None
            entries.append((SHORT.get(pol, pol_label.get(pol, pol)),
                            weighted_split(rows, w1, w2), mean, half, note,
                            wins, by[(arm, pol)], d if pol != baseline else None))
        blocks.append((arm_label[arm], entries))
    return blocks, w1, w2


def figure(blocks, export, sim_time):
    rows_only = [e for _, entries in blocks for e in entries]
    bar_max = max(e[2] for e in rows_only)

    fig, ax = plt.subplots(figsize=(COLUMNWIDTH, 0.24 * len(rows_only) + 0.72))
    ypos = np.arange(len(rows_only))[::-1]

    for y, (label, split, mean, half, note, wins, _run, _d) in zip(ypos, rows_only):
        left = 0.0
        for comp, _, color in COMPONENTS:
            ax.barh(y, split[comp], left=left, height=0.66, color=color,
                    edgecolor="white", linewidth=0.7, zorder=3)
            left += split[comp]
        ax.errorbar(mean, y, xerr=half, fmt="|", color="0.15", markersize=4,
                    markeredgewidth=0.9, elinewidth=0.7, capsize=1.2, zorder=5)
        if note:
            ax.annotate(note, xy=(mean + half, y), xytext=(3, 0),
                        textcoords="offset points", fontsize=5.6,
                        color="0.15" if note != "reference" else "0.45",
                        va="center", ha="left", zorder=7)

    ax.set_xlim(0, 1.22 * bar_max)
    ax.set_ylim(-0.7, len(rows_only) - 0.3)

    # Blocks on the y axis, separated by a rule and named on the right.
    centres, labels, ri = [], [], 0
    for bi, (arm_label, entries) in enumerate(blocks):
        idx = list(range(ri, ri + len(entries)))
        ri += len(entries)
        if bi:
            ax.axhline(ypos[idx[0]] + 0.5, color="0.25", linewidth=0.9, zorder=6)
        centres.append(sum(ypos[i] for i in idx) / float(len(idx)))
        labels.append(arm_label)

    ax.set_yticks(ypos)
    ax.set_yticklabels([e[0] for e in rows_only], fontsize=6)
    ax.set_xlabel("realized cost $C$")
    ax.xaxis.label.set_size(7)
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(axis="x", color="0.92", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)

    side = ax.twinx()
    side.set_ylim(ax.get_ylim())
    side.set_yticks(centres)
    side.set_yticklabels(labels, fontsize=6, rotation=90, va="center")
    side.tick_params(axis="y", length=0)

    handles = [Patch(facecolor=c, edgecolor="white", label=l)
               for _, l, c in COMPONENTS]
    fig.legend(handles=handles, ncol=3, fontsize=5.6, loc="upper center",
               bbox_to_anchor=(0.5, 1.01), frameon=False, handlelength=1.5,
               columnspacing=1.0, labelspacing=0.3, handletextpad=0.5)
    fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.93))

    if not os.path.isdir(export):
        os.makedirs(export)
    path = os.path.join(export, "trace_bars.pdf")
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote %s (and .png)" % path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--arms", default="poisson,trace_hi")
    parser.add_argument("--policies", default="det_ol,stoch_ol,stoch_ll")
    parser.add_argument("--export", default=DEFAULT_EXPORT)
    args = parser.parse_args()

    manifest = json.load(open(args.manifest))
    arms = args.arms.split(",")
    policies = args.policies.split(",")
    blocks, w1, w2 = build(manifest, arms, policies)

    # Print exactly what is drawn, so every number in the figure is checkable
    # against trace_table.py without reading the PDF.
    print("simulation_time = %d s, w_1 = %s, w_2 = %s, 10 paired streams"
          % (manifest["simulation_time"], w1, w2))
    print("%-24s %-30s %5s %10s %10s %10s %10s %9s %6s"
          % ("arm", "policy", "run", "w1 Z", "w2 B", "w2 W", "C", "paired", "wins"))
    for arm_label, entries in blocks:
        for label, split, mean, half, note, wins, run, d in entries:
            ptxt = "--" if d is None else "%+.2f +/- %.2f%%" % (d[0], d[1])
            print("%-24s %-30s %5d %10.1f %10.1f %10.1f %10.1f %9s %6s"
                  % (arm_label, label, run,
                     split["money_cost"], split["time_cost"],
                     split["complete_time_cost"], mean, ptxt,
                     "--" if wins is None else "%d/10" % wins))
            stack = sum(split.values())
            assert abs(stack - mean) < 1e-6, (stack, mean)
        arm_label = ""
    print("(stack sums to the measured cost exactly; the error bar is the "
          "ABSOLUTE t interval,")
    print(" the annotation the PAIRED difference to the open loop of the same arm)")

    figure(blocks, args.export, manifest["simulation_time"])
