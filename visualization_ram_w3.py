# Visualization of the RAM-model x w_3 x policy study.
#
# Usage: visualization_ram_w3.py [--manifest=settings/ram_w3_campaign.json]
#                                [--export=results/run_<base_run>]
#
# Draws the deterministic and the stochastic selection policy side by side over
# the w_3 sweep, in two subplots that share the y-axis:
#
#     top    - RAM capacity model OFF (aggregate capacity, the previous model)
#     bottom - eligibility + l2 capacity norm  (see ram_capacity_model.md)
#
# Every bar is stacked into the three weighted cost contributions
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
# Error bars are the 95% confidence interval of overall_cost over the
# independent request streams produced by resimulate_seeds.py. Without a
# results_seeds.json the script falls back to the single results.json and draws
# no error bars (a single simulation carries roughly 10% Monte-Carlo noise, so
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


def stats(rows, key):
    v = [r[key] for r in rows if key in r and r[key] is not None]
    if not v:
        return float("nan"), 0.0
    mean = sum(v) / len(v)
    if len(v) < 2:
        return mean, 0.0
    sd = math.sqrt(sum((x - mean) ** 2 for x in v) / (len(v) - 1))
    return mean, 1.96 * sd / math.sqrt(len(v))


def save(fig, export_path, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(export_path, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}.pdf / .png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--export", default=None,
                        help="output folder (default results/run_<base_run>)")
    args = parser.parse_args()

    with open(args.manifest, "r") as f:
        man = json.load(f)

    w1, w2 = man["w_1"], man["w_2"]
    w3_values = man["w3_values"]
    ram_settings = man["ram_settings"]
    policies = man["policies"]
    weight = {"money_cost": w1, "time_cost": w2, "complete_time_cost": w2}

    export_path = args.export or os.path.join("results", f"run_{man['base_run']}")
    os.makedirs(export_path, exist_ok=True)

    # index the manifest by (ram, w_3, policy)
    cell = {}
    for num_run, info in man["runs"].items():
        cell[(info["ram"], info["w_3"], info["policy"])] = int(num_run)

    replicated = True
    missing = []
    fig, axes = plt.subplots(len(ram_settings), 1, sharey=True,
                             figsize=(7.15, 6.4))
    axes = np.atleast_1d(axes)

    x = np.arange(len(w3_values), dtype=float)
    width = 0.38
    offsets = {p["key"]: (i - (len(policies) - 1) / 2) * width
               for i, p in enumerate(policies)}

    for ax, ram in zip(axes, ram_settings):
        for pol in policies:
            xs, bottoms = [], []
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
                    xs.append(x[wi] + offsets[pol["key"]])
                    continue
                replicated = replicated and has_seeds
                for k, _, _ in COMPONENTS:
                    m, _ = stats(rows, k)
                    heights[k].append(weight[k] * m)
                m, h = stats(rows, "overall_cost")
                totals.append(m)
                errs.append(h)
                xs.append(x[wi] + offsets[pol["key"]])

            bottom = np.zeros(len(w3_values))
            for k, label, color in COMPONENTS:
                vals = np.array(heights[k])
                ax.bar(xs, vals, width * 0.92, bottom=bottom, color=color,
                       edgecolor="black", linewidth=0.4,
                       hatch=POLICY_HATCH[pol["key"]],
                       label=label if (pol is policies[0] and ax is axes[0]) else None)
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

    axes[-1].set_xlabel("utilization weight $w_3$")

    handles = [Patch(facecolor=c, edgecolor="black", label=l) for _, l, c in COMPONENTS]
    handles += [Patch(facecolor="white", edgecolor="black",
                      hatch=POLICY_HATCH[p["key"]], label=p["label"]) for p in policies]
    handles += [Line2D([0], [0], color="black", marker="_", linestyle="none",
                       markersize=11, markeredgewidth=1.4,
                       label="measured overall cost" + (" (95% CI)" if replicated else ""))]
    axes[0].legend(handles=handles, ncol=3, fontsize=8, loc="upper center",
                   bbox_to_anchor=(0.5, 1.42), frameon=False)

    n_streams = len(man["simulation_seeds"]) if replicated else 1
    fig.suptitle(f"{man['num_levels']} levels, {man['num_workflows']} workflows, "
                 f"seed {man['seed']}, $t_{{sim}}={man['simulation_time']}$ s, "
                 f"{n_streams} request stream{'s' if n_streams > 1 else ''}",
                 y=0.995, fontsize=9)
    fig.tight_layout()
    save(fig, export_path, "ram_w3_comparison")

    if missing:
        print(f"  WARNING: {len(missing)} cells had no results and were drawn "
              f"empty: {sorted(set(missing))}")
    if not replicated:
        print("  WARNING: no results_seeds.json found - bars come from a single "
              "simulation each (~10% Monte-Carlo noise) and carry no error bars. "
              "Run resimulate_seeds.py for confidence intervals.")

    # numeric readout alongside the figure
    print("\noverall cost (mean over request streams):")
    print("%-6s %-6s %12s %12s %12s" % ("ram", "w_3", "det", "stoch", "stoch-det"))
    for ram in ram_settings:
        for w3 in w3_values:
            vals = {}
            for pol in policies:
                num_run = cell.get((ram["key"], w3, pol["key"]))
                rows, _ = load_run(num_run) if num_run else ([], False)
                vals[pol["key"]] = stats(rows, "overall_cost")[0] if rows else float("nan")
            print("%-6s %-6g %12.1f %12.1f %12.1f"
                  % (ram["key"], w3, vals.get("det", float("nan")),
                     vals.get("stoch", float("nan")),
                     vals.get("stoch", float("nan")) - vals.get("det", float("nan"))))
