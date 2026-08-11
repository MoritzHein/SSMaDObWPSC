# Waiting time against elapsed simulation time, for one or several runs.
#
# Usage: visualization_waiting_over_time.py --runs=316,317
#        visualization_waiting_over_time.py --runs=316 --window=25
#        visualization_waiting_over_time.py --runs=300,301,316,317 --export=some/dir
#
# Each request is placed in the bucket of its ARRIVAL time and the mean waiting
# time of the bucket is plotted against elapsed time. This answers a question the
# cost bars cannot: is the queue in a steady state, or is the backlog still
# growing when the simulation stops?
#
#   flat line     - the system has settled; the cost figures measure a steady
#                   state and are meaningful at any horizon
#   rising line   - the backlog is still building; total waiting grows faster
#                   than linearly in simulation_time, so every cost number is a
#                   function of the horizon and does not converge
#
# The fitted slope printed in the legend (seconds of waiting added per 1000 s of
# elapsed time) is the horizon-independent way to compare two policies: unlike a
# cost difference, it does not depend on where the simulation was stopped.
#
# Pass several run numbers to overlay them - the intended use is a deterministic
# and a stochastic run of the same cell, e.g. --runs=316,317. Labels are taken
# from settings/ram_w3_campaign.json when the runs belong to that campaign, so
# the legend reads "l2, w_3=1.5, stochastic" rather than "run 317".
#
# Data source: results/run_<N>/latency_seeds.json (all request streams, written
# by resimulate_seeds.py), falling back to the single baseline stream in
# results/run_<N>/detailed_simulation_run_<N>.json. Pooling over streams matters
# here - a single stream is noisy enough to show a dip that looks like recovery.
import argparse
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import scienceplots  # noqa: F401
    plt.style.use(["science", "no-latex"])
except Exception:
    pass

DEFAULT_MANIFEST = os.path.join("settings", "ram_w3_campaign.json")


def load_streams(num_run):
    """Request streams of one run as {seed, arrivals, waits}; pooled preferred."""
    base = os.path.join("results", f"run_{num_run}")
    latency_path = os.path.join(base, "latency_seeds.json")
    if os.path.exists(latency_path):
        with open(latency_path, "r") as f:
            entries = json.load(f)
        streams = [{"seed": e.get("simulation_seed"), "arrivals": e["arrivals"],
                    "waits": e["waits"]}
                   for e in entries if "arrivals" in e and "waits" in e]
        if streams:
            return streams, len(streams)
    detailed_path = os.path.join(base, f"detailed_simulation_run_{num_run}.json")
    if os.path.exists(detailed_path):
        with open(detailed_path, "r") as f:
            logs = json.load(f).get("request_logs", [])
        if logs:
            return [{"seed": None, "arrivals": [r["arrival"] for r in logs],
                     "waits": [r["wait"] for r in logs]}], 1
    return [], 0


def window_means(stream, window, sim_time):
    """Mean waiting per arrival window for one stream; NaN for empty windows."""
    edges = np.arange(0, sim_time + window, window, dtype=float)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        v = [w for a, w in zip(stream["arrivals"], stream["waits"]) if lo <= a < hi]
        out.append(sum(v) / len(v) if v else float("nan"))
    return np.array(out), (edges[:-1] + edges[1:]) / 2


def mean_ci(values):
    """Mean and 95% half-width of a list, ignoring NaNs."""
    v = [x for x in values if not math.isnan(x)]
    if not v:
        return float("nan"), 0.0
    m = sum(v) / len(v)
    if len(v) < 2:
        return m, 0.0
    sd = math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))
    return m, 1.96 * sd / math.sqrt(len(v))


def horizon(num_run):
    """simulation_time of a run, or None when the parameter file is missing."""
    path = os.path.join("settings", f"parameters_{num_run}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f).get("simulation_time")


def label_for(num_run, manifest):
    if manifest:
        info = manifest.get("runs", {}).get(str(num_run))
        if info:
            policy = {p["key"]: p["label"] for p in manifest["policies"]}
            return (f"run {num_run}: {info['ram']}, $w_3={info['w_3']:g}$, "
                    f"{policy.get(info['policy'], info['policy'])}")
    return f"run {num_run}"


def bucket(streams, window, sim_time):
    """Mean waiting per arrival window, and its 95% CI across request streams.

    The mean is taken per stream first and then across streams, so every stream
    carries the same weight regardless of how many requests it happened to draw.
    """
    per_stream, centres = [], None
    for s in streams:
        m, centres = window_means(s, window, sim_time)
        per_stream.append(m)
    keep, means, halfwidths = [], [], []
    for i, c in enumerate(centres):
        m, h = mean_ci([row[i] for row in per_stream])
        if math.isnan(m):
            continue
        keep.append(c)
        means.append(m)
        halfwidths.append(h)
    return np.array(keep), np.array(means), np.array(halfwidths)


def paired_bucket(streams_a, streams_b, window, sim_time):
    """Per-window mean waiting of b minus a, paired on the request stream.

    Both runs are simulated on the same list of seeds, so the difference can be
    formed within a stream. That removes the stream-to-stream variation the two
    runs share, which is what makes the unpaired bands overlap.
    """
    a = {s["seed"]: s for s in streams_a}
    b = {s["seed"]: s for s in streams_b}
    seeds = [k for k in a if k is not None and k in b]
    if not seeds:
        return np.array([]), np.array([]), np.array([]), 0
    diffs, centres = [], None
    for seed in sorted(seeds):
        ma, centres = window_means(a[seed], window, sim_time)
        mb, _ = window_means(b[seed], window, sim_time)
        diffs.append(mb - ma)
    keep, means, halfwidths = [], [], []
    for i, c in enumerate(centres):
        m, h = mean_ci([row[i] for row in diffs])
        if math.isnan(m):
            continue
        keep.append(c)
        means.append(m)
        halfwidths.append(h)
    return np.array(keep), np.array(means), np.array(halfwidths), len(seeds)


def slope_per_1000s(centres, means):
    """Least-squares trend of mean waiting against elapsed time."""
    if len(centres) < 2:
        return float("nan")
    mx, my = centres.mean(), means.mean()
    denom = ((centres - mx) ** 2).sum()
    if denom == 0:
        return float("nan")
    return 1000.0 * ((centres - mx) * (means - my)).sum() / denom


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", required=True,
                        help="one run number or a comma-separated list, "
                             "e.g. 316 or 316,317")
    parser.add_argument("--window", type=int, default=None,
                        help="arrival-window width in seconds "
                             "(default: simulation_time/8)")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST,
                        help="campaign manifest used for the legend labels")
    parser.add_argument("--export", default=None,
                        help="output folder (default results/run_<first run>)")
    parser.add_argument("--name", default="waiting_over_time",
                        help="output file name without extension")
    parser.add_argument("--difference", action="store_true",
                        help="add a lower panel with the paired difference "
                             "(exactly two runs, simulated on the same seeds). "
                             "The per-run bands above are unpaired and overlap "
                             "even when every stream agrees on the sign.")
    args = parser.parse_args()

    runs = [int(x) for x in args.runs.split(",") if x.strip()]

    manifest = None
    if os.path.exists(args.manifest):
        with open(args.manifest, "r") as f:
            manifest = json.load(f)

    horizons = [h for h in (horizon(n) for n in runs) if h]
    sim_time = max(horizons) if horizons else None
    if sim_time is None:
        raise SystemExit("no settings/parameters_<N>.json found for the given runs "
                         "- cannot tell how long the simulation ran")
    if len(set(horizons)) > 1:
        print(f"  WARNING: runs have different simulation_time {sorted(set(horizons))}"
              f" - the curves are not directly comparable")
    window = args.window or max(1, int(sim_time // 8))

    export_path = args.export or os.path.join("results", f"run_{runs[0]}")
    os.makedirs(export_path, exist_ok=True)

    if args.difference and len(runs) != 2:
        raise SystemExit("--difference needs exactly two runs, e.g. --runs=316,317")

    if args.difference:
        fig, (ax, dax) = plt.subplots(2, 1, figsize=(7.15, 4.8), sharex=True,
                                      gridspec_kw={"height_ratios": [3, 1.6]})
    else:
        fig, ax = plt.subplots(figsize=(7.15, 3.4))
        dax = None
    colors = [f"C{i}" for i in range(10)]
    drawn, single_stream = 0, False
    loaded = {}

    print("mean waiting time per arrival window:")
    for i, num_run in enumerate(runs):
        streams, n_streams = load_streams(num_run)
        if not streams:
            print(f"  run {num_run}: no request logs - skipped")
            continue
        loaded[num_run] = streams
        single_stream = single_stream or n_streams == 1
        centres, means, halfwidths = bucket(streams, window, sim_time)
        if not len(centres):
            print(f"  run {num_run}: no requests in any window - skipped")
            continue
        slope = slope_per_1000s(centres, means)
        label = f"{label_for(num_run, manifest)}  [{slope:+.1f} s / 1000 s]"
        color = colors[i % len(colors)]
        ax.plot(centres, means, marker="o", markersize=3.5, linewidth=1.3,
                color=color, label=label)
        if np.any(halfwidths > 0):
            ax.fill_between(centres, means - halfwidths, means + halfwidths,
                            color=color, alpha=0.15, linewidth=0)
        drawn += 1

        print(f"  {label_for(num_run, manifest)}  ({n_streams} stream"
              f"{'s' if n_streams > 1 else ''}, window {window} s)")
        for c, m, h in zip(centres, means, halfwidths):
            print("    t=%6.0f  mean wait %7.1f s  +-%.1f" % (c, m, h))
        print("    trend %+.1f s of waiting per 1000 s of elapsed time "
              "(%.1f -> %.1f)" % (slope, means[0], means[-1]))

    if not drawn:
        raise SystemExit("nothing to plot")

    ax.set_ylabel("mean waiting time\nper request [s]")
    ax.set_xlim(0, sim_time)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8, frameon=False)
    title = (f"waiting time over the run, $t_{{sim}}={sim_time:g}$ s, "
             f"{window} s windows")
    if single_stream:
        title += " (single stream)"
    ax.set_title(title, fontsize=9)

    if dax is not None and len(loaded) == 2:
        a_run, b_run = runs
        centres, means, halfwidths, n_paired = paired_bucket(
            loaded[a_run], loaded[b_run], window, sim_time)
        if n_paired:
            dax.axhline(0.0, color="black", linewidth=0.8)
            dax.plot(centres, means, marker="D", markersize=3.5, linewidth=1.3,
                     color="black")
            dax.fill_between(centres, means - halfwidths, means + halfwidths,
                             color="black", alpha=0.15, linewidth=0)
            lo, hi = dax.get_ylim()
            dax.axhspan(lo, 0.0, color="C2", alpha=0.06, zorder=0)
            dax.set_ylim(lo, hi)
            dax.set_ylabel(f"$\\Delta$ run {b_run} $-$ run {a_run}\n"
                           f"(negative: less waiting)")
            print(f"\npaired difference run {b_run} - run {a_run} "
                  f"({n_paired} shared streams):")
            for c, m, h in zip(centres, means, halfwidths):
                print("    t=%6.0f  %+7.1f s  +-%.1f%s"
                      % (c, m, h, "" if abs(m) > h else "   (CI includes 0)"))
        else:
            print("\n  --difference: the two runs share no request-stream seeds "
                  "- drawing the per-run curves only")
            dax.set_visible(False)

    (dax if dax is not None and dax.get_visible() else ax).set_xlabel(
        "elapsed simulation time [s]  (request arrival)")
    fig.tight_layout()

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(export_path, f"{args.name}.{ext}"),
                    bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {args.name}.pdf / .png to {export_path}")

    if single_stream:
        print("  WARNING: at least one run had no latency_seeds.json and was drawn "
              "from its single baseline stream. One stream is noisy enough to show "
              "a dip that looks like the queue recovering - run resimulate_seeds.py "
              "for pooled curves with confidence bands.")
