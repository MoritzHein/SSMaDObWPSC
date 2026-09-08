# Tables for the Poisson-vs-Azure-trace study (runs 660-674).
#
# Usage: trace_table.py [--manifest=settings/trace_campaign.json] [--latex]
#
# Reports, per arrival arm:
#
#   * the realised burstiness of the arrival stream itself, so the reader can
#     see how much the arms actually differ before reading any cost;
#   * absolute cost per policy with a t-based confidence interval;
#   * the PAIRED difference against the deterministic open-loop policy of the
#     SAME arm, with its CI and the win count over the ten streams. Paired,
#     because the ten streams differ by an order of magnitude more than the
#     policies do - the arms share a seed and therefore a trace window, so the
#     difference is taken within a stream and the window cancels;
#   * the backlog growth slope. There is no steady state at this arrival rate:
#     waiting grows monotonically to the end of the stream, so any cost is a
#     function of the horizon. The slope is the horizon-independent statistic
#     and must be quoted next to any percentage.
#
# Every number here is at the campaign horizon of simulation_time=400 s and is
# therefore directly comparable with the RAM x w_3 and policy figures. The
# arrival trace is normalized over a window that covers exactly that horizon
# (window_minutes=7); see the note at the top of trace_experiments.py for why a
# longer normalization window would silently change the offered load instead of
# only its shape.
import argparse
import json
import math
import os

import numpy as np

DEFAULT_MANIFEST = os.path.join("settings", "trace_campaign.json")
# t_{0.975, 9} for the ten paired streams. Using 1.96 here would understate the
# interval by 15% at n=10.
T_QUANTILE = {9: 2.262, 8: 2.306, 7: 2.365, 4: 2.776}
COMPONENTS = [("money_cost", "w1 Z"), ("time_cost", "w2 B"),
              ("complete_time_cost", "w2 W")]


def load_seeds(num_run):
    path = os.path.join("results", "run_%d" % num_run, "results_seeds.json")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


def load_latency(num_run):
    path = os.path.join("results", "run_%d" % num_run, "latency_seeds.json")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


def tci(values):
    """Mean and half-width of the t-based 95% interval."""
    v = [x for x in values if x is not None]
    if not v:
        return float("nan"), 0.0
    mean = sum(v) / len(v)
    if len(v) < 2:
        return mean, 0.0
    sd = math.sqrt(sum((x - mean) ** 2 for x in v) / (len(v) - 1))
    t = T_QUANTILE.get(len(v) - 1, 2.262)
    return mean, t * sd / math.sqrt(len(v))


def paired(base_rows, other_rows, key="overall_cost"):
    """Paired difference other-base as a percentage, its CI, and the win count.

    Joined on simulation_seed: both runs saw the same arrivals and, in the
    trace arms, the same window of the trace.

    RATIO OF MEANS, 100 * mean(a-b) / mean(b), not the mean of the per-stream
    ratios. The two differ here because the per-stream saving is correlated
    with how expensive the stream is - at w_3=1.5 the five cheap streams save
    2.95 % and the five expensive ones 4.53 % - and the ratio of means is
    algebraically a cost-weighted mean of the per-stream ratios, since
    sum(a-b)/sum(b) weights stream k by b_k. It therefore runs 0.05 to 0.08
    points more negative, enough to move a one-decimal label.

    This is the convention of visualization_policy.py and
    visualization_ram_w3.py, and the paper quotes all three, so they have to
    agree. The interval is the t interval of the ABSOLUTE differences rescaled
    by the same constant mean(b), which keeps the half-width consistent with
    the point estimate.
    """
    b = {r["simulation_seed"]: r for r in base_rows if "simulation_seed" in r}
    o = {r["simulation_seed"]: r for r in other_rows if "simulation_seed" in r}
    keys = sorted(set(b) & set(o))
    if not keys:
        return float("nan"), 0.0, 0, 0
    diffs = [o[k][key] - b[k][key] for k in keys]
    base_mean = sum(b[k][key] for k in keys) / len(keys)
    mean, half = tci(diffs)
    if not base_mean:
        return float("nan"), 0.0, len(diffs), 0
    return (100.0 * mean / base_mean, 100.0 * half / base_mean, len(diffs),
            sum(1 for x in diffs if x < 0))


def burstiness(num_run, sim_time):
    """CV and peak/mean of the per-minute arrival count, averaged over streams.

    Only whole minutes are binned, so at a horizon that is not a multiple of 60
    the trailing seconds are dropped: at 400 s the statistic covers the first
    six minutes, i.e. 360 of the 400 simulated seconds. Six buckets is few, and
    the CV of the arrival COUNT still carries the Poisson sampling noise of the
    stream on top of the trace intensity, so read this column as "how different
    are the arms" and not as an estimate of the trace's own CV.
    """
    cvs, peaks = [], []
    minutes = sim_time // 60
    for row in load_latency(num_run):
        a = np.asarray(row["arrivals"])
        h, _ = np.histogram(a, bins=np.arange(0, minutes * 60 + 1, 60))
        if h.mean() <= 0:
            continue
        cvs.append(h.std() / h.mean())
        peaks.append(h.max() / h.mean())
    if not cvs:
        return float("nan"), float("nan")
    return float(np.mean(cvs)), float(np.mean(peaks))


def backlog_slope(num_run, sim_time, buckets=6):
    """Seconds of added waiting per 1000 s of elapsed arrival time.

    Waiting is bucketed by arrival time and pooled over the streams, then a
    straight line is fitted. A flat slope would mean the system reached steady
    state; it does not at this arrival rate, which is exactly why this number
    has to be reported next to any cost percentage.
    """
    edges = np.linspace(0, sim_time, buckets + 1)
    sums = np.zeros(buckets)
    counts = np.zeros(buckets)
    for row in load_latency(num_run):
        a = np.asarray(row["arrivals"], dtype=float)
        w = np.asarray(row["waits"], dtype=float)
        idx = np.clip(np.digitize(a, edges) - 1, 0, buckets - 1)
        for i, wait in zip(idx, w):
            sums[i] += wait
            counts[i] += 1
    ok = counts > 0
    if ok.sum() < 2:
        return float("nan")
    centers = 0.5 * (edges[:-1] + edges[1:])[ok]
    means = (sums[ok] / counts[ok])
    slope = np.polyfit(centers, means, 1)[0]
    return float(slope * 1000.0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--latex", action="store_true")
    args = parser.parse_args()

    manifest = json.load(open(args.manifest))
    sim_time = manifest["simulation_time"]
    runs = manifest["runs"]
    arms = [a["key"] for a in manifest["arms"]]
    arm_label = {a["key"]: a["label"] for a in manifest["arms"]}
    policies = [p["key"] for p in manifest["policies"]]
    pol_label = {p["key"]: p["label"] for p in manifest["policies"]}

    # run number by (arm, policy)
    by = {}
    for num, info in runs.items():
        by[(info["arm"], info["policy"])] = int(num)

    print("Poisson vs. Azure Functions trace, simulation_time = %d s, "
          "10 paired streams" % sim_time)
    print("(a trace stream's window is picked from its seed, so the ten streams "
          "are ten different real times of day;")
    print(" the trace is normalized to unit mean over a %d-minute window, which "
          "covers the horizon)" % manifest.get("window_minutes", 30))
    print()
    print("ARRIVAL PROCESS AS SIMULATED")
    print("  %-24s %10s %12s %10s" % ("arm", "per-min CV", "vs Poisson", "peak/mean"))
    base_cv = None
    for arm in arms:
        cv, peak = burstiness(by[(arm, policies[0])], sim_time)
        if base_cv is None:
            base_cv = cv
        print("  %-24s %10.3f %11.2fx %10.2f"
              % (arm_label[arm], cv, cv / base_cv, peak))
    print()

    print("COST (t-based 95%% CI over 10 streams) and PAIRED DIFFERENCE vs the "
          "deterministic open loop of the same arm")
    header = ("  %-24s %-36s %12s %9s %9s %9s %6s %8s"
              % ("arm", "policy", "cost", "+/-", "paired", "+/-", "wins",
                 "slope"))
    print(header)
    print("  " + "-" * (len(header) - 2))
    table = {}
    for arm in arms:
        base_rows = load_seeds(by[(arm, policies[0])])
        for pol in policies:
            rows = load_seeds(by[(arm, pol)])
            mean, half = tci([r["overall_cost"] for r in rows])
            if pol == policies[0]:
                dtxt, dhtxt, wtxt = "--", "--", "--"
                d = dh = 0.0
                wins = 0
            else:
                d, dh, n, wins = paired(base_rows, rows)
                dtxt = "%+.2f%%" % d
                dhtxt = "%.2f" % dh
                wtxt = "%d/%d" % (wins, n)
            slope = backlog_slope(by[(arm, pol)], sim_time)
            table[(arm, pol)] = (mean, half, d, dh, wins, slope)
            print("  %-24s %-36s %12.1f %9.1f %9s %9s %6s %8.1f"
                  % (arm_label[arm] if pol == policies[0] else "",
                     pol_label[pol], mean, half, dtxt, dhtxt, wtxt, slope))
        print()

    print("both intervals are t-based 95% over the ten streams. The two are not")
    print("comparable in width and only the PAIRED one is a statement about the")
    print("policy: the absolute one is dominated by how much the ten streams")
    print("differ from each other, which the pairing cancels.")
    print("slope = seconds of added waiting per 1000 s of elapsed arrival time;")
    print("there is no steady state here, so the cost above is a function of the")
    print("horizon and the slope is what is horizon-independent.")

    # ---- Does the advantage survive the bursty arm? The question the review asks.
    print()
    print("EFFECT OF BURSTINESS ON EACH POLICY'S ADVANTAGE")
    print("  %-36s %s" % ("policy", "  ".join("%-22s" % arm_label[a] for a in arms)))
    for pol in policies[1:]:
        cells = []
        for arm in arms:
            _, _, d, dh, _, _ = table[(arm, pol)]
            cells.append("%-22s" % ("%+.2f%% +/- %.2f" % (d, dh)))
        print("  %-36s %s" % (pol_label[pol], "  ".join(cells)))

    if args.latex:
        print()
        print("% ---- tab:trace ----")
        print("%% simulation_time = %d s, 10 paired streams, paired against the"
              % sim_time)
        print("% deterministic open loop of the SAME arrival arm.")
        for arm in arms:
            cv, peak = burstiness(by[(arm, policies[0])], sim_time)
            print("\\multicolumn{5}{l}{\\textit{%s} (per-minute CV %.2f)} \\\\"
                  % (arm_label[arm], cv))
            for pol in policies:
                mean, half, d, dh, wins, slope = table[(arm, pol)]
                dtxt = ("--" if pol == policies[0]
                        else "$%+.1f \\pm %.1f$" % (d, dh))
                wtxt = "--" if pol == policies[0] else "%d/10" % wins
                print("\\quad %s & %.0f $\\pm$ %.0f & %s & %s & %.1f \\\\"
                      % (pol_label[pol], mean, half, dtxt, wtxt, slope))
            print("\\addlinespace")
