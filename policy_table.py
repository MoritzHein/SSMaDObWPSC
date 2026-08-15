# Tables of the blocking-aware selection campaign (settings/policy_campaign.json).
#
# Usage: policy_table.py [--manifest=settings/policy_campaign.json] [--latex]
#
# Every variant is paired on the request stream with two references:
#
#   * its own source run, i.e. the SAME placement served open-loop. This is the
#     effect of the selection rule alone.
#   * the deterministic open-loop run at the same w_3, i.e. the policy of the
#     base method. This is the effect a reader sees end to end.
#
# The comparison stoch_ll against det_ll is the one that matters most: both use
# the identical runtime rule, so what is left between them is the contribution
# of the relaxed placement itself.
#
# Intervals use the t quantile of n-1 degrees of freedom rather than 1.96. With
# ten streams that is 2.262, so a normal quantile understates every half-width
# by about 13 %.
import argparse
import json
import math
import os

# t_{0.975, df} for the sample sizes these campaigns produce.
T_QUANTILE = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
              7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
              14: 2.145, 19: 2.093, 29: 2.045}

# The eligibility + l2 open-loop cells the campaign was cloned from.
DET_BASELINE = {0.5: 312, 1.0: 314, 1.5: 316, 2.0: 318, 2.5: 320, 3.0: 322}


def t_quantile(df):
    if df in T_QUANTILE:
        return T_QUANTILE[df]
    keys = sorted(T_QUANTILE)
    return T_QUANTILE[min(keys, key=lambda k: abs(k - df))]


def load_seeds(num_run):
    path = os.path.join("results", "run_%d" % num_run, "results_seeds.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def paired(rows_a, rows_b, key="overall_cost"):
    """Mean of (a - b) over the shared request streams, with a t interval."""
    if not rows_a or not rows_b:
        return None
    by_seed = {r.get("simulation_seed"): r for r in rows_b}
    diffs = [r[key] - by_seed[r["simulation_seed"]][key]
             for r in rows_a if r.get("simulation_seed") in by_seed]
    n = len(diffs)
    if n < 2:
        return None
    mean = sum(diffs) / n
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    half = t_quantile(n - 1) * math.sqrt(var / n)
    return mean, half, n, sum(1 for d in diffs if d < 0)


def mean_of(rows, key):
    return sum(r[key] for r in rows) / len(rows) if rows else float("nan")


def main(manifest_path, latex):
    man = json.load(open(manifest_path))
    variants = man["variants"]
    order = [v["key"] for v in variants]
    label = {v["key"]: v["label"] for v in variants}
    cells = {}
    for num_run, info in man["runs"].items():
        cells[(info["w_3"], info["policy"])] = (int(num_run), info)

    w3_values = man["w3_values"]

    print("Blocking-aware selection, eligibility + l2 capacity, %d request streams"
          % len(man["simulation_seeds"]))
    print()
    header = ("%-32s %6s %9s %9s %11s %11s %7s"
              % ("policy", "w_3", "C", "w2*W", "vs own OL", "vs det OL", "wins"))
    print(header)
    print("-" * len(header))

    for w3 in w3_values:
        det_run = DET_BASELINE[w3]
        det_rows = load_seeds(det_run)
        if det_rows:
            print("%-32s %6s %9.0f %9.0f %11s %11s %7s"
                  % ("deterministic, open loop", w3, mean_of(det_rows, "overall_cost"),
                     mean_of(det_rows, "complete_time_cost"), "-", "-", "-"))
        # The published policy of the paper: relaxed H, sampled open loop.
        stoch_rows = load_seeds(man["source_runs"]["stoch"][str(w3)]
                                if str(w3) in man["source_runs"]["stoch"]
                                else man["source_runs"]["stoch"][w3])
        if stoch_rows:
            d = paired(stoch_rows, det_rows)
            print("%-32s %6s %9.0f %9.0f %11s %10s %7s"
                  % ("stochastic, open loop", w3, mean_of(stoch_rows, "overall_cost"),
                     mean_of(stoch_rows, "complete_time_cost"), "-",
                     "%+.1f%%" % (100 * d[0] / mean_of(det_rows, "overall_cost")) if d else "-",
                     "%d/%d" % (d[3], d[2]) if d else "-"))
        for key in order:
            entry = cells.get((w3, key))
            if entry is None:
                continue
            num_run, info = entry
            rows = load_seeds(num_run)
            if not rows:
                print("%-32s %6s   (not simulated yet, run %d)" % (label[key], w3, num_run))
                continue
            own_rows = load_seeds(info["source_run"])
            d_own = paired(rows, own_rows)
            d_det = paired(rows, det_rows)
            base_own = mean_of(own_rows, "overall_cost") if own_rows else float("nan")
            base_det = mean_of(det_rows, "overall_cost") if det_rows else float("nan")
            print("%-32s %6s %9.0f %9.0f %10s %10s %7s"
                  % (label[key], w3,
                     mean_of(rows, "overall_cost"), mean_of(rows, "complete_time_cost"),
                     "%+.1f%%" % (100 * d_own[0] / base_own) if d_own else "-",
                     "%+.1f%%" % (100 * d_det[0] / base_det) if d_det else "-",
                     "%d/%d" % (d_det[3], d_det[2]) if d_det else "-"))
        print()

    # The contribution of the relaxed placement once BOTH policies get the same
    # runtime recourse.
    print("Same runtime rule on both placements (least loaded over all placed):")
    print("%-6s %10s %10s %12s %10s %7s" % ("w_3", "det+LL", "stoch+LL", "paired diff", "rel", "wins"))
    for w3 in w3_values:
        a, b = cells.get((w3, "stoch_ll")), cells.get((w3, "det_ll"))
        if not (a and b):
            continue
        ra, rb = load_seeds(a[0]), load_seeds(b[0])
        d = paired(ra, rb)
        if not d:
            continue
        print("%-6s %10.0f %10.0f %7.0f+-%-4.0f %9.1f%% %6s"
              % (w3, mean_of(rb, "overall_cost"), mean_of(ra, "overall_cost"),
                 d[0], d[1], 100 * d[0] / mean_of(rb, "overall_cost"),
                 "%d/%d" % (d[3], d[2])))

    if latex:
        print()
        print("% ---- tab:policy ----")
        for w3 in w3_values:
            det_rows = load_seeds(DET_BASELINE[w3])
            parts = []
            for key in order:
                entry = cells.get((w3, key))
                rows = load_seeds(entry[0]) if entry else None
                if not rows:
                    parts.append("--")
                    continue
                d = paired(rows, det_rows)
                parts.append("$%.0f$&$%+.1f$" % (mean_of(rows, "overall_cost"),
                                                 100 * d[0] / mean_of(det_rows, "overall_cost"))
                             if d else "--")
            print("$%s$&$%.0f$&%s\\\\" % (w3, mean_of(det_rows, "overall_cost"), "&".join(parts)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=os.path.join("settings", "policy_campaign.json"))
    parser.add_argument("--latex", action="store_true")
    args = parser.parse_args()
    main(args.manifest, args.latex)
