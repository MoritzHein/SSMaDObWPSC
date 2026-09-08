# Turn the Azure Functions Trace 2019 into the per-minute arrival shapes that
# selection_simulation.py's "arrival_trace" flag reads.
#
# Usage:
#   build_arrival_trace.py --data_dir=<dir> [--day=1] [--n_apps=32]
#                          [--window_minutes=30] [--min_daily=10000]
#                          [--out=settings/arrival_trace_azure_d01.json]
#   build_arrival_trace.py --data_dir=<dir> --measure_only
#
# The dataset (Shahrad et al., "Serverless in the Wild", USENIX ATC 2020,
# CC-BY) ships invocations_per_function_md.anon.d[01-14].csv: one row per
# function with HashOwner, HashApp, HashFunction, Trigger and then 1440 columns
# holding the invocation COUNT in each minute of the day.
#
# Two facts drive everything here.
#
#   1. There are no arrival timestamps. The data is a count per minute; nothing
#      below 60 s exists in it, so sub-minute structure is a modelling choice we
#      make, never a trace we replay. The paper must say so.
#   2. Burstiness is a function of volume, and it was measured here rather than
#      assumed. Over the 6645 http applications of day 1 (--measure_only):
#
#          decile   mean inv/min    CV/Poisson   ACF(1)
#             1          0.001          1.01      0.003
#             5          0.026          1.73      0.166
#             8          0.529          2.92      0.184
#            10        201.957          7.37      0.493
#
#      The bottom deciles are pure shot noise - CV/Poisson 1.01 at 0.001
#      invocations/minute - so a rarely-invoked function carries no information
#      the simulator's own rng.poisson does not already produce. But this
#      architecture runs at ~0.26 invocations/minute per (workflow, leaf)
#      stream, which lands in deciles 7-8, where real applications are already
#      2-3x over-dispersed relative to Poisson. The shot-noise regime is two
#      decades BELOW our operating point, not at it.
#
# So the level mismatch is not what forces the design; the strength of the
# temporal structure is. Only the top decile shows autocorrelation worth the
# name (ACF(1) 0.49 against 0.18 at our own rate), and that is the regime the
# review asks about - "where blocking actually bites". This script therefore
# emits either band and the study runs both:
#
#   --min_daily 100000            high-volume shape, the bursty stress arm
#   --min_daily 200 --max_daily 800   applications at OUR OWN rate, the control
#                                     that shows the transplant is not doing
#                                     the work
#
# Either way the trace supplies the temporal SHAPE and lambda keeps the LEVEL.
# Run --measure_only first: if the selected band is not meaningfully more
# variable than Poisson, the arm collapses to a robustness statement and must
# be reported as one.
#
# We deliberately take invocation counts ONLY. The dataset's own durations
# (milliseconds to a few seconds, against this paper's 10-30 s functions) and
# its per-APPLICATION allocated memory would drop utilisation by orders of
# magnitude and drive every effect the paper measures to zero.
import argparse
import json
import os

import numpy as np
import pandas as pd

MINUTES_PER_DAY = 1440
META_COLS = ["HashOwner", "HashApp", "HashFunction", "Trigger"]


def load_day(data_dir, day):
    """Per-application per-minute invocation counts of one day.

    Trigger is recorded per function, so http filtering happens before the
    aggregation to applications; an app that mixes trigger types contributes
    only its http-triggered functions.
    """
    path = os.path.join(data_dir, "invocations_per_function_md.anon.d%02d.csv" % day)
    if not os.path.exists(path):
        raise SystemExit("missing %s - extract it from the release tarball" % path)
    df = pd.read_csv(path)
    minute_cols = [c for c in df.columns if c not in META_COLS]
    if len(minute_cols) != MINUTES_PER_DAY:
        raise SystemExit("expected %d minute columns, found %d"
                         % (MINUTES_PER_DAY, len(minute_cols)))
    http = df[df["Trigger"] == "http"]
    counts = http.groupby("HashApp")[minute_cols].sum()
    return counts


def series_stats(counts_row):
    """Coefficient of variation and lag-1 autocorrelation of a minute series.

    A Poisson process of mean m has CV = 1/sqrt(m) and zero autocorrelation, so
    these two numbers are exactly what separates a real workload pattern from
    the shot noise of a rarely-invoked function.
    """
    x = np.asarray(counts_row, dtype=float)
    mean = x.mean()
    if mean <= 0:
        return 0.0, 0.0, 0.0
    cv = x.std() / mean
    xc = x - mean
    denom = float((xc * xc).sum())
    acf1 = float((xc[:-1] * xc[1:]).sum() / denom) if denom > 0 else 0.0
    # Ratio against the CV a Poisson process of the same mean would show. > 1
    # means genuine over-dispersion; ~1 means the variation is pure shot noise.
    return cv, acf1, cv * np.sqrt(mean)


def measure(counts):
    """Print the burstiness evidence by volume decile. Writes nothing."""
    totals = counts.sum(axis=1)
    print("http applications: %d, daily invocations %d - %d (median %d)"
          % (len(totals), totals.min(), totals.max(), int(totals.median())))
    print("\nthis architecture runs at ~0.26 invocations/minute per stream, "
          "i.e. ~%d per day" % int(0.26 * MINUTES_PER_DAY))
    order = totals.sort_values()
    print("\n decile   apps   mean inv/min      CV   CV/Poisson    ACF(1)")
    n = len(order)
    for d in range(10):
        lo, hi = int(n * d / 10), int(n * (d + 1) / 10)
        apps = order.index[lo:hi]
        if len(apps) == 0:
            continue
        rows = counts.loc[apps].to_numpy(dtype=float)
        stats = [series_stats(r) for r in rows]
        mpm = rows.sum(axis=1).mean() / MINUTES_PER_DAY
        print("   %2d   %5d   %12.3f  %6.2f   %10.2f  %8.3f"
              % (d + 1, len(apps), mpm,
                 float(np.mean([s[0] for s in stats])),
                 float(np.mean([s[2] for s in stats])),
                 float(np.mean([s[1] for s in stats]))))
    print("\nCV/Poisson ~ 1 means the series is shot noise and carries no "
          "burstiness beyond what rng.poisson already produces.")


def select_apps(counts, n_apps, min_daily, max_daily=None, min_cv=0.0):
    """The n highest-volume http applications inside a daily-volume band.

    max_daily is what makes the matched-rate control possible: bounding the
    band around this architecture's own ~374 invocations/day per stream selects
    applications that really run at our load level, so the comparison cannot be
    dismissed as the shape of a 200 req/min service transplanted onto a 0.26
    req/min one.

    min_cv is the ABSOLUTE coefficient of variation, and which statistic to
    filter on depends on the band. cv_over_poisson answers "is this more than
    shot noise", the right question at ~0.37 invocations/minute. It is
    misleading at high volume: an application at 63599 invocations/minute has a
    Poisson CV of 0.004, so cv_over_poisson=12 while the intensity actually
    swings by only 5% and its unit-mean shape is nearly flat. Since the shape is
    normalised and multiplies lambda, the absolute CV is what decides how much
    the arrival rate really moves, so the high-volume stress arm filters on that.
    Requiring a bursty workload to be bursty is an inclusion criterion, not
    outcome selection - the outcome here is simulated cost - but state it.
    """
    totals = counts.sum(axis=1)
    keep = counts.loc[totals >= min_daily]
    if max_daily is not None:
        keep = keep.loc[keep.sum(axis=1) <= max_daily]
    if len(keep) == 0:
        raise SystemExit("no application in the band min_daily=%s max_daily=%s"
                         % (min_daily, max_daily))
    rows = []
    for app, row in keep.iterrows():
        cv, acf1, over = series_stats(row)
        # A constant series carries no burstiness and would silently turn the
        # trace arm back into the stationary Poisson control.
        if cv <= 0 or cv < min_cv:
            continue
        rows.append({"hash_app": app, "daily_total": int(row.sum()),
                     "cv": round(float(cv), 4), "acf1": round(float(acf1), 4),
                     "cv_over_poisson": round(float(over), 4),
                     "counts": [int(v) for v in row.to_numpy()]})
    rows.sort(key=lambda r: -r["daily_total"])
    return rows[:n_apps]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True,
                        help="directory holding invocations_per_function_md.anon.dNN.csv")
    parser.add_argument("--day", type=int, default=1)
    parser.add_argument("--n_apps", type=int, default=32)
    parser.add_argument("--window_minutes", type=int, default=30)
    parser.add_argument("--min_daily", type=int, default=10000,
                        help="minimum daily invocations; chosen from --measure_only")
    parser.add_argument("--min_cv", type=float, default=0.0,
                        help="absolute coefficient-of-variation floor on the "
                             "per-minute series; use it for the high-volume arm")
    parser.add_argument("--max_daily", type=int, default=None,
                        help="upper bound of the band; set it to select applications "
                             "running at this architecture's own rate (~374/day)")
    parser.add_argument("--out", default=None)
    parser.add_argument("--measure_only", action="store_true",
                        help="print the burstiness evidence and exit")
    args = parser.parse_args()

    counts = load_day(args.data_dir, args.day)
    if args.measure_only:
        measure(counts)
        raise SystemExit(0)

    apps = select_apps(counts, args.n_apps, args.min_daily, args.max_daily,
                       args.min_cv)
    window = args.window_minutes
    # Disjoint windows over the day. The simulator picks one per seed, so the
    # ten request streams become ten different real half-hours of a production
    # workload instead of ten draws from one synthetic distribution.
    offsets = list(range(0, MINUTES_PER_DAY - window + 1, window))
    out = args.out or os.path.join("settings",
                                   "arrival_trace_azure_d%02d.json" % args.day)
    payload = {
        "source": "Azure Functions Trace 2019 (Shahrad et al., USENIX ATC 2020), CC-BY",
        "day": args.day,
        "minutes_per_day": MINUTES_PER_DAY,
        "window_minutes": window,
        "selection": {"trigger": "http", "min_daily_invocations": args.min_daily,
                      "max_daily_invocations": args.max_daily,
                      "min_cv": args.min_cv,
                      "n_apps": len(apps), "ranked_by": "daily_total"},
        "offsets": offsets,
        # Raw counts, not a pre-normalised shape: the simulator normalises per
        # window or per day depending on the flag, and both modes read one file.
        "apps": apps,
    }
    with open(out, "w") as f:
        json.dump(payload, f)
    print("wrote %s" % out)
    print("  %d applications, %d disjoint %d-minute windows" % (len(apps), len(offsets), window))
    print("  daily totals %d - %d" % (apps[-1]["daily_total"], apps[0]["daily_total"]))
    print("  CV %.2f - %.2f, CV/Poisson %.2f - %.2f, ACF(1) %.2f - %.2f"
          % (min(a["cv"] for a in apps), max(a["cv"] for a in apps),
             min(a["cv_over_poisson"] for a in apps), max(a["cv_over_poisson"] for a in apps),
             min(a["acf1"] for a in apps), max(a["acf1"] for a in apps)))
