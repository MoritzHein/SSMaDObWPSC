# Re-simulate an existing placement on several independent request streams.
#
# Usage: resimulate_seeds.py [--seeds=10] [--manifest=settings/ram_w3_campaign.json]
#                            [--runs=300,301,...]
#
# A single simulation at simulation_time=200 yields only a few hundred requests,
# so its cost carries roughly 10% Monte-Carlo noise - larger than most of the
# effects the RAM/w_3 study is trying to resolve. This script therefore re-runs
# ONLY selection_simulation.py, once per seed in the manifest's
# "simulation_seeds" list, and records the cost components of each stream in
#
#     results/run_<N>/results_seeds.json
#
# and the per-request response/waiting times of the same streams in
#
#     results/run_<N>/latency_seeds.json
#
# which visualization_ram_w3.py turns into means with 95% confidence intervals
# and into the response-time distribution figure.
# placement.py is never invoked, so the placement files are untouched and the
# comparison across configurations stays paired on identical request streams.
#
# The parameter file's "simulation_seed" is restored to the manifest's baseline
# value afterwards, so results/run_<N>/results.json is left consistent with it.
#
# One simulation of the 2-level, 10-workflow architecture takes about half a
# minute, so the full 24-run campaign is a good two hours sequentially. Runs
# touch only their own parameter and result files, so --runs splits the work
# across as many processes as there are cores:
#
#     for grp in 300,301 302,303 ...; do
#         python resimulate_seeds.py --seeds=10 --runs=$grp &
#     done; wait
import argparse
import json
import os
import subprocess
import sys

DEFAULT_MANIFEST = os.path.join("settings", "ram_w3_campaign.json")


def load_manifest(path):
    with open(path, "r") as f:
        return json.load(f)


def latency_of(num_run):
    """Per-request response times and waiting times of the run just simulated.

    Reads the own-method trace written by selection_simulation.py. Only the three
    scalars per request are kept - the full trace is ~130 kB and would be 30 MB
    over the whole campaign, while these lists are a few kB and carry everything
    the response-time and waiting-over-time figures need. The arrival time is
    what lets visualization_waiting_over_time.py bucket waiting by elapsed time
    across all request streams instead of a single noisy one.
    """
    path = os.path.join("results", f"run_{num_run}",
                        f"detailed_simulation_run_{num_run}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        logs = json.load(f).get("request_logs", [])
    return {
        "arrivals": [int(r["arrival"]) for r in logs],
        "response_times": [int(r["end"]) - int(r["arrival"]) for r in logs],
        "waits": [int(r["wait"]) for r in logs],
    }


def simulate(num_run, seed, python_exe):
    """Run selection_simulation.py for num_run at the given simulation seed."""
    params_path = os.path.join("settings", f"parameters_{num_run}.json")
    with open(params_path, "r") as f:
        cfg = json.load(f)
    cfg["simulation_seed"] = seed
    with open(params_path, "w") as f:
        json.dump(cfg, f, indent=4)
    proc = subprocess.run([python_exe, "selection_simulation.py", f"--num_run={num_run}"],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"run {num_run} seed {seed} failed:\n"
                           f"{(proc.stderr or '')[-1500:]}")
    with open(os.path.join("results", f"run_{num_run}", "results.json"), "r") as f:
        res = json.load(f)
    row = {
        "simulation_seed": seed,
        "overall_cost": res["overall_cost"],
        "money_cost": res["money_cost"],
        "time_cost": res["time_cost"],
        "complete_time_cost": res["complete_time_cost"],
        "overall_cost_comparison": res.get("overall_cost_comparison"),
    }
    lat = latency_of(num_run)
    if lat is not None:
        row["latency"] = lat
    return row


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--seeds", type=int, default=None,
                        help="use only the first N seeds of the manifest list")
    parser.add_argument("--runs", default=None,
                        help="comma-separated subset of run numbers (default: all)")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    seeds = manifest["simulation_seeds"]
    if args.seeds is not None:
        seeds = seeds[:args.seeds]
    base_seed = manifest["simulation_seeds"][0]

    runs = sorted(int(k) for k in manifest["runs"])
    if args.runs:
        wanted = {int(x) for x in args.runs.split(",")}
        runs = [r for r in runs if r in wanted]

    python_exe = sys.executable
    print(f"re-simulating {len(runs)} runs x {len(seeds)} request streams "
          f"(placements are NOT re-solved)")

    for num_run in runs:
        out_dir = os.path.join("results", f"run_{num_run}")
        if not os.path.exists(os.path.join(out_dir, "results.json")):
            print(f"  run {num_run}: no results.json - run placement.py and "
                  f"selection_simulation.py first, skipping")
            continue
        rows = [simulate(num_run, s, python_exe) for s in seeds]
        # leave the parameter file (and results.json) at the baseline stream
        simulate(num_run, base_seed, python_exe)
        # The per-request response times go into their own file so that
        # results_seeds.json keeps the compact shape the cost figures read.
        latency = [{"simulation_seed": r["simulation_seed"], **r.pop("latency")}
                   for r in rows if "latency" in r]
        with open(os.path.join(out_dir, "results_seeds.json"), "w") as f:
            json.dump(rows, f, indent=4)
        if latency:
            with open(os.path.join(out_dir, "latency_seeds.json"), "w") as f:
                json.dump(latency, f)
        costs = [r["overall_cost"] for r in rows]
        # The per-run entries differ between campaigns (the RAM x w_3 study keys
        # them by ram/w_3/policy, the scalability study by num_levels/policy),
        # so describe a run by whatever its manifest happens to record instead
        # of by a fixed set of keys.
        info = manifest["runs"][str(num_run)]
        descr = " ".join(f"{k}={v}" for k, v in info.items()
                         if not isinstance(v, (dict, list)))
        print(f"  run {num_run} {descr}: {len(rows)} streams, overall "
              f"{sum(costs)/len(costs):9.1f} ({min(costs):.1f} - {max(costs):.1f})")
    print("done - visualize with visualization_ram_w3.py")
