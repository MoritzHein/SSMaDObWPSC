# Re-simulate an existing placement on several independent request streams.
#
# Usage: resimulate_seeds.py [--seeds=10] [--manifest=settings/ram_w3_campaign.json]
#                            [--runs=300,301,...]
#
# A single simulation at simulation_time=200 yields only a few hundred requests,
# so its cost carries roughly 10% Monte-Carlo noise - larger than most of the
# effects the RAM/w_3 study is trying to resolve. The placement, however, is the
# expensive part: solving it takes seconds, simulating it takes ~3 s. This script
# therefore re-runs ONLY selection_simulation.py, once per seed in the manifest's
# "simulation_seeds" list, and records the cost components of each stream in
#
#     results/run_<N>/results_seeds.json
#
# which visualization_ram_w3.py turns into means with 95% confidence intervals.
# placement.py is never invoked, so the placement files are untouched and the
# comparison across configurations stays paired on identical request streams.
#
# The parameter file's "simulation_seed" is restored to the manifest's baseline
# value afterwards, so results/run_<N>/results.json is left consistent with it.
import argparse
import json
import os
import subprocess
import sys

DEFAULT_MANIFEST = os.path.join("settings", "ram_w3_campaign.json")


def load_manifest(path):
    with open(path, "r") as f:
        return json.load(f)


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
    return {
        "simulation_seed": seed,
        "overall_cost": res["overall_cost"],
        "money_cost": res["money_cost"],
        "time_cost": res["time_cost"],
        "complete_time_cost": res["complete_time_cost"],
        "overall_cost_comparison": res.get("overall_cost_comparison"),
    }


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
        with open(os.path.join(out_dir, "results_seeds.json"), "w") as f:
            json.dump(rows, f, indent=4)
        costs = [r["overall_cost"] for r in rows]
        info = manifest["runs"][str(num_run)]
        print(f"  run {num_run} ram={info['ram']:<3s} w_3={info['w_3']:<4} "
              f"{info['policy']:<5s}: {len(rows)} streams, overall "
              f"{sum(costs)/len(costs):9.1f} ({min(costs):.1f} - {max(costs):.1f})")
    print("done - visualize with visualization_ram_w3.py")
