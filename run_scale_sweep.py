# One-factor-at-a-time (OFAT) scale-sweep runner for the sensitivity study.
#
# Usage: run_scale_sweep.py --num_run=<N>
#        (after placement.py --num_run=<N>)
#
# Complements run_sensitivity.py (which evaluates random combined scenarios).
# Here each uncertain parameter is varied ALONE across a set of deterministic
# scale factors while the others stay at their baseline, so the effect of every
# individual uncertainty can be attributed cleanly. The script generates its own
# scenarios (reusing the perturbation functions of
# generate_sensitivity_scenarios.py) under sensitivity/run_<N>/scale_sweep/,
# evaluates the fixed placement against each via selection_simulation.py, and
# aggregates the costs into results/run_<N>/sensitivity/scale_sweep_summary.json
# (consumed by visualization_sensitivity.py).
#
# With --adapt, every sweep point additionally gets an ADAPTED baseline: the
# placement (and the centralized/heuristic comparison, per the run's settings)
# is re-solved with the parameters scaled to the perturbed values, so the cost
# of keeping the stale nominal placement can be separated from the cost of the
# perturbation itself. Each adapted run lives in a self-contained sandbox
# sensitivity/run_<N>/scale_sweep/<param>/factor_<tag>/adapted/ and its costs
# are added to the summary as adapted_overall_cost(_comparison). Note the cost:
# one full placement solve per sweep point (a centralized comparison solve is
# a full-architecture MIQP each time), and one full run folder of storage.
#
# With --resume, sweep points whose results.json already exists are loaded from
# disk instead of re-solved, so an interrupted sweep (e.g. a solver crash) can
# be continued without redoing the finished points.
import argparse
import json
import os
import shutil
import subprocess
import sys

import numpy as np

from generate_sensitivity_scenarios import perturb_cloud, perturb_workflows

# The five uncertain parameters; request_rate/latency live in the cloud files,
# time/speedup/ram in the workflow files.
ALL_PARAMS = ["request_rate", "time", "speedup", "latency", "ram"]
DEFAULT_FACTORS = [0.75, 0.875, 1.0, 1.125,  1.25]


def load_json(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def completed_results(path):
    # A results.json is only written once its producing script finished, so a
    # loadable file with a real overall_cost marks the sweep point as done.
    res = load_json(path)
    return res if res.get("overall_cost") is not None else None


def factor_tag(f):
    # Filesystem-friendly tag, e.g. 1.25 -> "1p25".
    return ("%g" % f).replace(".", "p").replace("-", "m")


def run_adapted(num_run, param, factor, point_dir):
    """Adapted baseline for one sweep point: re-solve the placement with the
    pristine inputs scaled by `factor` (placement.py --scale_param/--scale_factor)
    and simulate it. Runs inside a sandbox <point_dir>/adapted/ that mirrors the
    standard {settings,cloud,workflows,placement,results}/run_<N> layout, so
    both scripts run unmodified with cwd=<sandbox> (their paths are cwd-relative;
    imports resolve via the script's own directory). Returns the sandbox
    results.json contents."""
    sandbox = os.path.join(point_dir, "adapted")
    for d in ("settings", "cloud", "workflows", "placement", "results"):
        os.makedirs(os.path.join(sandbox, d), exist_ok=True)
    shutil.copyfile(os.path.join("settings", f"parameters_{num_run}.json"),
                    os.path.join(sandbox, "settings", f"parameters_{num_run}.json"))
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    subprocess.run(
        [sys.executable, os.path.join(repo_dir, "placement.py"),
         f"--num_run={num_run}", f"--scale_param={param}", f"--scale_factor={factor}"],
        cwd=sandbox, check=True)
    subprocess.run(
        [sys.executable, os.path.join(repo_dir, "selection_simulation.py"),
         f"--num_run={num_run}"],
        cwd=sandbox, check=True)
    return load_json(os.path.join(sandbox, "results", f"run_{num_run}", "results.json"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_run", required=False, type=int, default=None)
    parser.add_argument("--factors", required=False, type=str, default=None,
                        help="Comma-separated scale factors (default 0.5,0.75,1.0,1.25,1.5,2.0).")
    parser.add_argument("--params", required=False, type=str, default=None,
                        help="Comma-separated parameters to sweep (default all five).")
    parser.add_argument("--adapt", action="store_true",
                        help="Additionally re-solve the placement (and the comparison, "
                             "per the run's settings) for every sweep point with the "
                             "parameters adapted to the perturbed values, as a "
                             "re-planning baseline. Stored under "
                             "sensitivity/run_<N>/scale_sweep/<param>/factor_<tag>/adapted/.")
    parser.add_argument("--resume", action="store_true",
                        help="Skip sweep points whose results.json already exists "
                             "(useful to continue after a crash; solved points are "
                             "loaded from disk instead of re-solved).")
    args = parser.parse_args()
    num_run = args.num_run
    if num_run is None:
        while True:
            try:
                num_run = int(input("Please enter num_run: "))
                break
            except ValueError:
                print("num_run must be an integer. Please try again.")

    factors = ([float(x) for x in args.factors.split(",")] if args.factors
               else list(DEFAULT_FACTORS))
    params = ([p.strip() for p in args.params.split(",")] if args.params
              else list(ALL_PARAMS))
    for p in params:
        if p not in ALL_PARAMS:
            raise SystemExit(f"Unknown parameter '{p}'. Choose from {ALL_PARAMS}.")

    cloud_dir = f"cloud/run_{num_run}"
    workflow_dir = f"workflows/run_{num_run}"

    # Baseline: evaluate the unperturbed placement once if not already present.
    baseline_path = os.path.join("results", f"run_{num_run}", "results.json")
    if not os.path.exists(baseline_path):
        print("Baseline results missing; running selection_simulation.py once ...")
        subprocess.run([sys.executable, "selection_simulation.py", f"--num_run={num_run}"],
                       check=True)
    baseline = load_json(baseline_path)

    # rng is unused for scale mode (draw_factor is deterministic) but the
    # perturbation functions expect one.
    rng = np.random.default_rng(0)

    sweep = {}
    for param in params:
        print(f"\n=== Sweeping {param} ===")
        per_param = {"factors": [], "overall_cost": [], "overall_cost_comparison": []}
        if args.adapt:
            per_param["adapted_overall_cost"] = []
            per_param["adapted_overall_cost_comparison"] = []
        for f in factors:
            sens = {p: {"mode": "off"} for p in ALL_PARAMS}
            sens[param] = {"mode": "scale", "factor": f, "per_element": False}
            out_dir = os.path.join("sensitivity", f"run_{num_run}", "scale_sweep",
                                    param, f"factor_{factor_tag(f)}")
            os.makedirs(out_dir, exist_ok=True)
            res = (completed_results(os.path.join(out_dir, "results", "results.json"))
                   if args.resume else None)
            if res is not None:
                print(f"  {param} x{f} already solved, skipping")
            else:
                manifest = {"num_run": num_run, "param": param, "factor": f, "config": sens}
                # Only the matching family of files is written: perturb_cloud no-ops
                # for workflow parameters and perturb_workflows no-ops for cloud ones.
                perturb_cloud(rng, sens, cloud_dir, out_dir, manifest)
                perturb_workflows(rng, sens, workflow_dir, out_dir, manifest)
                with open(os.path.join(out_dir, "scenario_manifest.json"), "w") as fh:
                    json.dump(manifest, fh, indent=4)

                print(f"  {param} x{f} ...")
                subprocess.run(
                    [sys.executable, "selection_simulation.py",
                     f"--num_run={num_run}", f"--scenario_dir={out_dir}"],
                    check=True)
                res = load_json(os.path.join(out_dir, "results", "results.json"))
            per_param["factors"].append(f)
            per_param["overall_cost"].append(res.get("overall_cost"))
            per_param["overall_cost_comparison"].append(res.get("overall_cost_comparison"))

            if args.adapt:
                adapted = (completed_results(os.path.join(
                    out_dir, "adapted", "results", f"run_{num_run}", "results.json"))
                    if args.resume else None)
                if adapted is not None:
                    print(f"  {param} x{f} adapted placement already solved, skipping")
                else:
                    print(f"  {param} x{f} adapted placement ...")
                    adapted = run_adapted(num_run, param, f, out_dir)
                per_param["adapted_overall_cost"].append(adapted.get("overall_cost"))
                per_param["adapted_overall_cost_comparison"].append(
                    adapted.get("overall_cost_comparison"))
        sweep[param] = per_param

    out_dir = os.path.join("results", f"run_{num_run}", "sensitivity")
    os.makedirs(out_dir, exist_ok=True)
    summary = {
        "num_run": num_run,
        "factors": factors,
        "baseline": {
            "overall_cost": baseline.get("overall_cost"),
            "overall_cost_comparison": baseline.get("overall_cost_comparison"),
            "money_cost": baseline.get("money_cost"),
            "time_cost": baseline.get("time_cost"),
            "complete_time_cost": baseline.get("complete_time_cost"),
        },
        "params": sweep,
    }
    if args.adapt:
        summary["adapt"] = True
    out_path = os.path.join(out_dir, "scale_sweep_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"\nWrote {out_path}")
    base_c = baseline.get("overall_cost")
    print(f"baseline overall_cost = {base_c}")
    for param in params:
        costs = [c for c in sweep[param]["overall_cost"] if isinstance(c, (int, float))]
        if costs:
            print(f"  {param:13s} overall_cost range [{min(costs):.4g}, {max(costs):.4g}]")
        acosts = [c for c in sweep[param].get("adapted_overall_cost", [])
                  if isinstance(c, (int, float))]
        if acosts:
            print(f"  {param:13s} adapted overall_cost range [{min(acosts):.4g}, {max(acosts):.4g}]")


if __name__ == "__main__":
    main()
