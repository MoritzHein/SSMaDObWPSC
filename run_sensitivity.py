# Sensitivity-analysis runner.
#
# Usage: run_sensitivity.py --num_run=<N>
#        (after placement.py --num_run=<N> and
#         generate_sensitivity_scenarios.py --num_run=<N>)
#
# Iterates the scenario folders produced by generate_sensitivity_scenarios.py,
# evaluates the (fixed) placement against each perturbed parameter realization
# by invoking selection_simulation.py with --scenario_dir, and aggregates the
# resulting costs into results/run_<N>/sensitivity/summary.{json,csv}. The
# baseline run (results/run_<N>/results.json) is left untouched and shown for
# comparison.
import argparse
import csv
import glob
import json
import os
import re
import subprocess
import sys

import numpy as np

COST_KEYS = ["overall_cost", "overall_cost_comparison", "money_cost",
             "time_cost", "complete_time_cost"]


def load_json(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_run", required=False, type=int, default=None)
    args = parser.parse_args()
    num_run = args.num_run
    if num_run is None:
        while True:
            try:
                num_run = int(input("Please enter num_run: "))
                break
            except ValueError:
                print("num_run must be an integer. Please try again.")

    scenario_root = os.path.join("sensitivity", f"run_{num_run}")
    scenario_dirs = sorted(
        glob.glob(os.path.join(scenario_root, "scenario_*")),
        key=lambda p: int(re.search(r"scenario_(\d+)$", p).group(1)))
    if not scenario_dirs:
        raise SystemExit(
            f"No scenarios found under {scenario_root}. "
            f"Run generate_sensitivity_scenarios.py --num_run={num_run} first.")

    rows = []
    for sdir in scenario_dirs:
        print(f"Running {sdir} ...")
        subprocess.run(
            [sys.executable, "selection_simulation.py",
             f"--num_run={num_run}", f"--scenario_dir={sdir}"],
            check=True)
        results = load_json(os.path.join(sdir, "results", "results.json"))
        manifest = load_json(os.path.join(sdir, "scenario_manifest.json"))
        row = {"scenario": os.path.basename(sdir)}
        for k in COST_KEYS:
            row[k] = results.get(k)
        # Flatten the per-parameter factor summaries from the manifest.
        for param, info in manifest.items():
            if isinstance(info, dict) and "factor_mean" in info:
                row[f"{param}_mode"] = info.get("mode")
                row[f"{param}_factor_mean"] = info.get("factor_mean")
                row[f"{param}_factor_min"] = info.get("factor_min")
                row[f"{param}_factor_max"] = info.get("factor_max")
        rows.append(row)

    baseline = load_json(os.path.join("results", f"run_{num_run}", "results.json"))

    out_dir = os.path.join("results", f"run_{num_run}", "sensitivity")
    os.makedirs(out_dir, exist_ok=True)

    summary = {"num_run": num_run, "baseline": {k: baseline.get(k) for k in COST_KEYS},
               "scenarios": rows}
    # Distribution of overall_cost across scenarios vs. the baseline.
    costs = [r["overall_cost"] for r in rows if isinstance(r.get("overall_cost"), (int, float))]
    if costs:
        summary["overall_cost_stats"] = {
            "baseline": baseline.get("overall_cost"),
            "min": float(np.min(costs)),
            "mean": float(np.mean(costs)),
            "max": float(np.max(costs)),
            "std": float(np.std(costs)),
        }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    # Flat CSV: union of all row keys.
    fieldnames = []
    for r in rows:
        for k in r:
            if k not in fieldnames:
                fieldnames.append(k)
    with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"\nWrote {os.path.join(out_dir, 'summary.json')} and summary.csv")
    if costs:
        s = summary["overall_cost_stats"]
        print(f"overall_cost  baseline={s['baseline']}  "
              f"min={s['min']:.4g}  mean={s['mean']:.4g}  max={s['max']:.4g}  std={s['std']:.4g}")


if __name__ == "__main__":
    main()
