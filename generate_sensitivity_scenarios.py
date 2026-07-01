# Sensitivity-analysis scenario generator.
#
# Usage: generate_sensitivity_scenarios.py --num_run=<N>
#        (after placement.py --num_run=<N>)
#
# The placement computed by placement.py is optimized against the *assumed*
# parameter values realized in cloud/run_<N>/ and workflows/run_<N>/. This
# script keeps that placement decision (P, H) fixed and materializes a set of
# scenarios in which the parameters that are hard to know in advance deviate
# from those assumptions:
#
#   - demand        request_rate  (cloud/run_<N>/requirements_cloud_full.json)
#   - running time  time          (workflows/run_<N>/requirements_workflow_*.json)
#   - speedup       speedup       (same workflow files)
#   - RAM demand    ram           (same workflow files)
#   - latency       L matrix      (cloud/run_<N>/requirements_cloud_*.json; the
#                                  solved L is written there by placement.py, so
#                                  the placement solution files stay immutable)
#
# Each parameter is configured independently via the "sensitivity_config" block
# of settings/parameters_<N>.json. Per scenario, only the perturbed input files
# are written into sensitivity/run_<N>/scenario_<i>/; selection_simulation.py
# (run with --scenario_dir) reads those first and falls back to the baseline
# files for everything else.
import argparse
import glob
import json
import os
import re

import numpy as np

# Parameters that live in the workflow function configs.
WORKFLOW_PARAMS = ["time", "speedup", "ram"]


def draw_factor(rng, cfg):
    """One multiplicative perturbation factor for the given parameter config.

    mode "scale" => deterministic x factor (for one-factor-at-a-time sweeps);
    mode "noise" => x (1 + U(-noise_pct, +noise_pct)); anything else => 1.0."""
    mode = cfg.get("mode", "off")
    if mode == "scale":
        return float(cfg.get("factor", 1.0))
    if mode == "noise":
        p = float(cfg.get("noise_pct", 0.0))
        return 1.0 + float(rng.uniform(-p, p))
    return 1.0


def enabled(cfg):
    return cfg.get("mode", "off") != "off"


def perturb_cloud(rng, sens, cloud_dir, out_dir, manifest):
    """Perturb the node parameters that live in the cloud requirement files:
    request_rate (demand, full cloud file only) and the latency matrix L
    (full cloud file + every region cloud file). Both are handled in one pass so
    the full cloud file carries both perturbations without one clobbering the
    other. Latency L is written into the cloud configs by placement.py."""
    cfg_demand = sens.get("request_rate", {})
    cfg_lat = sens.get("latency", {})
    do_demand = enabled(cfg_demand)
    do_lat = enabled(cfg_lat)
    if not (do_demand or do_lat):
        return

    dst_dir = os.path.join(out_dir, "cloud")
    os.makedirs(dst_dir, exist_ok=True)

    # Latency: per-element => independent draw per matrix entry; otherwise a
    # single factor shared across every entry of every cloud file (consistent).
    lat_per_element = bool(cfg_lat.get("per_element", False))
    lat_shared = None if (lat_per_element or not do_lat) else draw_factor(rng, cfg_lat)
    lat_applied = []

    def scale_L(L):
        if not isinstance(L, list):
            return
        for row in L:
            if not isinstance(row, list):
                continue
            for j in range(len(row)):
                fac = draw_factor(rng, cfg_lat) if lat_per_element else lat_shared
                row[j] = row[j] * fac
                lat_applied.append(fac)

    # Full cloud file: demand and/or latency.
    with open(os.path.join(cloud_dir, "requirements_cloud_full.json"), "r") as f:
        full = json.load(f)
    if do_demand:
        providers = full["providers"]
        keys = list(providers.keys())
        if bool(cfg_demand.get("per_element", False)):
            dfac = {k: draw_factor(rng, cfg_demand) for k in keys}
        else:
            f0 = draw_factor(rng, cfg_demand)
            dfac = {k: f0 for k in keys}
        for k in keys:
            providers[k]["request_rate"] *= dfac[k]
        dvals = list(dfac.values())
        manifest["request_rate"] = {
            "mode": cfg_demand.get("mode"),
            "per_element": bool(cfg_demand.get("per_element", False)),
            "factor_min": min(dvals),
            "factor_mean": float(np.mean(dvals)),
            "factor_max": max(dvals),
        }
    if do_lat and "L" in full:
        scale_L(full["L"])
    with open(os.path.join(dst_dir, "requirements_cloud_full.json"), "w") as f:
        json.dump(full, f, indent=4)

    # Region cloud files: latency only.
    if do_lat:
        region_re = re.compile(r"requirements_cloud_\d+_\d+\.json$")
        for src in sorted(glob.glob(os.path.join(cloud_dir, "requirements_cloud_*_*.json"))):
            if not region_re.search(os.path.basename(src)):
                continue
            with open(src, "r") as f:
                conf = json.load(f)
            if "L" in conf:
                scale_L(conf["L"])
            with open(os.path.join(dst_dir, os.path.basename(src)), "w") as f:
                json.dump(conf, f, indent=4)

    if do_lat:
        manifest["latency"] = {
            "mode": cfg_lat.get("mode"),
            "per_element": lat_per_element,
            "factor_min": min(lat_applied) if lat_applied else None,
            "factor_mean": float(np.mean(lat_applied)) if lat_applied else None,
            "factor_max": max(lat_applied) if lat_applied else None,
        }


def perturb_workflows(rng, sens, workflow_dir, out_dir, manifest):
    """Scale time/speedup/ram in the workflow function configs.

    A factor is drawn per (workflow, function) and applied identically to every
    copy of that workflow (the level-0 file used by the event simulation and the
    subproblem copies used by the analytical cost loop), so both evaluation
    paths see a consistent perturbation."""
    active = [p for p in WORKFLOW_PARAMS if enabled(sens.get(p, {}))]
    if not active:
        return
    wf_files = sorted(glob.glob(os.path.join(workflow_dir, "requirements_workflow_*.json")))
    name_re = re.compile(r"requirements_workflow_(\d+)_(\d+)\.json$")

    # Workflow indices present, and how many functions each has (from its
    # level-0 file).
    workflows = sorted({int(name_re.search(os.path.basename(p)).group(1))
                        for p in wf_files if name_re.search(os.path.basename(p))})
    func_counts = {}
    for w in workflows:
        with open(os.path.join(workflow_dir, f"requirements_workflow_{w}_0.json"), "r") as f:
            func_counts[w] = json.load(f)["num_functions"]

    # factor_tables[param][(w, m)] -> factor
    factor_tables = {p: {} for p in active}
    for p in active:
        cfg = sens[p]
        per_element = bool(cfg.get("per_element", False))
        shared = None if per_element else draw_factor(rng, cfg)
        for w in workflows:
            for m in range(func_counts[w]):
                factor_tables[p][(w, m)] = draw_factor(rng, cfg) if per_element else shared

    dst_dir = os.path.join(out_dir, "workflows")
    os.makedirs(dst_dir, exist_ok=True)
    for src in wf_files:
        base = os.path.basename(src)
        match = name_re.search(base)
        if not match:
            continue
        w = int(match.group(1))
        with open(src, "r") as f:
            conf = json.load(f)
        for fname, fdict in conf.get("functions", {}).items():
            try:
                m = int(fname.split("_")[1])
            except (IndexError, ValueError):
                continue
            for p in active:
                key = (w, m)
                if p in fdict and key in factor_tables[p]:
                    fdict[p] = fdict[p] * factor_tables[p][key]
        with open(os.path.join(dst_dir, base), "w") as f:
            json.dump(conf, f, indent=4)

    for p in active:
        vals = list(factor_tables[p].values())
        manifest[p] = {
            "mode": sens[p].get("mode"),
            "per_element": bool(sens[p].get("per_element", False)),
            "factor_min": min(vals),
            "factor_mean": float(np.mean(vals)),
            "factor_max": max(vals),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_run", required=False, type=int, default=None)
    parser.add_argument("--num_scenarios", required=False, type=int, default=None,
                        help="Override sensitivity_config.num_scenarios.")
    parser.add_argument("--seed", required=False, type=int, default=None,
                        help="Override sensitivity_config.seed.")
    args = parser.parse_args()
    num_run = args.num_run
    if num_run is None:
        while True:
            try:
                num_run = int(input("Please enter num_run: "))
                break
            except ValueError:
                print("num_run must be an integer. Please try again.")

    with open(f"settings/parameters_{num_run}.json", "r") as f:
        params = json.load(f)
    sens = params.get("sensitivity_config")
    if not sens:
        raise SystemExit(
            f"settings/parameters_{num_run}.json has no 'sensitivity_config' block. "
            "Add one (see README section e) before generating scenarios.")

    num_scenarios = args.num_scenarios if args.num_scenarios is not None else int(sens.get("num_scenarios", 1))
    base_seed = args.seed if args.seed is not None else int(sens.get("seed", 0))

    cloud_dir = f"cloud/run_{num_run}"
    workflow_dir = f"workflows/run_{num_run}"

    # Remove stale scenario_* folders from a previous (possibly larger) run so a
    # reduced num_scenarios does not leave orphans that run_sensitivity.py would
    # still pick up. The scale_sweep/ subfolder is left untouched.
    import shutil
    for old in glob.glob(os.path.join("sensitivity", f"run_{num_run}", "scenario_*")):
        if os.path.isdir(old):
            shutil.rmtree(old, ignore_errors=True)

    for i in range(num_scenarios):
        rng = np.random.default_rng(base_seed + i)
        out_dir = os.path.join("sensitivity", f"run_{num_run}", f"scenario_{i}")
        os.makedirs(out_dir, exist_ok=True)
        manifest = {
            "scenario": i,
            "num_run": num_run,
            "seed": base_seed + i,
            "config": sens,
        }
        # Cloud-resident node parameters (demand + latency L); the placement
        # solution files (opt_*.json) are never touched.
        perturb_cloud(rng, sens, cloud_dir, out_dir, manifest)
        perturb_workflows(rng, sens, workflow_dir, out_dir, manifest)
        with open(os.path.join(out_dir, "scenario_manifest.json"), "w") as f:
            json.dump(manifest, f, indent=4)
        print(f"scenario_{i}: wrote {out_dir}")

    print(f"Generated {num_scenarios} scenario(s) under sensitivity/run_{num_run}/")


if __name__ == "__main__":
    main()
