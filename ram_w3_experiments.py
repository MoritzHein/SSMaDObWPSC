# Generate the parameter files for the RAM-model x w_3 x policy study.
#
# Usage: ram_w3_experiments.py
#
# Starting from settings/parameters_1.json (2 levels, 10 workflows, 10-11 nodes
# per region => 40 physical nodes / 20 leaves, seed 42), 24 runs are written to
# settings/parameters_300.json ... parameters_323.json, spanning
#
#   RAM capacity model : "off" (ram_eligibility=False, ram_capacity_norm="off")
#                        "l2"  (ram_eligibility=True,  ram_capacity_norm="l2")
#   w_3                : 0.5, 1.0, 1.5, 2.0, 2.5, 3.0
#   selection policy   : deterministic / stochastic (stochastic_selection)
#
# The run number of a cell is
#     run = BASE_RUN + ram_index*12 + w3_index*2 + policy_index
# but nothing downstream needs that arithmetic: the mapping is also written to
# settings/ram_w3_campaign.json, which run_ram_w3_study.bat, resimulate_seeds.py
# and visualization_ram_w3.py all read (same idea as the weights_2.json that
# generate_different_weights.py writes for visualization.py).
#
# Run afterwards:  run_ram_w3_study.bat   then   visualization_ram_w3.py
import json
import os

BASE_RUN = 300
TEMPLATE = os.path.join("settings", "parameters_1.json")
MANIFEST = os.path.join("settings", "ram_w3_campaign.json")

W3_VALUES = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

# Length of the simulated arrival stream, written explicitly into every generated
# parameter file rather than inherited from the template, because the campaign's
# results depend on it and the value should be visible in the manifest.
#
# The queue does not reach a steady state at this arrival rate: waiting per
# request keeps growing to the end of the arrival stream, roughly 2.75x faster
# under the deterministic policy than under the stochastic one, so the cost gap
# between the two policies is a function of this horizon and does not converge.
# At the previous value of 200 the run was also drain-dominated - the last
# request finished around t=370, so 44% of all requests completed after arrivals
# had stopped. At 400 that share is down to about a third.
#
# simulation_time enters only selection_simulation.py and
# evaluate_comparison_objective.py; the optimization objective uses the rates
# lambda rather than lambda * T. Changing it therefore needs NO re-solve - it is
# enough to re-run resimulate_seeds.py over the existing placements.
SIMULATION_TIME = 400

# The two RAM capacity models compared (see ram_capacity_model.md).
RAM_SETTINGS = [
    {"key": "off", "label": "RAM model off (aggregate capacity)",
     "ram_eligibility": False, "ram_capacity_norm": "off"},
    {"key": "l2", "label": "eligibility + $\\ell_2$ capacity norm",
     "ram_eligibility": True, "ram_capacity_norm": "l2"},
]

POLICIES = [
    {"key": "det", "label": "deterministic", "stochastic_selection": False},
    {"key": "stoch", "label": "stochastic", "stochastic_selection": True},
]

# A single simulation at simulation_time=200 carries ~11% Monte-Carlo noise, so
# every placement is re-simulated on these request streams by
# resimulate_seeds.py. The first entry is the baseline seed left in the
# parameter file.
SIMULATION_SEEDS = [43, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# No early stopping: every subproblem is solved to proven optimality. The
# template's own_gap / comparison_gap are 0 and no TimeLimit is set, so the
# reported costs are optima and not incumbents.
#
# This is deliberately the slow setting. Relaxing H turns each subproblem into a
# nonconvex MIQCQP and the stochastic cells are far slower than the
# deterministic ones (run 301 needs more than 10 min against 18 s for run 300),
# so budget hours rather than minutes for the twelve stochastic runs.
#
# If you ever need a quick, non-publishable scan, set this to a number of
# seconds. Capped solves then return their incumbent, and the achieved gap of
# every subproblem is recorded in results/run_<N>/time.json under "own_mipgap"
# so a non-proven solution stays visible. Leave it at None for real results.
OWN_TIME_LIMIT = None


def run_number(ram_index, w3_index, policy_index):
    return BASE_RUN + ram_index * (len(W3_VALUES) * len(POLICIES)) \
        + w3_index * len(POLICIES) + policy_index


if __name__ == "__main__":
    with open(TEMPLATE, "r") as f:
        template = json.load(f)

    runs = {}
    for ri, ram in enumerate(RAM_SETTINGS):
        for wi, w3 in enumerate(W3_VALUES):
            for pi, pol in enumerate(POLICIES):
                num_run = run_number(ri, wi, pi)
                cfg = json.loads(json.dumps(template))      # deep copy
                cfg["w_3"] = w3
                cfg["stochastic_selection"] = pol["stochastic_selection"]
                cfg["ram_eligibility"] = ram["ram_eligibility"]
                cfg["ram_capacity_norm"] = ram["ram_capacity_norm"]
                cfg["simulation_time"] = SIMULATION_TIME
                cfg["simulation_seed"] = SIMULATION_SEEDS[0]
                if OWN_TIME_LIMIT is not None:
                    cfg["own_time_limit"] = OWN_TIME_LIMIT
                else:
                    cfg.pop("own_time_limit", None)
                path = os.path.join("settings", f"parameters_{num_run}.json")
                with open(path, "w") as f:
                    json.dump(cfg, f, indent=4)
                runs[str(num_run)] = {
                    "ram": ram["key"], "w_3": w3, "policy": pol["key"],
                    "stochastic_selection": pol["stochastic_selection"],
                    "ram_eligibility": ram["ram_eligibility"],
                    "ram_capacity_norm": ram["ram_capacity_norm"],
                }

    manifest = {
        "base_run": BASE_RUN,
        "template": TEMPLATE,
        "w3_values": W3_VALUES,
        "ram_settings": RAM_SETTINGS,
        "policies": POLICIES,
        "simulation_seeds": SIMULATION_SEEDS,
        "own_time_limit": OWN_TIME_LIMIT,
        "own_gap": template.get("own_gap"),
        "comparison_gap": template.get("comparison_gap"),
        "num_levels": template["num_levels"],
        "num_workflows": template["num_workflows"],
        "seed": template["seed"],
        "simulation_time": SIMULATION_TIME,
        "w_1": template["w_1"],
        "w_2": template["w_2"],
        "runs": runs,
    }
    with open(MANIFEST, "w") as f:
        json.dump(manifest, f, indent=4)

    lo, hi = min(int(k) for k in runs), max(int(k) for k in runs)
    print(f"wrote {len(runs)} parameter files: settings/parameters_{lo}.json "
          f"... parameters_{hi}.json")
    print(f"wrote manifest {MANIFEST}")
    print(f"  RAM settings : {[r['key'] for r in RAM_SETTINGS]}")
    print(f"  w_3          : {W3_VALUES}")
    print(f"  policies     : {[p['key'] for p in POLICIES]}")
    print(f"  architecture : {template['num_levels']} levels, "
          f"{template['num_workflows']} workflows, seed {template['seed']}")
    print(f"  horizon      : simulation_time={SIMULATION_TIME} "
          f"(template has {template['simulation_time']})")
    print(f"  solver       : own_gap={template.get('own_gap')}, "
          f"comparison_gap={template.get('comparison_gap')}, "
          f"own_time_limit={OWN_TIME_LIMIT}"
          + ("  (solved to proven optimality)" if not OWN_TIME_LIMIT
             and not template.get("own_gap") else
             "  (WARNING: early stopping active, results are incumbents)"))
    print("next: run_ram_w3_study.bat   then   visualization_ram_w3.py")
