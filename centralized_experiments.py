# Generate the parameter files for the comparison against the centralized solve.
#
# Usage: centralized_experiments.py
#
# The decomposed placement is compared against the centralized optimization on
# the flattened architecture, which contains every physical device. That problem
# is a nonconvex MIQCQP over all of them, so it is only tractable on small
# architectures. With two levels and nodes_min = n every one of the four
# level-0 nodes expands into a region of n devices, giving 4n physical devices:
#
#     n = 2 ->  8 devices    n = 4 -> 16 devices
#     n = 3 -> 12 devices    n = 5 -> 20 devices
#
# w_3 is fixed at 1 and the capacity model and the selection policy are varied,
# so the table answers whether the two extensions change the price of
# decomposing rather than only its speed.
#
#     run = BASE_RUN + size_index*4 + ram_index*2 + policy_index
#
# heuristic = False switches the comparison run from the all-on-cloud heuristic
# to the centralized optimization. comparison_gap = 0 keeps it a proven optimum,
# and COMPARISON_TIME_LIMIT = None lets every size run to proven optimality
# however long that takes - the point of the table is the true price of
# decomposing, which a capped solve only bounds. Set it to a number of seconds
# to cap the large sizes again: a solve that hits the cap records its achieved
# gap in results/run_<N>/time.json under "comparison_mipgap", and
# paper_tables.py marks any entry above 1e-4 as an incumbent rather than an
# optimum.
#
# Run afterwards:  run_centralized_study.bat  then  paper_tables.py
import json
import os

BASE_RUN = 340
TEMPLATE = os.path.join("settings", "parameters_1.json")
MANIFEST = os.path.join("settings", "centralized_campaign.json")

NODES_PER_REGION = [2, 3, 4, 5]
W3 = 1.0
SIMULATION_TIME = 400
SIMULATION_SEEDS = [43, 1, 2, 3, 4, 5, 6, 7, 8, 9]
COMPARISON_TIME_LIMIT = None   # seconds, None = run to proven optimality

RAM_SETTINGS = [
    {"key": "off", "label": "aggregate capacity",
     "ram_eligibility": False, "ram_capacity_norm": "off"},
    {"key": "l2", "label": "eligibility $+$ $\\ell_2$ capacity",
     "ram_eligibility": True, "ram_capacity_norm": "l2"},
]
POLICIES = [
    {"key": "det", "label": "deterministic", "stochastic_selection": False},
    {"key": "stoch", "label": "stochastic", "stochastic_selection": True},
]


def run_number(size_index, ram_index, policy_index):
    return (BASE_RUN + size_index * len(RAM_SETTINGS) * len(POLICIES)
            + ram_index * len(POLICIES) + policy_index)


if __name__ == "__main__":
    with open(TEMPLATE, "r") as f:
        template = json.load(f)

    runs = {}
    for si, n in enumerate(NODES_PER_REGION):
        for ri, ram in enumerate(RAM_SETTINGS):
            for pi, pol in enumerate(POLICIES):
                num_run = run_number(si, ri, pi)
                cfg = json.loads(json.dumps(template))   # deep copy
                cfg["num_levels"] = 2
                cfg["w_3"] = W3
                cfg["config_random_cloud"]["nodes_min"] = n
                cfg["config_random_cloud"]["nodes_max"] = n + 1
                cfg["comparison"] = True
                cfg["heuristic"] = False                 # centralized, not the heuristic
                cfg["stochastic_selection"] = pol["stochastic_selection"]
                cfg["ram_eligibility"] = ram["ram_eligibility"]
                cfg["ram_capacity_norm"] = ram["ram_capacity_norm"]
                cfg["simulation_time"] = SIMULATION_TIME
                cfg["simulation_seed"] = SIMULATION_SEEDS[0]
                cfg["comparison_time_limit"] = COMPARISON_TIME_LIMIT
                cfg.pop("own_time_limit", None)
                path = os.path.join("settings", f"parameters_{num_run}.json")
                with open(path, "w") as f:
                    json.dump(cfg, f, indent=4)
                runs[str(num_run)] = {
                    "nodes_per_region": n,
                    "physical_nodes": 4 * n,
                    "ram": ram["key"],
                    "policy": pol["key"],
                    "stochastic_selection": pol["stochastic_selection"],
                    "ram_eligibility": ram["ram_eligibility"],
                    "ram_capacity_norm": ram["ram_capacity_norm"],
                }

    manifest = {
        "base_run": BASE_RUN,
        "template": TEMPLATE,
        "nodes_per_region": NODES_PER_REGION,
        "ram_settings": RAM_SETTINGS,
        "policies": POLICIES,
        "w_3": W3,
        "simulation_seeds": SIMULATION_SEEDS,
        "simulation_time": SIMULATION_TIME,
        "comparison_time_limit": COMPARISON_TIME_LIMIT,
        "num_levels": 2,
        "num_workflows": template["num_workflows"],
        "seed": template["seed"],
        "runs": runs,
    }
    with open(MANIFEST, "w") as f:
        json.dump(manifest, f, indent=4)

    lo, hi = min(int(k) for k in runs), max(int(k) for k in runs)
    print(f"wrote {len(runs)} parameter files: settings/parameters_{lo}.json "
          f"... parameters_{hi}.json")
    print(f"wrote manifest {MANIFEST}")
    print(f"  devices  : {[4 * n for n in NODES_PER_REGION]}")
    print(f"  models   : {[r['key'] for r in RAM_SETTINGS]}, "
          f"policies {[p['key'] for p in POLICIES]}, w_3={W3}")
    print("  centralized comparison, time limit "
          + (f"{COMPARISON_TIME_LIMIT} s" if COMPARISON_TIME_LIMIT else "none"))
