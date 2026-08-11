# Generate the parameter files for the scalability study over the number of
# hierarchical levels.
#
# Usage: scaling_levels_experiments.py
#
# Level 0 always holds num_tiny_first + len(provider_list) - 1 = 4 nodes, and
# every further level expands each node into nodes_min..nodes_max nodes
# (generate_cloud in cloud_workflow_setup.py). With nodes_min = 10 the depth
# therefore fixes the size of the architecture:
#
#     2 levels ->   40 physical devices,   5 subproblems
#     3 levels ->  400 physical devices,  45 subproblems
#     4 levels -> 4000 physical devices, 445 subproblems
#
# Everything else is held at the values of the RAM x w_3 x policy campaign:
# eligibility + l2 capacity model, w_3 = 1, ten workflows, architecture seed 43,
# simulation_time = 400. Only the selection policy is varied alongside the
# depth, so the table shows how the decomposed solution time and the effect of
# the policy behave as the architecture grows.
#
#     run = BASE_RUN + level_index*2 + policy_index
#
# The mapping is also written to settings/scaling_campaign.json, which
# resimulate_seeds.py reads through --manifest.
#
# All six runs set comparison = True, the four-level ones included. That flag
# does double duty: it is what makes placement.py write the flattened
# architecture (generate_full_cloud), and selection_simulation.py reads that
# file for the request rates and the per-node RAM limits even when it only
# evaluates the decomposed solution. Without it the four-level runs cannot be
# simulated at all and paper_tables.py prints "not simulated" for them. The
# price is that the centralized solve runs over 4000 devices, whose latency
# matrix has 16 million entries; those two placements are the expensive part of
# the study.
#
# Run afterwards:  run_scaling_levels_study.bat  then  paper_tables.py
import json
import os

BASE_RUN = 330
TEMPLATE = os.path.join("settings", "parameters_1.json")
MANIFEST = os.path.join("settings", "scaling_campaign.json")

LEVELS = [2, 3, 4]
W3 = 1.0
SIMULATION_TIME = 400
SIMULATION_SEEDS = [43, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# Depth up to which the flattened architecture is built, see above. Runs deeper
# than this are solved but not simulated.
MAX_LEVELS_WITH_COMPARISON = 100

# A subproblem that is still open after this many seconds returns its incumbent
# and records the achieved gap in results/run_<N>/time.json under "own_mipgap",
# so a solution that is not proven optimal stays visible in the table. The two
# level campaigns of the paper are checked for this before any time is quoted.
OWN_TIME_LIMIT = None

POLICIES = [
    {"key": "det", "label": "deterministic", "stochastic_selection": False},
    {"key": "stoch", "label": "stochastic", "stochastic_selection": True},
]


def run_number(level_index, policy_index):
    return BASE_RUN + level_index * len(POLICIES) + policy_index


if __name__ == "__main__":
    with open(TEMPLATE, "r") as f:
        template = json.load(f)

    runs = {}
    for li, levels in enumerate(LEVELS):
        for pi, pol in enumerate(POLICIES):
            num_run = run_number(li, pi)
            cfg = json.loads(json.dumps(template))       # deep copy
            cfg["num_levels"] = levels
            cfg["w_3"] = W3
            cfg["stochastic_selection"] = pol["stochastic_selection"]
            cfg["ram_eligibility"] = True
            cfg["ram_capacity_norm"] = "l2"
            cfg["simulation_time"] = SIMULATION_TIME
            cfg["simulation_seed"] = SIMULATION_SEEDS[0]
            cfg["comparison"] = levels <= MAX_LEVELS_WITH_COMPARISON
            cfg["own_time_limit"] = OWN_TIME_LIMIT
            path = os.path.join("settings", f"parameters_{num_run}.json")
            with open(path, "w") as f:
                json.dump(cfg, f, indent=4)
            runs[str(num_run)] = {
                "num_levels": levels,
                "policy": pol["key"],
                "stochastic_selection": pol["stochastic_selection"],
                "simulated": cfg["comparison"],
            }

    manifest = {
        "base_run": BASE_RUN,
        "template": TEMPLATE,
        "levels": LEVELS,
        "policies": POLICIES,
        "w_3": W3,
        "simulation_seeds": SIMULATION_SEEDS,
        "simulation_time": SIMULATION_TIME,
        "own_time_limit": OWN_TIME_LIMIT,
        "num_workflows": template["num_workflows"],
        "seed": template["seed"],
        "nodes_min": template["config_random_cloud"]["nodes_min"],
        "runs": runs,
    }
    with open(MANIFEST, "w") as f:
        json.dump(manifest, f, indent=4)

    lo, hi = min(int(k) for k in runs), max(int(k) for k in runs)
    print(f"wrote {len(runs)} parameter files: settings/parameters_{lo}.json "
          f"... parameters_{hi}.json")
    print(f"wrote manifest {MANIFEST}")
    print(f"  levels   : {LEVELS} (simulated up to {MAX_LEVELS_WITH_COMPARISON})")
    print(f"  w_3      : {W3}, eligibility + l2")
    print(f"  policies : {[p['key'] for p in POLICIES]}")
    print(f"  horizon  : simulation_time={SIMULATION_TIME}, "
          f"{len(SIMULATION_SEEDS)} request streams")
