# Blocking-aware selection campaign (runs 500-607) + heuristic references (620-625).
#
# Usage: policy_experiments.py [--base=500] [--pilot]
#
# Three axes:
#
#   memory model  aggregate RAM_max,i ("off")  vs  eligibility + l2
#   w_3           0.5 .. 3.0
#   selection     9 runtime rules on top of the deterministic or the
#                 stochastic placement
#
# A selection rule is a runtime decision. It changes which deployment serves a
# request, never where the functions were placed, so a variant needs no new
# solve: every run CLONES the generated inputs and the solved placement of its
# source run and differs only in the "selection_policy" block of its parameter
# file. Cloning rather than re-solving also removes the risk that the solver
# returns a different optimum of equal objective, which would confound the
# comparison against the open-loop baseline.
#
# Sources are the RAM x w_3 campaign, which supplies the open-loop baselines too:
#
#     w_3          0.5   1.0   1.5   2.0   2.5   3.0
#     off,   det   300   302   304   306   308   310
#     off,   stoch 301   303   305   307   309   311
#     l2,    det   312   314   316   318   320   322
#     l2,    stoch 313   315   317   319   321   323
#
# On the DETERMINISTIC placement the support of H holds a single deployment -
# selection is not recomputed below the top level, so a whole region shares one -
# and a rule restricted to that support is a no-op. Runtime recourse there has to
# ignore H and rank the placed deployments directly. That is the competing policy
# a reader will ask for, and it isolates how much of the gain is merely "load is
# no longer pinned".
#
# The policy runs set comparison=false: the model-free baselines do not depend on
# w_3, on the memory model or on the selection rule, so they are simulated once
# each as runs 620-625 instead of being recomputed in all 108 cells. That halves
# the campaign.
import argparse
import json
import os
import shutil

BASE_RUN = 500
HEURISTIC_BASE = 620
MANIFEST = os.path.join("settings", "policy_campaign.json")

W3_VALUES = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
RAM_MODELS = [
    {"key": "off", "label": "aggregate", "offset": 0},
    {"key": "l2", "label": "eligibility + $\\ell_2$", "offset": 12},
]
# run = 300 + ram_offset + w3_index*2 + (0 det | 1 stoch)
SOURCE_BASE = 300

SIMULATION_SEEDS = [43, 1, 2, 3, 4, 5, 6, 7, 8, 9]

VARIANTS = [
    # Runtime recourse on the deterministic placement: the competing policy.
    {"key": "det_ll", "source": "det", "label": "det. + least loaded",
     "policy": {"candidates": "placed", "rule": "least_loaded"}},
    {"key": "det_free", "source": "det", "label": "det. + free sample",
     "policy": {"candidates": "placed", "rule": "free_sample"}},
    # Rules confined to the support of H: the realized routing stays on the
    # distribution the placement was priced with.
    {"key": "free_sup", "source": "stoch", "label": "stoch. + free sample, supp.",
     "policy": {"candidates": "support", "rule": "free_sample"}},
    {"key": "deficit_sup", "source": "stoch", "label": "stoch. + deficit, supp.",
     "policy": {"candidates": "support", "rule": "deficit_free"}},
    {"key": "uniform_sup", "source": "stoch", "label": "stoch. + uniform, supp.",
     "policy": {"candidates": "support", "rule": "free_uniform"}},
    # Rules that may leave the support once nothing in it is free.
    {"key": "free_wide", "source": "stoch", "label": "stoch. + free sample, wide",
     "policy": {"candidates": "support_then_placed", "rule": "free_sample"}},
    {"key": "argmaxh", "source": "stoch", "label": "stoch. + argmax H, wide",
     "policy": {"candidates": "support_then_placed", "rule": "free_argmax_h"}},
    {"key": "deficit_wide", "source": "stoch", "label": "stoch. + deficit, wide",
     "policy": {"candidates": "support_then_placed", "rule": "deficit_free"}},
    # Ignores H entirely; the upper bound of what runtime recourse can do.
    {"key": "stoch_ll", "source": "stoch", "label": "stoch. + least loaded",
     "policy": {"candidates": "placed", "rule": "least_loaded"}},
]

# Model-free placements evaluated as the comparison run. None depends on w_3,
# on the memory model or on the selection rule, so one run each is enough.
HEURISTICS = [
    {"key": "all_cloud", "label": "all on cloud"},
    {"key": "all_cloud_rr", "label": "all on cloud, spread"},
    {"key": "local_first", "label": "edge first"},
    {"key": "best_fit_edge", "label": "best fit edge"},
    {"key": "greedy_cost", "label": "greedy cost"},
    {"key": "random_feasible", "label": "random"},
]
# Settings the heuristic runs inherit; only the comparison placement matters.
HEURISTIC_TEMPLATE = 316

CLONED_TREES = ("cloud", "workflows", "placement")


def source_run(ram_offset, w3, policy):
    return SOURCE_BASE + ram_offset + W3_VALUES.index(w3) * 2 + (1 if policy == "stoch" else 0)


def clone_inputs(source, target):
    """Copy the generated architecture, workflows and solved placement.

    Only these three trees are read by selection_simulation.py; results/ is
    written by it and must not be inherited.
    """
    for tree in CLONED_TREES:
        src = os.path.join(tree, "run_%d" % source)
        dst = os.path.join(tree, "run_%d" % target)
        if not os.path.isdir(src):
            raise SystemExit("missing %s - run the RAM x w_3 campaign first" % src)
        if os.path.isdir(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=int, default=BASE_RUN)
    parser.add_argument("--pilot", action="store_true",
                        help="only w_3=1.5, for a quick end-to-end check")
    args = parser.parse_args()

    w3_values = [1.5] if args.pilot else W3_VALUES

    runs = {}
    for ram in RAM_MODELS:
        for w3 in w3_values:
            for vi, variant in enumerate(VARIANTS):
                src = source_run(ram["offset"], w3, variant["source"])
                cfg = json.load(open(os.path.join("settings", "parameters_%d.json" % src)))
                num_run = (args.base + RAM_MODELS.index(ram) * len(W3_VALUES) * len(VARIANTS)
                           + W3_VALUES.index(w3) * len(VARIANTS) + vi)
                cfg["selection_policy"] = dict(variant["policy"])
                cfg["simulation_seed"] = SIMULATION_SEEDS[0]
                # The model-free baselines are runs 620-625; simulating one in
                # every cell would double the campaign for no extra information.
                cfg["comparison"] = False
                json.dump(cfg, open(os.path.join("settings", "parameters_%d.json" % num_run), "w"),
                          indent=4)
                clone_inputs(src, num_run)
                runs[str(num_run)] = {
                    "ram": ram["key"], "w_3": w3, "policy": variant["key"],
                    "label": variant["label"], "placement": variant["source"],
                    "source_run": src,
                    "candidates": variant["policy"]["candidates"],
                    "rule": variant["policy"]["rule"],
                }
    print("policy runs: %d" % len(runs))

    # ---- Model-free references. These DO need placement.py, because the
    # comparison placement is built inside the solve call.
    heuristic_runs = {}
    tmpl = json.load(open(os.path.join("settings", "parameters_%d.json" % HEURISTIC_TEMPLATE)))
    for hi, heur in enumerate(HEURISTICS):
        num_run = HEURISTIC_BASE + hi
        cfg = json.loads(json.dumps(tmpl))
        cfg["comparison"] = True
        cfg["heuristic"] = True
        cfg["comparison_mode"] = heur["key"]
        cfg["simulation_seed"] = SIMULATION_SEEDS[0]
        cfg.pop("selection_policy", None)          # the baselines stay open loop
        json.dump(cfg, open(os.path.join("settings", "parameters_%d.json" % num_run), "w"),
                  indent=4)
        heuristic_runs[str(num_run)] = {"mode": heur["key"], "label": heur["label"]}
        # resimulate_seeds.py iterates manifest["runs"], so a reference that
        # lives only under "heuristic_runs" would be silently skipped.
        runs[str(num_run)] = {"kind": "heuristic", "mode": heur["key"],
                              "label": heur["label"]}
    print("heuristic runs: %d (need placement.py, unlike the policy runs)"
          % len(heuristic_runs))

    baselines = {}
    for ram in RAM_MODELS:
        for w3 in w3_values:
            for pol in ("det", "stoch"):
                baselines["%s|%s|%s" % (ram["key"], w3, pol)] = source_run(ram["offset"], w3, pol)

    manifest = {
        "base_run": args.base,
        "heuristic_base": HEURISTIC_BASE,
        "w3_values": w3_values,
        "ram_models": RAM_MODELS,
        "variants": VARIANTS,
        "heuristics": HEURISTICS,
        "simulation_seeds": SIMULATION_SEEDS,
        "baselines": baselines,
        "heuristic_runs": heuristic_runs,
        "runs": runs,
    }
    json.dump(manifest, open(MANIFEST, "w"), indent=4)
    lo, hi = min(int(k) for k in runs), max(int(k) for k in runs)
    print("\nwrote %s" % MANIFEST)
    print("  policy    runs %d-%d  (no solve: simulate once, then resimulate_seeds)" % (lo, hi))
    print("  heuristic runs %d-%d  (placement.py first)"
          % (HEURISTIC_BASE, HEURISTIC_BASE + len(HEURISTICS) - 1))
