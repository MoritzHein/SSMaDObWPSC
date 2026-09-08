# Diagnostic evaluator of the COMPARISON placement's model objective.
#
# Usage: evaluate_comparison_objective.py --num_run=<N> [--scenario_dir=<dir>]
#                                         [--count_weights]
#
# Recomputes the analytic money/time cost of the comparison solution
# (opt_<w>_full.json) exactly like the comparison cost loop of
# selection_simulation.py, but weights every (workflow, leaf) term by the
# EXPECTED number of requests lambda_{t,k} * simulation_time instead of the
# sampled Poisson counts. This is the quantity the optimization problem
# actually minimizes (up to the w_3 terms), so it can check solver guarantees
# (e.g. adapted placement <= stale nominal placement in the perturbed world)
# without the sampling noise of the simulation's count weighting: a run has
# only ~lambda*sim_time requests spread over num_workflows*num_leaves pairs,
# so per-pair count weights carry large Poisson noise that can flip the sign
# of small objective differences between two placements.
#
# --count_weights reproduces the simulation's sampled-count weighting instead
# (same rng seed 42 => identical Poisson stream), as an anchor: with it, the
# printed money/time must equal results_comparison.json to float precision.
#
# Like the simulation, run from the repo root for the nominal placement
# (optionally with --scenario_dir for a perturbed world) and with
# cwd=<sweep-point>/adapted/ for an adapted sandbox. Deterministic one-hot
# selection only (stochastic_selection runs are rejected). Prints JSON to
# stdout; writes nothing.
import argparse
import json
import os

import numpy as np

if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument("--num_run", required=True, type=int)
        parser.add_argument("--scenario_dir", required=False, type=str, default=None)
        parser.add_argument("--count_weights", action="store_true",
                            help="Weight by the simulation's sampled Poisson counts "
                                 "instead of lambda*sim_time (anchor check against "
                                 "results_comparison.json).")
        parser.add_argument("--model_objective", action="store_true",
                            help="Evaluate the OPTIMIZATION MODEL's objective formula "
                                 "(Eq. (1), w_3 terms excluded) instead of the "
                                 "simulation's analytic formula. Differences: money is "
                                 "counted once per function (the analytic formula counts "
                                 "shared functions once per branch), inter-function "
                                 "transfers follow the adjacency a, and the user<->node "
                                 "latencies of the input/output functions are added "
                                 "outside the branch maximum (the analytic formula puts "
                                 "them inside). This is the quantity the solver "
                                 "provably minimizes.")
        args = parser.parse_args()
        num_run = args.num_run
        scenario_dir = args.scenario_dir

        def in_path(kind, name):
                if scenario_dir:
                        cand = os.path.join(scenario_dir, kind, name)
                        if os.path.exists(cand):
                                return cand
                return os.path.join(kind, f"run_{num_run}", name)

        def load(path):
                with open(path, "r") as f:
                        return json.load(f)

        parameter_general_config = load(f"settings/parameters_{num_run}.json")
        if parameter_general_config.get("stochastic_selection", False):
                raise SystemExit("stochastic_selection runs are not supported "
                                 "(the comparison uses a sampled routing policy).")
        w_1 = parameter_general_config["w_1"]
        w_2 = parameter_general_config["w_2"]
        num_workflows = parameter_general_config["num_workflows"]
        sim_time = parameter_general_config["simulation_time"]
        provider_cost_list = parameter_general_config["provider_cost_list"]

        cloud_general_config = load(f"cloud/run_{num_run}/general_config.json")
        num_leaves = cloud_general_config["num_leaves"]
        provider_list = cloud_general_config["provider_list"]
        num_other = len(provider_list[-1]) - num_leaves

        cloud_config = load(in_path("cloud", "requirements_cloud_full.json"))
        request_rates = np.zeros((num_workflows, num_leaves))
        for t in range(num_workflows):
                for k in range(num_leaves):
                        request_rates[t, k] = cloud_config["providers"][f"provider_{k + num_other}"]["request_rate"]

        if args.count_weights:
                # The simulation's exact request sample: same seed, same draw
                # order (one poisson over the full rate matrix per second).
                # That replica is only valid for the stationary Poisson stream.
                # With an arrival trace the simulator multiplies the rate matrix
                # by a per-second, per-leaf intensity taken from the trace, so
                # this loop would silently report weights for a workload the run
                # never saw. Fail instead of reporting a wrong number.
                if parameter_general_config.get("arrival_trace"):
                        raise SystemExit(
                                "--count_weights cannot reproduce run %d: it sets "
                                "arrival_trace, so the request counts come from a "
                                "trace-modulated intensity rather than a stationary "
                                "Poisson draw. Use the default lambda*sim_time "
                                "weighting, which the trace leaves unchanged in "
                                "expectation." % num_run)
                rng = np.random.default_rng(42)
                overall_requests = np.zeros((num_workflows, num_leaves))
                for cur_t in range(sim_time):
                        overall_requests += rng.poisson(request_rates)
                weights = overall_requests
        else:
                weights = request_rates * sim_time

        # ---- Model-objective evaluation (Eq. (1) of optimization.py with the
        # given placement/selection plugged in, w_3 terms excluded): rebuild
        # the parameter-dependent matrices (C, Speed*Time) from the perturbed
        # workflow/cloud files, take D/D_in from the solution file (they only
        # depend on data sizes and egress prices, which no sweep axis
        # perturbs), and evaluate money once per function via the adjacency a
        # and time as T[n] (branch max) plus the start/end user latencies
        # added outside the max - exactly as solve_opt's objective does.
        if args.model_objective:
                obj_money = 0.0
                obj_time = 0.0
                for t in range(num_workflows):
                        workflow_config = load(in_path("workflows", f"requirements_workflow_{t}_0.json"))
                        placement_config = load(in_path("placement", f"opt_{t}_full.json"))
                        num_functions = workflow_config["num_functions"]
                        H = placement_config['H']
                        P = placement_config['P']
                        D = placement_config['D']
                        D_in = placement_config['D_in']
                        a = np.array(placement_config['a'])
                        b = np.array(placement_config['b'])
                        L = cloud_config.get('L', placement_config['L'])
                        num_paths = workflow_config['paths']['num_paths']
                        func_names = list(workflow_config['functions'])
                        m_start = [func_names.index(f) for f in workflow_config['input_functions']]
                        m_end = [func_names.index(f) for f in workflow_config['output_functions']]

                        for k in range(num_leaves):
                                n = H[k].index(1)
                                leaf_node = k + num_other
                                places = [P[n * num_functions + m].index(1) for m in range(num_functions)]
                                money = 0.0
                                for m in range(num_functions):
                                        i = places[m]
                                        func_conf = workflow_config['functions'][f"function_{m}"]
                                        prov = cloud_config['providers'][f"provider_{i}"]
                                        for key, value in func_conf['data_dependencies'].items():
                                                if key != prov['name']:
                                                        money += provider_cost_list[key]["pricing_Storage_Transfer"] * value
                                        if prov['name'] != 'tinyfaas':
                                                money += prov['pricing_RAM'] * func_conf['ram'] * func_conf['time']
                                        for m_s in range(num_functions):
                                                if a[m][m_s] == 1:
                                                        money += D[m][i][places[m_s]]
                                for m in m_start:
                                        money += np.ravel(D_in[places[m]])[0]
                                for m in m_end:
                                        money += D[m][places[m]][leaf_node]

                                branch_times = []
                                for s in range(num_paths):
                                        tot = 0.0
                                        for m in range(num_functions):
                                                if b[s][m] != 1:
                                                        continue
                                                func_conf = workflow_config['functions'][f"function_{m}"]
                                                prov = cloud_config['providers'][f"provider_{places[m]}"]
                                                speedup = 1 if prov['name'] == 'tinyfaas' else func_conf['speedup']
                                                tot += speedup * func_conf['time']
                                                for m_s in range(num_functions):
                                                        if a[m][m_s] == 1:
                                                                tot += L[places[m]][places[m_s]]
                                        branch_times.append(tot)
                                time_n = max(branch_times) if branch_times else 0.0
                                # solve_opt adds L[leaf, i] for both the input and the
                                # output functions (optimization.py:326-328).
                                for m in m_start + m_end:
                                        time_n += L[leaf_node][places[m]]

                                obj_money += weights[t, k] * money
                                obj_time += weights[t, k] * time_n

                        print_result = {
                                "num_run": num_run,
                                "scenario_dir": scenario_dir,
                                "weighting": "sampled_counts" if args.count_weights else "lambda*sim_time",
                                "formula": "model_objective",
                                "money": obj_money,
                                "time": obj_time,
                                "objective": w_1 * obj_money + w_2 * obj_time,
                        }
                print(json.dumps(print_result, indent=4))
                raise SystemExit(0)

        # ---- Analytic cost of the comparison solution, mirroring the loop of
        # selection_simulation.py (comparison block) term by term; only the
        # weight source differs. Quirks of that loop are kept intentionally so
        # --count_weights reproduces results_comparison.json exactly.
        time_cost_all = 0
        money_cost_all = 0
        for t in range(num_workflows):
                workflow_config = load(in_path("workflows", f"requirements_workflow_{t}_0.json"))
                placement_config = load(in_path("placement", f"opt_{t}_full.json"))
                num_functions = workflow_config["num_functions"]
                H = placement_config['H']
                P = placement_config['P']
                D = placement_config['D']
                D_in = placement_config['D_in']
                L = cloud_config.get('L', placement_config['L'])
                num_paths = workflow_config['paths']['num_paths']
                b = placement_config['b']
                # Branch traversal in EXECUTION order. b only records membership, so
                # np.where(b[s]==1) yields ascending function ids, which is not the
                # order the DAG runs them in (paths are non-monotone). Mirrors
                # branch_functions() in selection_simulation.py, which this loop has
                # to reproduce term by term.
                path_order = [list(workflow_config['paths'][str(s)]) for s in range(num_paths)]

                work_items = [(k, H[k].index(1), weights[t, k]) for k in range(num_leaves)]
                for k, selected_workflow, weight in work_items:
                        time_cost = 0
                        money_cost = 0
                        T_paths = np.zeros((num_paths, 1))
                        D_func = np.zeros((num_functions, 1))
                        C_func = np.zeros((num_functions, 1))
                        L_func = np.zeros((num_functions, 1))
                        for s in range(num_paths):
                                pred_k = k + num_other
                                func_nums = [m for m in path_order[s] if b[s][m] == 1]
                                for r, m in enumerate(func_nums):
                                        selected_place = P[selected_workflow * num_functions + m].index(1)
                                        # r is the POSITION of m in func_nums (enumerate); the
                                        # end-of-branch test must use it, not the function id m.
                                        if r < len(func_nums) - 1:
                                                next_selected_place = P[selected_workflow * num_functions + func_nums[r + 1]].index(1)
                                        else:
                                                next_selected_place = pred_k

                                        L_func[m] = L[pred_k][selected_place]
                                        if m == 0:
                                                D_in_0 = D_in[selected_place]
                                        D_func[m] = D[m][selected_place][next_selected_place]

                                        if cloud_config['providers'][f"provider_{selected_place}"]["name"] == "tinyfaas":
                                                speedup = 1
                                        else:
                                                speedup = workflow_config["functions"][f"function_{m}"]['speedup']
                                        ram = workflow_config['functions'][f"function_{m}"]['ram']
                                        time = workflow_config['functions'][f"function_{m}"]['time']
                                        T_func = speedup * time
                                        data_dependencies = workflow_config['functions'][f"function_{m}"]['data_dependencies']
                                        for key in data_dependencies.keys():
                                                value = data_dependencies[key]
                                                pricing_Storage_Transfer = provider_cost_list[key]["pricing_Storage_Transfer"]
                                                if key != cloud_config["providers"][f"provider_{selected_place}"]["name"]:
                                                        C_func[m] = C_func[m] + pricing_Storage_Transfer * value
                                        pricing_RAM = cloud_config['providers'][f"provider_{selected_place}"]['pricing_RAM']
                                        C_func[m] += pricing_RAM * ram * time

                                        T_paths[s] += (T_func + L_func[m])
                                        pred_k = selected_place

                                T_paths[s] += L[selected_place][k + num_other]

                        time_cost += np.max(T_paths)
                        money_cost += np.sum(C_func) + np.sum(D_func) + D_in_0

                        time_cost_all += weight * time_cost
                        money_cost_all += weight * money_cost

        money = float(np.ravel(money_cost_all)[0])
        time_c = float(np.ravel(time_cost_all)[0])
        print(json.dumps({
                "num_run": num_run,
                "scenario_dir": scenario_dir,
                "weighting": "sampled_counts" if args.count_weights else "lambda*sim_time",
                "money": money,
                "time": time_c,
                "objective": w_1 * money + w_2 * time_c,
        }, indent=4))
