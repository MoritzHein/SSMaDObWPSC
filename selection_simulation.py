# Online selection simulation (Sec. IV-A/IV-B of the paper).
#
# Usage: selection_simulation.py --num_run=<N>  (after placement.py --num_run=<N>)
#
# Evaluates the placement computed by placement.py by simulating user requests:
# requests arrive at every edge leaf with Poisson rate lambda_k over
# "simulation_time" seconds and are executed FIFO on the placed nodes. Two
# blocking mechanisms are modeled: a function deployment cannot serve two
# requests at once, and a node whose RAM is fully occupied cannot start
# further functions. The accumulated waiting time W (delay beyond the ideal
# branch time base_T), the monetary cost M and the basic time cost B yield the
# overall cost C = w_1*M + w_2*(B+W) used in the case study. If "comparison"
# is set, the same simulation is repeated for the centralized/heuristic
# solution (opt_<w>_full.json) and stored as overall_cost_comparison.
#
# The first half of the script evaluates the DECOMPOSED solution: because each
# subproblem only knows its own region, the end-to-end placement has to be
# reconstructed by following each function from the top-level problem down the
# hierarchy (reconstruct_full_solution / upward_trajectory).
import json
import numpy as np
import argparse
import os

# Cached JSON reader for the simulation's input files. The cost loops re-read
# the same placement/cloud/workflow files thousands of times (per request,
# function and hierarchy level), which is prohibitively expensive on network
# drives. Every access re-validates the cache entry against the file's
# (st_mtime_ns, st_size, st_ino), so a file that changes on disk between two
# reads is re-parsed - only genuinely unchanged files are served from memory.
# Windows caveat: the file-time clock ticks coarsely (~16 ms), so a same-size
# rewrite within the same tick would keep an identical stamp. Therefore a
# matching stamp is only trusted once the file's mtime is at least
# _JSON_CACHE_GRACE_S old; younger files are re-read on every access. Files
# in this pipeline are written once per run stage and only read afterwards,
# so past the grace window an unchanged stamp implies unchanged content.
# Callers must treat the returned object as read-only; the two sites that edit
# the branch matrix `b` copy it first.
from time import time as _wall_clock
_JSON_CACHE = {}
_JSON_CACHE_GRACE_S = 5.0


def load_json_cached(path):
        st = os.stat(path)
        stamp = (st.st_mtime_ns, st.st_size, st.st_ino)
        hit = _JSON_CACHE.get(path)
        if (hit is not None and hit[0] == stamp
                        and _wall_clock() - st.st_mtime_ns / 1e9 > _JSON_CACHE_GRACE_S):
                return hit[1]
        with open(path, "r") as f:
                data = json.load(f)
        _JSON_CACHE[path] = (stamp, data)
        return data


# Branch traversal order. The branch-membership matrix b only records WHICH
# functions belong to a branch, so np.where(b[s] == 1) returns them in ascending
# function id - which is not the order the DAG executes them in. The workflow
# generator writes genuinely non-monotone paths (e.g. [0, 4, 3, 1, 2, 5]), and
# optimization.py builds the successor matrix a - and hence the model's T[n]
# constraint - from exactly those ordered paths. Deriving the traversal from b
# therefore charges execution times, transfer latencies and data-transfer costs
# along a chain the optimizer never modelled.
#
# The order is taken from the workflow config; b still decides MEMBERSHIP, so the
# b_full truncation that drops the shared start/end functions from the secondary
# branches keeps working unchanged. Since b is built as b[s, path] = 1, every
# member of b[s] appears in paths[str(s)] and the filter below can only reorder,
# never add or drop a function.
_PATH_ORDER_WARNED = set()


def path_order_of(wf_conf):
        """Per-branch function lists in EXECUTION order, from the workflow config."""
        paths = (wf_conf or {}).get('paths', {})
        try:
                num_paths = int(paths.get('num_paths', 0))
        except (TypeError, ValueError):
                return []
        return [list(paths.get(str(s), [])) for s in range(num_paths)]


def branch_functions(b, s, path_order=None):
        """Functions of branch s of the branch matrix b, in execution order."""
        if not isinstance(b, (list, np.ndarray)) or len(b) <= s:
                return []
        row = b[s]
        order = None
        if path_order is not None and len(path_order) > s:
                order = path_order[s]
        if not order:
                # No ordered path available: fall back to the historical ascending
                # order, but say so - a workflow config without "paths" is malformed
                # and the traversal is then only as good as the function numbering.
                key = id(b)
                if key not in _PATH_ORDER_WARNED:
                        _PATH_ORDER_WARNED.add(key)
                        print("WARNING: no ordered 'paths' for this workflow; falling back "
                              "to ascending function order for the branch traversal.")
                return np.where(np.array(row) == 1)[0].tolist()
        return [int(m) for m in order if 0 <= int(m) < len(row) and row[int(m)] == 1]


if __name__ == "__main__":


        parser = argparse.ArgumentParser()
        parser.add_argument("--num_run", required=False, type=int, default=None)
        # Optional sensitivity-analysis hook: a folder holding perturbed copies
        # of (some of) the input files. When set, inputs are read from here
        # first and fall back to the baseline run folders; results are written
        # under this folder instead of results/run_<N>/. Unset => unchanged.
        parser.add_argument("--scenario_dir", required=False, type=str, default=None)
        args = parser.parse_args()
        num_run = args.num_run
        scenario_dir = args.scenario_dir
        if num_run is None:
                while True:
                        try:
                                num_run = int(input("Please enter num_run: "))
                                break
                        except ValueError:
                                print("num_run must be an integer. Please try again.")

        # Resolve an input file of a given kind ("cloud"/"workflows"/"placement"):
        # prefer the scenario copy if it exists, otherwise the baseline run dir.
        def in_path(kind, name):
                if scenario_dir:
                        cand = os.path.join(scenario_dir, kind, name)
                        if os.path.exists(cand):
                                return cand
                return os.path.join(kind, f"run_{num_run}", name)

        placement_path=f"placement/run_{num_run}"
        parameter_config_path=f"settings/parameters_{num_run}.json"
        parameter_general_config = load_json_cached(parameter_config_path)
        provider_cost_list=parameter_general_config["provider_cost_list"]
        # Stage 1 stochastic selection: when on, the comparison/centralized
        # evaluation samples each request's deployment from the continuous policy
        # H[leaf,:] and reports a Monte-Carlo cost. Absent => off => unchanged.
        stochastic_selection=parameter_general_config.get("stochastic_selection", False)
        general_config_path=f"cloud/run_{num_run}/general_config.json"
        cloud_general_config = load_json_cached(general_config_path)
        num_leaves=cloud_general_config["num_leaves"]
        num_workflows= parameter_general_config["num_workflows"]
        num_levels=parameter_general_config["num_levels"]

        num_other_first_level=len(parameter_general_config["provider_list"])-1
        num_nodes_in_level = cloud_general_config["num_nodes_in_level"]
        num_probs=0
        comparison=parameter_general_config["comparison"]
        heuristic=parameter_general_config["heuristic"]

        time_cost_all = 0
        money_cost_all = 0
        w_1 = parameter_general_config["w_1"]
        w_2 = parameter_general_config["w_2"]
        w_3 = parameter_general_config["w_3"]
        provider_list=cloud_general_config["provider_list"]
        predecessor_list=cloud_general_config["predecessor_list"]
        num_nodes_in_level_list=cloud_general_config["num_nodes_in_level"]
        num_other=len(provider_list[-1])-num_leaves
        # ind_start_list[l] = index of the first subproblem of level l in the
        # flat opt_<workflow>_<subproblem>.json numbering used by placement.py.
        ind_start_list=[0]
        for value in predecessor_list:
                ind_start_list.append(len(value)+ind_start_list[-1])

        # Walk from a physical node (usually a leaf) up through the hierarchy.
        # Returns, per level: the region index (pred_k_list), the latency from
        # the region's head node to the node (lat_upward_list) and the node's
        # local position inside its region (pos_list). Used to compose
        # end-to-end latencies across regions (Algorithm 1 of the paper).
        def upward_trajectory(leave_k):

                pos_list=[]
                pred_k_list=[]
                lat_upward_list=[]
                for r in range(num_levels-1,-1,-1):
                        ind_r = 0
                        num_curr_nodes = 0
                        for node_len in num_nodes_in_level_list[r]:
                                num_curr_nodes += node_len
                                if num_curr_nodes > leave_k:
                                        pos=leave_k-num_curr_nodes+node_len
                                        pos_list.append(pos)
                                        break
                                ind_r += 1


                        pred_k = ind_r
                        pred_k_list.append(pred_k)

                        placement_path = in_path("placement", f"opt_{t}_{pred_k+ind_start_list[r]}.json")
                        curr_placement = load_json_cached(placement_path)
                        L=curr_placement["L"]

                        lat_upward=L[0][pos]
                        lat_upward_list.append(lat_upward)
                        leave_k=pred_k

                pos_list.reverse()
                pred_k_list.reverse()
                lat_upward_list.reverse()

                return pred_k_list,lat_upward_list,pos_list

        # Given the subproblem opt_num of level r-1 and the node selected_place
        # chosen there, return the index of the level-r subproblem that refines
        # this node (i.e. descend one level in the decomposition tree).
        def find_opt(opt_num,selected_place, r):
                prev_value=opt_num - ind_start_list[r-1]
                if prev_value>0:
                        num_curr_nodes=0
                        for node_len in num_nodes_in_level_list[r-1][0:prev_value]:
                                num_curr_nodes += node_len
                        return selected_place+ind_start_list[r]+num_curr_nodes
                else:
                        return selected_place + ind_start_list[r]
        # Map a deployment index of the parent problem to the row index in a
        # subproblem, where pruned deployments were removed
        # (active_workflow_list keeps the surviving parent indices in order).
        def find_correct_workflow(curr_selected_workflow,workflow_list):
                index=workflow_list.index(curr_selected_workflow)
                return index
        # Convert a node chosen in last-level subproblem opt_num into its
        # global physical node index (offset by all preceding regions' nodes).
        def get_leaf(selected_place,opt_num):
                num_curr_nodes=0

                for node_len in num_nodes_in_level_list[-1][0:opt_num-ind_start_list[-2]]:
                        num_curr_nodes += node_len
                num_curr_nodes+=selected_place
                return num_curr_nodes


        # ---- Simulation setup: request rate lambda_k of every (workflow, leaf)
        # pair, taken from the physical leaves of the flattened architecture.
        # The Monte-Carlo seed is a parameter so the SAME placement can be
        # re-evaluated on independent request streams (sampling error of the
        # cost); absent => 42 => the previous, fixed stream. The deployment
        # sampler is offset by 81 so the default reproduces the old 42/123 pair.
        simulation_seed = int(parameter_general_config.get("simulation_seed", 42))
        np.random.seed(simulation_seed)
        sim_time = parameter_general_config["simulation_time"]
        time_cost_all = 0
        money_cost_all = 0
        select_cost_all = 0
        time_cost = 0
        money_cost = 0
        request_rates = np.zeros((num_workflows, num_leaves))
        for t in range(num_workflows):
                cloud_name = f"requirements_cloud_full.json"
                path_cloud = in_path("cloud", cloud_name)
                cloud_config = load_json_cached(path_cloud)
                for k in range(num_leaves):
                        request_rates[t, k] = cloud_config["providers"][f"provider_{k + num_other}"]["request_rate"]
        requests = np.zeros((sim_time, num_workflows, num_leaves))

        # RAM capacity per physical node: only edge (tinyfaas) nodes are
        # limited; cloud nodes never block due to RAM.
        cloud_full_path = in_path("cloud", "requirements_cloud_full.json")
        cloud_full = load_json_cached(cloud_full_path)
        num_nodes_global = cloud_full.get('num_nodes', 0)
        node_ram_max = [cloud_full['providers'][f'provider_{i}'].get('max_RAM_curr', 1e9) if cloud_full['providers'][f'provider_{i}'].get('name')=="tinyfaas" else np.inf for i in
                        range(num_nodes_global)]


        perwf = {}



        # Reassemble a global placement/selection (P_full, H_full on physical
        # nodes) for workflow w from the per-level subproblem solutions: start
        # at the top-level problem, read which deployment each leaf's region
        # selected (H of opt_<w>_0), then follow every function's placement
        # down the decomposition tree with find_opt until the physical level,
        # converting the final choice to a global node index with get_leaf.
        def reconstruct_full_solution(w):

                wf_path = in_path("workflows", f"requirements_workflow_{w}_0.json")
                try:
                        wf_conf = load_json_cached(wf_path)
                except Exception:
                        wf_conf = {}
                deployment_number=wf_conf["deployment_number"]
                num_funcs = wf_conf.get('num_functions', 0)

                num_nodes_all = cloud_full.get('num_nodes', 0)

                P_full = [[0 for _ in range(num_nodes_all)] for _ in range(num_funcs)]

                H_full = [[0 for _ in range(deployment_number)] for _ in range(num_leaves)]


                L_full = []
                a_full = []
                b_full = []
                full_path = in_path("placement", f"opt_{w}_full.json")
                try:
                        full_place_tmp = load_json_cached(full_path)
                        if 'L' in full_place_tmp:
                                L_full = full_place_tmp.get('L', [])
                        a_full = full_place_tmp.get('a', [])
                        b_full = full_place_tmp.get('b', [])
                        if len(b_full)>1:
                                # Copy before zeroing: the loaded object is shared
                                # via the cache and must stay unmodified.
                                b_full = [list(row) for row in b_full]
                                for r in range(1,len(b_full)):
                                        b_full[r][0]=0
                                        b_full[r][-1]=0
                except Exception:
                        pass
                # Latency now lives in the cloud config; the opt_<w>_full.json L
                # read above is only a fallback for runs generated before L was
                # relocated. Prefer the (perturbable) cloud-config L.
                L_full = cloud_full.get('L', L_full)


                top_path = in_path("placement", f"opt_{w}_0.json")
                try:
                        top_place = load_json_cached(top_path)
                except Exception:
                        top_place = {}

                # ---- Stage 2 stochastic decomposition. The placement of every
                # deployment is leaf-independent, so P_full is built once for ALL
                # deployments; each leaf's routing distribution is read from its
                # own last-level subproblem H and mapped back to original
                # deployment indices via the active_workflow_list chain.
                if stochastic_selection:
                        def _load(path):
                                try:
                                        return load_json_cached(path)
                                except Exception:
                                        return {}

                        def place_deployment(d, m):
                                # Follow deployment d's function m down its placement
                                # chain from the top and return its physical node.
                                prev_selected_place = 0
                                opt_num = 0
                                curr_selected_workflow = d
                                k_curr = 0
                                for r in range(num_levels):
                                        if r != 0:
                                                opt_num = find_opt(opt_num, prev_selected_place, r)
                                        curr_placement = _load(in_path("placement", f"opt_{w}_{opt_num}.json"))
                                        curr_workflow = _load(in_path("workflows", f"requirements_workflow_{w}_{opt_num}.json"))
                                        curr_P = curr_placement.get('P', [])
                                        if r != 0:
                                                try:
                                                        awl = curr_workflow.get('active_workflow_list', [])
                                                        if awl:
                                                                curr_selected_workflow = find_correct_workflow(curr_selected_workflow, awl)
                                                except Exception:
                                                        pass
                                        try:
                                                selected_place = int(np.where(np.array(curr_P[curr_selected_workflow * num_funcs + m]) == 1)[0][0]) if curr_P else 0
                                        except Exception:
                                                selected_place = 0
                                        if r == num_levels - 1:
                                                try:
                                                        k_curr = get_leaf(selected_place, opt_num)
                                                except Exception:
                                                        k_curr = selected_place
                                        prev_selected_place = selected_place
                                return k_curr

                        def leaf_distribution(leaf):
                                dist = [0.0] * deployment_number
                                try:
                                        pred_k_list, _lu, pos_list = upward_trajectory(leaf + num_other)
                                except Exception:
                                        return dist
                                last = num_levels - 1
                                opt_num = pred_k_list[last] + ind_start_list[last]
                                sub = _load(in_path("placement", f"opt_{w}_{opt_num}.json"))
                                Hsub = sub.get('H', [])
                                csub = _load(in_path("cloud", f"requirements_cloud_{last}_{pred_k_list[last]}.json"))
                                try:
                                        leaves_pos = [j for j in range(csub['num_nodes']) if csub['providers'][f'provider_{j}'].get('leave')]
                                        leaf_index = leaves_pos.index(pos_list[last])
                                except Exception:
                                        leaf_index = 0
                                if leaf_index >= len(Hsub):
                                        return dist
                                Hrow = Hsub[leaf_index]
                                # Map each local deployment ld up to the original (top)
                                # deployment index by composing active_workflow_list.
                                for ld in range(len(Hrow)):
                                        if Hrow[ld] <= 0:
                                                continue
                                        d = ld
                                        ok = True
                                        for r in range(last, 0, -1):
                                                onum = pred_k_list[r] + ind_start_list[r]
                                                awl = _load(in_path("workflows", f"requirements_workflow_{w}_{onum}.json")).get('active_workflow_list', [])
                                                if 0 <= d < len(awl):
                                                        d = awl[d]
                                                else:
                                                        ok = False
                                                        break
                                        if ok and 0 <= d < deployment_number:
                                                dist[d] = float(Hrow[ld])
                                return dist

                        for d in range(deployment_number):
                                for m in range(num_funcs):
                                        k_curr = place_deployment(d, m)
                                        row_idx = d * num_funcs + m
                                        while row_idx >= len(P_full):
                                                P_full.append([0 for _ in range(num_nodes_all)])
                                        if 0 <= k_curr < num_nodes_all:
                                                P_full[row_idx][k_curr] = 1
                        for leaf in range(num_leaves):
                                H_full[leaf] = leaf_distribution(leaf)
                        return {'P': P_full, 'H': H_full, 'L': L_full, 'a': a_full, 'b': b_full}

                for leaf in range(num_leaves):
                        try:
                                pred_k_list, lat_upward_list, pos_list = upward_trajectory(leaf + num_other)
                        except Exception:
                                pred_k_list, lat_upward_list, pos_list = ([], [], [])
                        sel_work = 0
                        try:
                                Htop = top_place.get('H', [])
                                if Htop:
                                        sel_work = int(
                                                np.where(np.array(Htop[pred_k_list[1] - num_other_first_level]) == 1)[
                                                        0][0])
                        except Exception:
                                sel_work = 0

                        try:
                                vec = [0] * deployment_number
                                if 0 <= sel_work < deployment_number:
                                        vec[int(sel_work)] = 1
                                H_full[leaf] = vec
                        except Exception:
                                H_full[leaf] = [0 for _ in range(num_workflows)]


                        for m in range(num_funcs):
                                prev_selected_place = 0
                                opt_num = 0
                                curr_selected_workflow = sel_work
                                k_curr = 0
                                for r in range(num_levels):
                                        if r != 0:
                                                opt_num = find_opt(opt_num, prev_selected_place, r)
                                        placement_path = in_path("placement", f"opt_{w}_{opt_num}.json")
                                        try:
                                                curr_placement = load_json_cached(placement_path)
                                        except Exception:
                                                curr_placement = {}

                                        curr_P = curr_placement.get('P', [])
                                        curr_workflow = {}
                                        try:
                                                workflow_path = in_path("workflows", f"requirements_workflow_{w}_{opt_num}.json")
                                                curr_workflow = load_json_cached(workflow_path)
                                        except Exception:
                                                curr_workflow = {}

                                        if r == 0:

                                                try:
                                                        if curr_P:
                                                                selected_place = int(np.where(np.array(curr_P[
                                                                                                               curr_selected_workflow * num_funcs + m]) == 1)[
                                                                                             0][0])
                                                        else:
                                                                selected_place = 0
                                                except Exception:
                                                        selected_place = 0
                                        else:

                                                try:
                                                        if 'active_workflow_list' in curr_workflow and curr_workflow[
                                                                'active_workflow_list']:
                                                                curr_selected_workflow = find_correct_workflow(
                                                                        curr_selected_workflow,
                                                                        curr_workflow['active_workflow_list'])
                                                except Exception:
                                                        pass
                                                try:
                                                        if curr_P:
                                                                selected_place = int(np.where(np.array(curr_P[
                                                                                                               curr_selected_workflow * num_funcs + m]) == 1)[
                                                                                             0][0])
                                                        else:
                                                                selected_place = 0
                                                except Exception:
                                                        selected_place = 0


                                        if not L_full and 'L' in curr_placement:
                                                L_full = curr_placement.get('L', L_full)


                                        if r == num_levels - 1:
                                                try:
                                                        k_curr = get_leaf(selected_place, opt_num)
                                                except Exception:
                                                        k_curr = selected_place

                                        prev_selected_place = selected_place


                                try:
                                        row_idx = sel_work * num_funcs + m
                                except Exception:
                                        row_idx = m

                                if row_idx >= len(P_full):

                                        for _ in range(row_idx - len(P_full) + 1):
                                                P_full.append([0 for _ in range(num_nodes_all)])
                                if 0 <= k_curr < num_nodes_all:
                                        P_full[row_idx][k_curr] = 1

                return {'P': P_full, 'H': H_full, 'L': L_full, 'a': a_full, 'b': b_full}


        # ---- Precompute, for every (workflow w, leaf), the static execution
        # plan of the decomposed solution: which deployment the leaf uses,
        # on which physical node each function runs (selected_places), its
        # rounded execution duration (cloud speedup applied) and RAM demand.
        # This per-workflow table (perwf) drives the event simulation below.
        for w in range(num_workflows):
                wf_path = in_path("workflows", f"requirements_workflow_{w}_0.json")
                sol_path = in_path("placement", f"opt_{w}_full.json")
                node_path = in_path("cloud", f"requirements_cloud_full.json")
                try:
                        wf_conf = load_json_cached(wf_path)
                        node_conf = load_json_cached(node_path)
                except Exception:
                        wf_conf = {}
                        node_conf = {}
                sol_conf = {}
                try:
                        sol_conf = reconstruct_full_solution(w)
                except Exception:
                        sol_conf = {}
                P = sol_conf.get('P', [])
                H = sol_conf.get('H', [])
                L = sol_conf.get('L', [])
                Speed = sol_conf.get('Speed', None)
                a = sol_conf.get('a', [])
                b = sol_conf.get('b', [])
                num_funcs = wf_conf.get('num_functions', 0)
                perwf[w] = {}
                num_deployment = wf_conf.get('deployment_number', 1)
                num_paths_w = wf_conf.get('paths', {}).get('num_paths', 0)
                # Execution order of each branch (see branch_functions above).
                path_order = path_order_of(wf_conf)

                # Execution plan (nodes/durations/rams) of a given deployment d;
                # shared by the deterministic and stochastic paths.
                def build_plan(d):
                        selected_places = []
                        for m in range(num_funcs):
                                try:
                                        row = P[d * num_funcs + m]
                                        selected_places.append(int(np.where(np.array(row) == 1)[0][0]))
                                except Exception:
                                        selected_places.append(0)
                        durations = [0] * num_funcs
                        rams = [0] * num_funcs
                        for m in range(num_funcs):
                                node = selected_places[m]
                                node_name = node_conf["providers"][f"provider_{node}"]["name"]
                                try:
                                        func_conf = wf_conf['functions'][f"function_{m}"]
                                        base_time = func_conf.get('time', 0)
                                        if Speed:
                                                try:
                                                        speedup = float(np.array(Speed)[m][node])
                                                except Exception:
                                                        speedup = func_conf.get('speedup', 1)
                                        else:
                                                if node_name != "tinyfaas":
                                                        speedup = func_conf.get('speedup', 1)
                                                else:
                                                        speedup = 1
                                        dur = int(np.ceil(speedup * base_time))
                                        durations[m] = dur
                                        rams[m] = func_conf.get('ram', 0)
                                except Exception:
                                        durations[m] = 0
                                        rams[m] = 0
                        return selected_places, durations, rams

                def plan_base_T(sel_places, durs, lf):
                        base_T_paths = []
                        for s in range(num_paths_w):
                                func_nums = branch_functions(b, s, path_order)
                                tot = 0
                                for idx_m in func_nums:
                                        tot += durs[idx_m]
                                        pos = func_nums.index(idx_m)
                                        next_place = sel_places[func_nums[pos + 1]] if pos < len(func_nums) - 1 else lf + num_other
                                        try:
                                                tot += int(np.ceil(L[sel_places[idx_m]][next_place]))
                                        except Exception:
                                                pass
                                base_T_paths.append(tot)
                        return max(base_T_paths) if base_T_paths else 0

                for leaf in range(num_leaves):
                        if stochastic_selection:
                                try:
                                        dist = [max(0.0, float(x)) for x in H[leaf]]
                                except Exception:
                                        dist = [1.0] + [0.0] * (num_deployment - 1)
                                tot_p = sum(dist)
                                dist = [x / tot_p for x in dist] if tot_p > 0 else [1.0 / num_deployment] * num_deployment
                                plans = {}
                                for d in range(num_deployment):
                                        if dist[d] <= 0:
                                                continue
                                        sp, du, rm = build_plan(d)
                                        plans[d] = {
                                                'selected_places': sp,
                                                'durations': du,
                                                'rams': rm,
                                                'base_T': plan_base_T(sp, du, leaf),
                                        }
                                perwf[w][leaf] = {'dist': dist, 'plans': plans, 'L': L}
                        else:
                                sel_work = 0
                                try:
                                        sel_work = int(np.where(np.array(H[leaf]) == 1)[0][0]) if H else 0
                                except Exception:
                                        sel_work = 0
                                selected_places, durations, rams = build_plan(sel_work)
                                perwf[w][leaf] = {
                                        'selected_places': selected_places,
                                        'deployment': int(sel_work),
                                        'durations': durations,
                                        'rams': rams,
                                        'L': L
                                }

                perwf[w]['a'] = a
                perwf[w]['b'] = b
                perwf[w]['num_paths'] = num_paths_w
                perwf[w]['path_order'] = path_order

        # ---- Discrete-event simulation state (decomposed solution):
        # node_active[i] holds the currently running executions on node i as
        # tuples (end_time, ram, deployment, function_idx[, req_id], workflow);
        # waiting_acc accumulates the waiting time W per (workflow, leaf);
        # per_node_util / ram_area record the RAM occupancy over time.
        curr_utilization = np.zeros((num_nodes_global, 1))

        node_active = [[] for _ in range(num_nodes_global)]

        ram_area = [0.0] * num_nodes_global

        waiting_acc = np.zeros((num_workflows, num_leaves))


        per_node_util = np.zeros((sim_time, num_nodes_global), dtype=float)


        request_logs = []
        req_id_counter = 0
        req_done_list = []

        rng = np.random.default_rng(simulation_seed)
        # Stage 2 stochastic decomposition: separate RNG for per-request deployment
        # sampling (keeps the Poisson stream identical to the deterministic run),
        # and per-(workflow, leaf, deployment) counters for the Monte-Carlo cost.
        rng_sel = np.random.default_rng(simulation_seed + 81)
        sel_counts = [[{} for _ in range(num_leaves)] for _ in range(num_workflows)]

        # Simulate second by second: sample Poisson arrivals for every
        # (workflow, leaf) pair, then execute them under the two blocking
        # rules (RAM capacity and one-request-per-function-deployment).
        # Two scheduling variants exist: "lazy_start" keeps a pending queue and
        # advances requests step-wise in real time; the default variant
        # resolves each request immediately by searching the next free slot
        # (jumping curr_time to the earliest finishing execution if blocked).
        for cur_t in range(sim_time):

                requests[cur_t] = rng.poisson(request_rates)
                lazy_start = parameter_general_config.get('lazy_start', False)
                if lazy_start:

                        for ni in range(num_nodes_global):
                                node_active[ni] = [entry for entry in node_active[ni] if entry[0] > cur_t]


                        if cur_t == 0:
                                pending_requests = []


                        for w in range(num_workflows):
                                for leaf in range(num_leaves):
                                        n = int(requests[cur_t, w, leaf])
                                        if n <= 0:
                                                continue
                                        info = perwf.get(w, {}).get(leaf, None)
                                        if info is None:
                                                continue
                                        sel_places = info['selected_places']
                                        durs = info['durations']
                                        rams = info['rams']
                                        Lmat = info['L']
                                        deployment_id = int(info.get('deployment', w))

                                        # base_T = ideal (blocking-free) evaluation time of the
                                        # workflow: the slowest branch's sum of function durations
                                        # plus inter-node latencies (= T_n of the placement model).
                                        # Waiting time is measured as delay beyond base_T.
                                        base_T_paths = []
                                        a_mat = perwf[w].get('a', [])
                                        b_mat = perwf[w].get('b', [])
                                        num_paths_local = perwf[w].get('num_paths', 0)
                                        for s in range(num_paths_local):
                                                func_nums = branch_functions(b_mat, s, perwf[w].get('path_order'))

                                                tot = 0
                                                for idx_m in func_nums:
                                                        tot += durs[idx_m]

                                                        # func_nums holds function IDs, so the successor
                                                        # test must compare POSITIONS in that list, not
                                                        # the function id against the list length.
                                                        pos = func_nums.index(idx_m)
                                                        if pos < len(func_nums) - 1:
                                                                next_m = func_nums[pos + 1]
                                                                next_place = sel_places[next_m]
                                                        else:
                                                                next_place = leaf + num_other
                                                        try:
                                                                tot += int(np.ceil(Lmat[sel_places[idx_m]][next_place]))
                                                        except Exception:
                                                                pass
                                                base_T_paths.append(tot)
                                        base_T = max(base_T_paths) if base_T_paths else 0

                                        # Enqueue n new requests; they progress function by
                                        # function (next_idx) in later time steps.
                                        for i in range(n):
                                                req = {
                                                        'id': int(req_id_counter),
                                                        'workflow': int(w),
                                                        'leaf': int(leaf),
                                                        'arrival': int(cur_t),
                                                        'deployment': int(info.get('deployment', w)),
                                                        'sel_places': sel_places,
                                                        'durs': durs,
                                                        'rams': rams,
                                                        'L': Lmat,
                                                        'base_T': int(base_T),
                                                        'next_idx': 0,
                                                        'start_times': [],
                                                        'end_times': []
                                                }
                                                pending_requests.append(req)
                                                req_id_counter += 1


                        # Advance every pending request by at most one function start
                        # per time step. Finished requests (all functions done) are
                        # logged with their waiting time wait = makespan - base_T.
                        still_pending = []
                        for req in pending_requests:
                                w = req['workflow']
                                leaf = req['leaf']
                                sel_places = req['sel_places']
                                durs = req['durs']
                                rams = req['rams']
                                Lmat = req['L']
                                idx = req['next_idx']
                                num_funcs = len(durs)

                                if idx >= num_funcs:

                                        if req['end_times']:
                                                last_node = sel_places[-1]
                                                last_end = req['end_times'][-1]
                                                try:
                                                        final_transfer = int(np.ceil(Lmat[last_node][leaf + num_other]))
                                                except Exception:
                                                        final_transfer = 0
                                                req_done = last_end + final_transfer
                                        else:
                                                req_done = req['arrival']
                                        # req_done_list collects RESPONSE TIMES (the w_2 term of the
                                        # overall cost is sum(req_done_list)), so the arrival time has
                                        # to be subtracted here exactly as in the default branch below.
                                        req_done_list.append(req_done - req['arrival'])
                                        wait = max(0, req_done - req['arrival'] - req['base_T'])
                                        waiting_acc[w, leaf] += wait

                                        request_logs.append({
                                                'id': req['id'],
                                                'workflow': w,
                                                'leaf': leaf,
                                                'arrival': req['arrival'],
                                                'start': req['start_times'][0] if req['start_times'] else req[
                                                        'arrival'],
                                                'end': req_done,
                                                'base_T': req['base_T'],
                                                'wait': int(wait)
                                        })
                                        continue


                                # The next function may only start once the previous one
                                # has finished and its output has been transferred
                                # (predecessor end time + latency must have passed).
                                ready = False
                                if idx == 0:
                                        ready = True
                                else:
                                        if len(req['end_times']) >= idx:
                                                prev_end = req['end_times'][idx - 1]
                                                prev_node = sel_places[idx - 1]
                                                try:
                                                        next_place = sel_places[idx]
                                                except Exception:
                                                        next_place = leaf + num_other
                                                try:
                                                        transfer = int(np.ceil(Lmat[prev_node][next_place]))
                                                except Exception:
                                                        transfer = 0
                                                if prev_end + transfer <= cur_t:
                                                        ready = True

                                if not ready:
                                        still_pending.append(req)
                                        continue


                                # Blocking check (Sec. IV-A): start the function only if
                                # the node has enough free RAM and this function of this
                                # deployment is not already serving another request.
                                node = sel_places[idx]
                                dur = durs[idx]
                                ram_req = rams[idx]
                                used = sum(entry[1] for entry in node_active[node] if entry[0] > cur_t)
                                deployment_id = req.get('deployment', w)
                                func_busy = any(
                                        len(entry) >= 4 and entry[0] > cur_t and entry[2] == deployment_id and entry[3] == idx for
                                        entry in node_active[node])
                                if used + ram_req <= node_ram_max[node] and (not func_busy):

                                        start = cur_t
                                        endt = start + dur
                                        node_active[node].append((endt, ram_req, deployment_id, idx, req['id'],w))
                                        req['start_times'].append(start)
                                        req['end_times'].append(endt)
                                        req['next_idx'] += 1

                                        if req['next_idx'] < num_funcs:
                                                still_pending.append(req)
                                        else:

                                                still_pending.append(req)
                                else:
                                        still_pending.append(req)

                        pending_requests = still_pending


                        # Record the RAM occupied on every node in this time step.
                        for ni in range(num_nodes_global):
                                used_now = sum(entry[1] for entry in node_active[ni] if entry[0] > cur_t)
                                ram_area[ni] += used_now
                                per_node_util[cur_t, ni] = used_now
                else:

                        for ni in range(num_nodes_global):
                                node_active[ni] = [entry for entry in node_active[ni] if entry[0] > cur_t]
                        for w in range(num_workflows):
                                for leaf in range(num_leaves):
                                        n = int(requests[cur_t, w, leaf])
                                        if n <= 0:
                                                continue
                                        info = perwf.get(w, {}).get(leaf, None)
                                        if info is None:
                                                continue
                                        Lmat = info['L']
                                        a_mat = perwf[w].get('a', [])
                                        b_mat = perwf[w].get('b', [])
                                        num_paths_local = perwf[w].get('num_paths', 0)
                                        if not stochastic_selection:
                                                sel_places = info['selected_places']
                                                durs = info['durations']
                                                rams = info['rams']
                                                deployment_id = int(info.get('deployment', w))
                                                base_T_paths = []
                                                for s in range(num_paths_local):
                                                        func_nums = branch_functions(b_mat, s, perwf[w].get('path_order'))
                                                        tot = 0
                                                        for idx_m in func_nums:
                                                                tot += durs[idx_m]
                                                                # Positional successor (see plan_base_T);
                                                                # comparing the function id idx_m against
                                                                # len(func_nums) is wrong on branchy DAGs.
                                                                pos = func_nums.index(idx_m)
                                                                if pos < len(func_nums) - 1:
                                                                        next_m = func_nums[pos + 1]
                                                                        next_place = sel_places[next_m]
                                                                else:
                                                                        next_place = leaf + num_other
                                                                try:
                                                                        tot += int(np.ceil(Lmat[sel_places[idx_m]][next_place]))
                                                                except Exception:
                                                                        pass
                                                        base_T_paths.append(tot)
                                                base_T = max(base_T_paths) if base_T_paths else 0

                                        # Resolve each arriving request immediately: walk every
                                        # branch function-by-function, booking each execution at
                                        # the earliest time slot that satisfies the RAM and
                                        # function-busy blocking rules.
                                        for i in range(n):
                                                if stochastic_selection:
                                                        # Sample this request's deployment from the leaf
                                                        # policy and use that deployment's plan/base_T.
                                                        dist = info['dist']
                                                        d_sel = int(rng_sel.choice(len(dist), p=dist))
                                                        plan = info['plans'].get(d_sel)
                                                        if plan is None:
                                                                continue
                                                        sel_places = plan['selected_places']
                                                        durs = plan['durations']
                                                        rams = plan['rams']
                                                        base_T = plan['base_T']
                                                        deployment_id = d_sel
                                                        sel_counts[w][leaf][d_sel] = sel_counts[w][leaf].get(d_sel, 0) + 1
                                                arrival = cur_t
                                                path_finish_times = []

                                                a_mat = perwf[w].get('a', [])
                                                b_mat = perwf[w].get('b', [])
                                                num_paths_local = perwf[w].get('num_paths', 0)
                                                for s in range(num_paths_local):
                                                        func_nums = branch_functions(b_mat, s, perwf[w].get('path_order'))

                                                        curr_time = arrival
                                                        for idx_m in func_nums:
                                                                node = sel_places[idx_m]
                                                                dur = durs[idx_m]
                                                                ram_req = rams[idx_m]

                                                                # Find the earliest start time: if blocked, jump
                                                                # to the next finishing execution on the node and
                                                                # re-check (this waiting is the simulated cost of
                                                                # the blocking mechanisms).
                                                                while True:

                                                                        # An entry occupies the node only while it is
                                                                        # actually running: [entry[5], entry[0]). Testing
                                                                        # end > curr_time alone would also count bookings
                                                                        # that START later (this loop books far into the
                                                                        # future), which invents blocking that never
                                                                        # happens. Compare against the candidate slot
                                                                        # [curr_time, curr_time + dur).
                                                                        slot_end = curr_time + dur
                                                                        overlapping = [entry for entry in node_active[node]
                                                                                       if entry[5] < slot_end and curr_time < entry[0]]
                                                                        used = sum(entry[1] for entry in overlapping)

                                                                        func_busy_ends = [entry[0] for entry in overlapping
                                                                                          if entry[2] == deployment_id
                                                                                          and entry[3] == idx_m
                                                                                          and entry[4] == w]

                                                                        if used + ram_req <= node_ram_max[node] and (
                                                                        not func_busy_ends):

                                                                                start = curr_time
                                                                                break

                                                                        # Capacity and the function deployment are only
                                                                        # released by an execution ending, so the next
                                                                        # candidate start is the earliest future end.
                                                                        future_ends = [entry[0] for entry in
                                                                                       node_active[node] if
                                                                                       entry[0] > curr_time]
                                                                        if future_ends:
                                                                                curr_time = int(min(future_ends))
                                                                        else:
                                                                                start = curr_time
                                                                                break
                                                                endt = start + dur

                                                                node_active[node].append((endt, ram_req, deployment_id, idx_m, w, start))

                                                                # Move on to the successor: execution end plus the
                                                                # transfer latency to the next function's node (or
                                                                # back to the requesting leaf after the last one).
                                                                idx_pos = func_nums.index(idx_m)
                                                                if idx_pos < len(func_nums) - 1:
                                                                        next_m = func_nums[idx_pos + 1]
                                                                        next_place = sel_places[next_m]
                                                                else:
                                                                        next_place = leaf + num_other
                                                                try:
                                                                        transfer = int(np.ceil(Lmat[node][next_place]))
                                                                except Exception:
                                                                        transfer = 0
                                                                curr_time = endt + transfer
                                                        path_finish_times.append(curr_time)
                                                # Request completes when its slowest branch returns;
                                                # everything beyond base_T counts as waiting time W.
                                                req_done = max(path_finish_times) if path_finish_times else arrival
                                                req_done_list.append(req_done-arrival)
                                                wait = (req_done - arrival) - base_T
                                                if wait < 0:
                                                        wait = 0

                                                waiting_acc[w, leaf] += wait

                                                req_log = {
                                                        'id': int(req_id_counter),
                                                        'workflow': int(w),
                                                        'leaf': int(leaf),
                                                        'arrival': int(arrival),
                                                        'start': int(arrival + wait),
                                                        'end': int(req_done),
                                                        'base_T': int(base_T),
                                                        'wait': int(wait)
                                                }
                                                request_logs.append(req_log)
                                                req_id_counter += 1

                        for ni in range(num_nodes_global):

                                # RAM in use at cur_t = executions already started and
                                # not yet finished; bookings placed further in the
                                # future do not occupy the node now.
                                used_now = sum(entry[1] for entry in node_active[ni]
                                               if entry[5] <= cur_t < entry[0])
                                ram_area[ni] += used_now
                                per_node_util[cur_t, ni] = used_now

        # ---- Monetary/time cost evaluation of the decomposed solution: weight
        # the per-request costs by the number of requests actually sampled.
        overall_requests = np.zeros((num_workflows, num_leaves))
        for t in range(num_workflows):
                for k in range(num_leaves):
                        overall_requests[t, k] = sum(requests[:, t, k])
        # For every workflow t and leaf k, recompute the exact monetary cost
        # (execution C, data transfer D, dependency storage transfer, input
        # data D_in) and the branch times of the decomposed placement by
        # following each function down the hierarchy, exactly mirroring the
        # cost terms of Eq. (1). Latencies across regions are composed with
        # upward_trajectory (Algorithm 1).
        select_cost_all=0
        for t in range(num_workflows):
                select_cost=0
                workflow_path=in_path("workflows", f"requirements_workflow_{t}_0.json")
                placement_path=in_path("placement", f"opt_{t}_0.json")
                workflow_config = load_json_cached(workflow_path)
                placement_config = load_json_cached(placement_path)
                num_functions=workflow_config["num_functions"]
                H=placement_config['H']
                T=placement_config['T']
                P=placement_config['P']
                C=placement_config['C']
                D=placement_config['D']
                D_in=placement_config['D_in']
                L=placement_config['L']
                S=placement_config['S']
                num_paths= workflow_config['paths']['num_paths']
                # Execution order of each branch (see branch_functions).
                path_order = path_order_of(workflow_config)
                b = placement_config['b']
                a = placement_config['a']
                num_deployment=workflow_config["deployment_number"]
                workflow_overall=np.zeros((num_deployment))
                num_nodes_all=sum(num_nodes_in_level[-1])
                invocations=np.zeros(num_nodes_all)
                placement_overall=0
                # Monetary cost per (leaf, deployment): deterministic weights the
                # top-level regional selection by all requests at the leaf; stochastic
                # weights each deployment by how many requests actually sampled it
                # (Monte-Carlo expectation over the routing policy).
                if stochastic_selection:
                        work_items = [(k, d, c) for k in range(num_leaves) for d, c in sel_counts[t][k].items()]
                else:
                        work_items = []
                        for k in range(num_leaves):
                                _pkl, _lu, _pl = upward_trajectory(k + num_other)
                                work_items.append((k, H[_pkl[1] - num_other_first_level].index(1), overall_requests[t, k]))
                for k, selected_workflow, weight in work_items:

                        # Which deployment serves this leaf: read the selection H of
                        # the top-level problem for the leaf's level-1 region.
                        pred_k_list, lat_upward_list,pos_list = upward_trajectory(k+num_other)

                        workflow_overall[selected_workflow]+=1
                        time_cost=0
                        money_cost=0

                        T_paths= np.zeros((num_paths,1))

                        D_func = np.zeros((num_functions, 1))
                        C_func = np.zeros((num_functions, 1))
                        placement_func = np.zeros((num_functions,num_nodes_all))
                        for s in range(num_paths):

                                pred_k_list, lat_upward_list, pos_list = upward_trajectory(k + num_other)
                                L_func = np.zeros((num_functions, 1))

                                func_nums = branch_functions(b, s, path_order)

                                # Trace function m through all levels of the decomposition
                                # (r=0 is the top problem); costs/latencies are accumulated
                                # per level, the physical execution node is reached at the
                                # last level.
                                for m in func_nums:
                                        prev_selected_place = 0
                                        opt_num=0
                                        for r in range(num_levels):
                                                if r!=0:
                                                        opt_num=find_opt(opt_num,prev_selected_place,r)
                                                placement_path=in_path("placement", f"opt_{t}_{opt_num}.json")
                                                workflow_path = in_path("workflows", f"requirements_workflow_{t}_{opt_num}.json")
                                                cloud_path = in_path("cloud", f"requirements_cloud_{r}_{opt_num-ind_start_list[r]}.json")
                                                curr_placement = load_json_cached(placement_path)
                                                curr_cloud = load_json_cached(cloud_path)
                                                curr_workflow = load_json_cached(workflow_path)

                                                curr_P=curr_placement['P']
                                                # Region latency from the (perturbable) cloud config;
                                                # opt-file L kept only as a legacy fallback.
                                                curr_L = curr_cloud.get('L', curr_placement['L'])
                                                if r==0:

                                                        curr_selected_workflow = selected_workflow
                                                        try:
                                                                selected_place = P[selected_workflow * num_functions + m].index(1)
                                                        except (ValueError, IndexError):
                                                                selected_place = 0
                                                else:

                                                        try:
                                                                curr_selected_workflow=find_correct_workflow(curr_selected_workflow,curr_workflow["active_workflow_list"])
                                                        except (ValueError, IndexError):
                                                                pass
                                                if curr_P==[]:
                                                        selected_place=0
                                                else:
                                                        # A sampled deployment whose function m is placed in
                                                        # another region appears here only virtually (pinned to
                                                        # the head node 0); default there if no explicit 1.
                                                        try:
                                                                selected_place=curr_P[curr_selected_workflow*num_functions+m].index(1)
                                                        except (ValueError, IndexError):
                                                                selected_place=0

                                                if r==0:
                                                        # Top level: dependency-data transfer cost (if the
                                                        # function does not run at the provider storing its
                                                        # data), latency from the user's region, input-data
                                                        # cost D_in for the first function and output-transfer
                                                        # cost D towards the successor's node.
                                                        data_dependencies = curr_workflow['functions'][f"function_{m}"][
                                                                'data_dependencies']
                                                        for key in data_dependencies.keys():
                                                                value = data_dependencies[key]
                                                                pricing_Storage_Transfer = provider_cost_list[key]["pricing_Storage_Transfer"]
                                                                if key != curr_cloud["providers"][f"provider_{selected_place}"][
                                                                                "name"]:

                                                                        C_func[m] = C_func[m] + pricing_Storage_Transfer * value
                                                        L_func[m] += curr_L[pred_k_list[1]][selected_place]
                                                        if m==0:
                                                                D_in_0=0
                                                                if pred_k_list[1] != selected_place:
                                                                        pricing_data_sent = curr_cloud['providers'][f"provider_{selected_place}"]['pricing_data_sent']
                                                                        D_in_0=curr_workflow['input_data'] * pricing_data_sent

                                                        # m is a function ID; func_nums must be indexed by
                                                        # POSITION, and the end-of-branch test compares
                                                        # positions too. Getting either wrong makes D_func
                                                        # bill the transfer to the wrong successor node on
                                                        # branches that are not a 0..k prefix.
                                                        pos = func_nums.index(m)
                                                        if pos < len(func_nums) - 1:

                                                                next_selected_place = curr_P[
                                                                        selected_workflow * num_functions + func_nums[
                                                                                pos + 1]].index(1)
                                                        else:
                                                                next_selected_place = pred_k_list[1]
                                                        D_func[m]=D[m][selected_place][next_selected_place]

                                                else:
                                                        # Lower levels: if the user's trajectory passes through
                                                        # the same region, add the intra-region latency;
                                                        # otherwise route via the head nodes (latency to the
                                                        # own head node plus the user's upward latency).
                                                        if pred_k_list[r]==opt_num-ind_start_list[r]:

                                                                curr_placement = load_json_cached(placement_path)


                                                                curr_L = curr_cloud.get('L', curr_placement['L'])
                                                                L_func[m] += curr_L[pos_list[r]][selected_place]
                                                        else:
                                                                L_func[m] += curr_L[0][selected_place]
                                                                L_func[m] += lat_upward_list[r]


                                                # Physical level reached: add execution time (cloud
                                                # speedup applied) and RAM-seconds execution cost on
                                                # the actually chosen node.
                                                if r == num_levels-1:
                                                        if curr_cloud['providers'][f"provider_{selected_place}"][
                                                                "name"] == "tinyfaas":
                                                                speedup = 1
                                                        else:
                                                                speedup = curr_workflow["functions"][f"function_{m}"][
                                                                        'speedup']

                                                        ram = curr_workflow['functions'][f"function_{m}"]['ram']
                                                        time = curr_workflow['functions'][f"function_{m}"]['time']
                                                        T_func = speedup * time

                                                        pricing_RAM=curr_cloud['providers'][f"provider_{selected_place}"]['pricing_RAM']
                                                        C_func[m] += pricing_RAM*ram*time
                                                        k_curr=get_leaf(selected_place,opt_num)
                                                        placement_func[m,k_curr]=1

                                                        T_paths[s]+=(T_func+L_func[m])
                                                        invocations[k_curr]+=1

                                                prev_selected_place = selected_place
                                        pred_k_list,lat_upward_list,pos_list=upward_trajectory(k_curr)
                                # Latency for returning the branch's result from the last
                                # function's node back to the requesting leaf, composed
                                # across the hierarchy as in Algorithm 1.
                                L_end=0
                                pred_k_list_new, lat_upward_list_new, pos_list_new = upward_trajectory(k+num_other)
                                for r in range(num_levels-1,-1,-1):

                                        if pred_k_list[r] == pred_k_list_new[r]:
                                                opt_num=pred_k_list[r]+ind_start_list[r]
                                                placement_path = in_path("placement", f"opt_{t}_{opt_num}.json")
                                                curr_placement = load_json_cached(placement_path)
                                                # Region latency from the (perturbable) cloud config
                                                # (opt_num - ind_start_list[r] == pred_k_list[r]);
                                                # opt-file L kept only as a legacy fallback.
                                                cloud_path = in_path("cloud", f"requirements_cloud_{r}_{pred_k_list[r]}.json")
                                                curr_cloud = load_json_cached(cloud_path)
                                                curr_L = curr_cloud.get('L', curr_placement['L'])
                                                L_end += curr_L[pos_list[r]][pos_list_new[r]]
                                        else:
                                                L_end += lat_upward_list_new[r]
                                                L_end += lat_upward_list[r]
                                T_paths[s]+=L_end

                        # Per-request costs: time = slowest branch, money = execution +
                        # data transfer + input data; weighted by the simulated number
                        # of requests of this (workflow, leaf) pair.
                        for r in range(num_nodes_all):
                                placement_overall+=sum(placement_func[:,r])**2
                        time_cost+=np.max(T_paths)
                        money_cost+=np.sum(C_func)+np.sum(D_func)+D_in_0




                        time_cost_all+=weight*time_cost

                        money_cost_all+=weight*money_cost
                select_cost=np.sum(workflow_overall**2)+np.sum(invocations**2)
                select_cost_all+=select_cost

        # Overall simulated cost C = w_1*M + w_2*(B+W): req_done_list holds the
        # complete per-request response times (basic time + waiting).
        overall_cost = w_1 * money_cost_all+ w_2 * sum(req_done_list)
        print(sum(sum(waiting_acc)))
        print([overall_cost])
        # Results go under the scenario folder in sensitivity mode, else the
        # baseline results/run_<N>/ folder (unchanged default behavior).
        results_dir = os.path.join(scenario_dir, "results") if scenario_dir else os.path.join("results", f"run_{num_run}")
        os.makedirs(results_dir, exist_ok=True)
        results_path = os.path.join(results_dir, "results.json")
        results={}
        results['overall_cost']=overall_cost
        results['money_cost']=money_cost_all
        results['complete_time_cost'] = sum(sum(waiting_acc))
        results['time_cost']=time_cost_all

        # ================= Comparison run =================
        # Repeat the identical simulation (same RNG seeds, hence the same
        # request streams) for the centralized/heuristic solution stored in
        # opt_<w>_full.json. The code mirrors the decomposed evaluation above,
        # but P/H/L can be used directly since the comparison was solved on
        # the flattened architecture - no hierarchy reconstruction is needed
        # (reconstruct_full_solution only serves as fallback).
        if comparison:
                np.random.seed(simulation_seed)
                sim_time=parameter_general_config["simulation_time"]
                time_cost_all=0
                money_cost_all=0
                select_cost_all=0
                time_cost=0
                money_cost=0
                request_rates=np.zeros((num_workflows,num_leaves))
                for t in range(num_workflows):
                        cloud_name = f"requirements_cloud_full.json"
                        path_cloud = in_path("cloud", cloud_name)
                        cloud_config = load_json_cached(path_cloud)
                        for k in range(num_leaves):
                                request_rates[t,k]=cloud_config["providers"][f"provider_{k+num_other}"]["request_rate"]
                requests=np.zeros((sim_time,num_workflows,num_leaves))
                cloud_full_path = in_path("cloud", "requirements_cloud_full.json")
                cloud_full = load_json_cached(cloud_full_path)
                num_nodes_global = cloud_full.get('num_nodes', 0)
                node_ram_max = [cloud_full['providers'][f'provider_{i}'].get('max_RAM_curr', 1e9) if cloud_full['providers'][f'provider_{i}'].get('name')=="tinyfaas" else np.inf for i in
                        range(num_nodes_global)]

                perwf = {}

                def reconstruct_full_solution(w):

                        wf_path = in_path("workflows", f"requirements_workflow_{w}_0.json")
                        try:
                                wf_conf = load_json_cached(wf_path)
                        except Exception:
                                wf_conf = {}
                        num_funcs = wf_conf.get('num_functions', 0)

                        num_nodes_all = cloud_full.get('num_nodes', 0)

                        P_full = [[0 for _ in range(num_nodes_all)] for _ in range(num_funcs)]

                        H_full = [[0] for _ in range(num_leaves)]


                        L_full = []
                        full_path = in_path("placement", f"opt_{w}_full.json")
                        try:
                                full_place_tmp = load_json_cached(full_path)
                                if 'L' in full_place_tmp:
                                        L_full = full_place_tmp.get('L', [])
                        except Exception:
                                pass
                        # Prefer the (perturbable) cloud-config L; the opt-file
                        # read above is only a backward-compatibility fallback.
                        L_full = cloud_full.get('L', L_full)


                        top_path = in_path("placement", f"opt_{w}_0.json")
                        try:
                                top_place = load_json_cached(top_path)
                        except Exception:
                                top_place = {}

                        for leaf in range(num_leaves):

                                try:
                                        pred_k_list, lat_upward_list, pos_list = upward_trajectory(leaf + num_other)
                                except Exception:
                                        pred_k_list, lat_upward_list, pos_list = ([], [], [])
                                sel_work = 0
                                try:
                                        Htop = top_place.get('H', [])
                                        if Htop:
                                                sel_work = int(np.where(np.array(Htop[pred_k_list[1] - num_other_first_level]) == 1)[0][0])
                                except Exception:
                                        sel_work = 0

                                try:
                                        H_full[leaf] = [int(sel_work)]
                                except Exception:
                                        H_full[leaf] = [0]

                                for m in range(num_funcs):
                                        prev_selected_place = 0
                                        opt_num = 0
                                        curr_selected_workflow = sel_work
                                        k_curr = 0
                                        for r in range(num_levels):
                                                if r != 0:
                                                        opt_num = find_opt(opt_num, prev_selected_place, r)
                                                placement_path = in_path("placement", f"opt_{w}_{opt_num}.json")
                                                try:
                                                        curr_placement = load_json_cached(placement_path)
                                                except Exception:
                                                        curr_placement = {}

                                                curr_P = curr_placement.get('P', [])
                                                curr_workflow = {}
                                                try:
                                                        workflow_path = in_path("workflows", f"requirements_workflow_{w}_{opt_num}.json")
                                                        curr_workflow = load_json_cached(workflow_path)
                                                except Exception:
                                                        curr_workflow = {}

                                                if r == 0:

                                                        try:
                                                                if curr_P:
                                                                        selected_place = int(np.where(np.array(curr_P[curr_selected_workflow * num_funcs + m]) == 1)[0][0])
                                                                else:
                                                                        selected_place = 0
                                                        except Exception:
                                                                selected_place = 0
                                                else:

                                                        try:
                                                                if 'active_workflow_list' in curr_workflow and curr_workflow['active_workflow_list']:
                                                                        curr_selected_workflow = find_correct_workflow(curr_selected_workflow, curr_workflow['active_workflow_list'])
                                                        except Exception:
                                                                pass
                                                        try:
                                                                if curr_P:
                                                                        selected_place = int(np.where(np.array(curr_P[curr_selected_workflow * num_funcs + m]) == 1)[0][0])
                                                                else:
                                                                        selected_place = 0
                                                        except Exception:
                                                                selected_place = 0


                                                if not L_full and 'L' in curr_placement:
                                                        L_full = curr_placement.get('L', L_full)

                                                if r == num_levels - 1:
                                                        try:
                                                                k_curr = get_leaf(selected_place, opt_num)
                                                        except Exception:
                                                                k_curr = selected_place

                                                prev_selected_place = selected_place

                                        try:
                                                row_idx = sel_work * num_funcs + m
                                        except Exception:
                                                row_idx = m

                                        if row_idx >= len(P_full):

                                                for _ in range(row_idx - len(P_full) + 1):
                                                        P_full.append([0 for _ in range(num_nodes_all)])
                                        if 0 <= k_curr < num_nodes_all:
                                                P_full[row_idx][k_curr] = 1

                        return {'P': P_full, 'H': H_full, 'L': L_full}
                for w in range(num_workflows):
                        wf_path = in_path("workflows", f"requirements_workflow_{w}_0.json")
                        sol_path = in_path("placement", f"opt_{w}_full.json")
                        node_path = in_path("cloud", f"requirements_cloud_full.json")
                        try:
                                wf_conf = load_json_cached(wf_path)
                                node_conf = load_json_cached(node_path)
                        except Exception:
                                wf_conf = {}
                                node_conf = {}

                        # The comparison solution is read directly from the full
                        # (centralized/heuristic) placement file.
                        decomposed_mode = False
                        sol_conf = {}
                        if decomposed_mode:
                                try:
                                        sol_conf = reconstruct_full_solution(w)
                                except Exception:
                                        sol_conf = {}
                        else:
                                try:
                                        sol_conf = load_json_cached(sol_path)
                                except Exception:
                                        sol_conf = {}

                                if not sol_conf.get('P'):
                                        try:
                                                sol_conf = reconstruct_full_solution(w)
                                        except Exception:
                                                sol_conf = sol_conf or {}
                        P = sol_conf.get('P', [])
                        H = sol_conf.get('H', [])
                        # Latency from the (perturbable) full cloud config; fall
                        # back to the placement solution L for legacy runs.
                        L = cloud_full.get('L', sol_conf.get('L', []))
                        # Derive the execution speedup from the (perturbable) workflow
                        # 'speedup' field, like the decomposed path, instead of the
                        # opt_<t>_full.json 'Speed' matrix. That matrix freezes the
                        # baseline speedup (it is built from the workflow field: the
                        # workflow speedup on cloud nodes, 1 on tinyfaas), so reading
                        # it would make the comparison blind to speedup perturbation.
                        # Using None here is baseline-preserving and makes the
                        # comparison respond to a perturbed 'speedup'.
                        Speed = None
                        a = sol_conf.get('a', [])
                        b = sol_conf.get('b', [])
                        # Count the shared start/finish functions in only one branch so
                        # they are not executed once per branch in the simulation.
                        if len(b) > 1:
                                # Copy before zeroing: sol_conf may come from the shared
                                # JSON cache and must stay unmodified.
                                b = [list(row) for row in b]
                                for r in range(1, len(b)):
                                        b[r][0] = 0
                                        b[r][-1] = 0
                        num_funcs = wf_conf.get('num_functions', 0)
                        perwf[w] = {}
                        num_deployment = wf_conf.get('deployment_number', 1)
                        num_paths_w = wf_conf.get('paths', {}).get('num_paths', 0)
                        # Execution order of each branch (see branch_functions above).
                        path_order = path_order_of(wf_conf)

                        # Build the execution plan (nodes/durations/rams) of a given
                        # deployment d for this workflow. Shared by the deterministic
                        # and stochastic paths so both compute durations identically.
                        def build_plan(d):
                                selected_places = []
                                for m in range(num_funcs):
                                        try:
                                                row = P[d * num_funcs + m]
                                                selected_places.append(int(np.where(np.array(row) == 1)[0][0]))
                                        except Exception:
                                                selected_places.append(0)
                                durations = [0] * num_funcs
                                rams = [0] * num_funcs
                                for m in range(num_funcs):
                                        node = selected_places[m]
                                        node_name = node_conf["providers"][f"provider_{node}"]["name"]
                                        try:
                                                func_conf = wf_conf['functions'][f"function_{m}"]
                                                base_time = func_conf.get('time', 0)
                                                if Speed:
                                                        try:
                                                                speedup = float(np.array(Speed)[m][node])
                                                        except Exception:
                                                                speedup = func_conf.get('speedup', 1)
                                                else:
                                                        if node_name != "tinyfaas":
                                                                speedup = func_conf.get('speedup', 1)
                                                        else:
                                                                speedup = 1
                                                dur = int(np.ceil(speedup * base_time))
                                                durations[m] = dur
                                                rams[m] = func_conf.get('ram', 0)
                                        except Exception:
                                                durations[m] = 0
                                                rams[m] = 0
                                return selected_places, durations, rams

                        # Ideal (blocking-free) branch time of a plan from leaf `lf`.
                        def plan_base_T(sel_places, durs, lf):
                                base_T_paths = []
                                for s in range(num_paths_w):
                                        func_nums = branch_functions(b, s, path_order)
                                        tot = 0
                                        for idx_m in func_nums:
                                                tot += durs[idx_m]
                                                pos = func_nums.index(idx_m)
                                                next_place = sel_places[func_nums[pos + 1]] if pos < len(func_nums) - 1 else lf + num_other
                                                try:
                                                        tot += int(np.ceil(L[sel_places[idx_m]][next_place]))
                                                except Exception:
                                                        pass
                                        base_T_paths.append(tot)
                                return max(base_T_paths) if base_T_paths else 0

                        for leaf in range(num_leaves):
                                if stochastic_selection:
                                        # Stochastic policy: keep the full distribution H[leaf,:]
                                        # and a plan (incl. base_T) for every deployment with
                                        # positive probability. A request samples one per arrival.
                                        try:
                                                dist = [max(0.0, float(x)) for x in H[leaf]]
                                        except Exception:
                                                dist = [1.0] + [0.0] * (num_deployment - 1)
                                        tot_p = sum(dist)
                                        dist = [x / tot_p for x in dist] if tot_p > 0 else [1.0 / num_deployment] * num_deployment
                                        plans = {}
                                        for d in range(num_deployment):
                                                if dist[d] <= 0:
                                                        continue
                                                sp, du, rm = build_plan(d)
                                                plans[d] = {
                                                        'selected_places': sp,
                                                        'durations': du,
                                                        'rams': rm,
                                                        'base_T': plan_base_T(sp, du, leaf),
                                                }
                                        perwf[w][leaf] = {'dist': dist, 'plans': plans, 'L': L}
                                else:
                                        sel_work = 0
                                        try:
                                                sel_work = int(np.where(np.array(H[leaf]) == 1)[0][0]) if H else 0
                                        except Exception:
                                                sel_work = 0
                                        selected_places, durations, rams = build_plan(sel_work)
                                        perwf[w][leaf] = {
                                                'selected_places': selected_places,
                                                'deployment': int(sel_work),
                                                'durations': durations,
                                                'rams': rams,
                                                'L': L
                                        }

                        perwf[w]['a'] = a
                        perwf[w]['b'] = b
                        perwf[w]['num_paths'] = num_paths_w
                        perwf[w]['path_order'] = path_order

                curr_utilization=np.zeros((num_nodes_global,1))
                func_blocked=np.zeros((num_workflows,num_functions))

                node_active = [[] for _ in range(num_nodes_global)]

                ram_area = [0.0] * num_nodes_global


                waiting_acc = np.zeros((num_workflows, num_leaves))

                per_node_util = np.zeros((sim_time, num_nodes_global), dtype=float)

                request_logs = []
                req_id_counter = 0
                req_done_list=[]
                rng = np.random.default_rng(simulation_seed)
                # Separate RNG stream for deployment sampling so the Poisson
                # request stream (rng) stays identical to the deterministic run.
                rng_sel = np.random.default_rng(simulation_seed + 81)
                # sel_counts[w][leaf][d] = number of requests that sampled
                # deployment d; drives the Monte-Carlo monetary cost below.
                sel_counts = [[{} for _ in range(num_leaves)] for _ in range(num_workflows)]
                for cur_t in range(sim_time):

                        requests[cur_t]=rng.poisson(request_rates)

                        lazy_start = parameter_general_config.get('lazy_start', False)
                        if lazy_start:

                                for ni in range(num_nodes_global):
                                        node_active[ni] = [entry for entry in node_active[ni] if entry[0] > cur_t]


                                if cur_t == 0:
                                        pending_requests = []

                                for w in range(num_workflows):
                                        for leaf in range(num_leaves):
                                                n = int(requests[cur_t, w, leaf])
                                                if n <= 0:
                                                        continue
                                                info = perwf.get(w, {}).get(leaf, None)
                                                if info is None:
                                                        continue
                                                sel_places = info['selected_places']
                                                durs = info['durations']
                                                rams = info['rams']
                                                Lmat = info['L']

                                                base_T_paths = []
                                                a_mat = perwf[w].get('a', [])
                                                b_mat = perwf[w].get('b', [])
                                                num_paths_local = perwf[w].get('num_paths', 0)
                                                for s in range(num_paths_local):
                                                        func_nums = branch_functions(b_mat, s, perwf[w].get('path_order'))
                                                        tot = 0
                                                        for idx_m in func_nums:
                                                                tot += durs[idx_m]

                                                                # Positional successor (see plan_base_T).
                                                                pos = func_nums.index(idx_m)
                                                                if pos < len(func_nums) - 1:
                                                                        next_m = func_nums[pos + 1]
                                                                        next_place = sel_places[next_m]
                                                                else:
                                                                        next_place = leaf + num_other
                                                                try:
                                                                        tot += int(np.ceil(Lmat[sel_places[idx_m]][next_place]))
                                                                except Exception:
                                                                        pass
                                                        base_T_paths.append(tot)
                                                base_T = max(base_T_paths) if base_T_paths else 0

                                                deployment_id = int(info.get('deployment', w))
                                                for i in range(n):
                                                        req = {
                                                                'id': int(req_id_counter),
                                                                'workflow': int(w),
                                                                'leaf': int(leaf),
                                                                'arrival': int(cur_t),
                                                                'deployment': int(info.get('deployment', w)),
                                                                'sel_places': sel_places,
                                                                'durs': durs,
                                                                'rams': rams,
                                                                'L': Lmat,
                                                                'base_T': int(base_T),
                                                                'next_idx': 0,
                                                                'start_times': [],
                                                                'end_times': []
                                                        }
                                                        pending_requests.append(req)
                                                        req_id_counter += 1


                                still_pending = []
                                for req in pending_requests:
                                        w = req['workflow']
                                        leaf = req['leaf']
                                        sel_places = req['sel_places']
                                        durs = req['durs']
                                        rams = req['rams']
                                        Lmat = req['L']
                                        idx = req['next_idx']
                                        num_funcs = len(durs)

                                        if idx >= num_funcs:

                                                if req['end_times']:
                                                        last_node = sel_places[-1]
                                                        last_end = req['end_times'][-1]
                                                        try:
                                                                final_transfer = int(np.ceil(Lmat[last_node][leaf + num_other]))
                                                        except Exception:
                                                                final_transfer = 0
                                                        req_done = last_end + final_transfer
                                                else:
                                                        req_done = req['arrival']
                                                # Response time, not absolute completion time - see the
                                                # decomposed branch above.
                                                req_done_list.append(req_done - req['arrival'])
                                                wait = max(0, req_done - req['arrival'] - req['base_T'])
                                                waiting_acc[w, leaf] += wait

                                                request_logs.append({
                                                        'id': req['id'],
                                                        'workflow': w,
                                                        'leaf': leaf,
                                                        'arrival': req['arrival'],
                                                        'start': req['start_times'][0] if req['start_times'] else req['arrival'],
                                                        'end': req_done,
                                                        'base_T': req['base_T'],
                                                        'wait': int(wait)
                                                })
                                                continue


                                        ready = False
                                        if idx == 0:
                                                ready = True
                                        else:
                                                if len(req['end_times']) >= idx:
                                                        prev_end = req['end_times'][idx-1]
                                                        prev_node = sel_places[idx-1]
                                                        try:
                                                                next_place = sel_places[idx]
                                                        except Exception:
                                                                next_place = leaf + num_other
                                                        try:
                                                                transfer = int(np.ceil(Lmat[prev_node][next_place]))
                                                        except Exception:
                                                                transfer = 0
                                                        if prev_end + transfer <= cur_t:
                                                                ready = True

                                        if not ready:
                                                still_pending.append(req)
                                                continue


                                        node = sel_places[idx]
                                        dur = durs[idx]
                                        ram_req = rams[idx]
                                        used = sum(entry[1] for entry in node_active[node] if entry[0] > cur_t)
                                        deployment_id = req.get('deployment', w)
                                        func_busy = any(len(entry) >= 4 and entry[0] > cur_t and entry[2] == deployment_id and entry[3] == idx for entry in node_active[node])
                                        if used + ram_req <= node_ram_max[node] and (not func_busy):

                                                start = cur_t
                                                endt = start + dur
                                                node_active[node].append((endt, ram_req, deployment_id, idx, req['id'],w))
                                                req['start_times'].append(start)
                                                req['end_times'].append(endt)
                                                req['next_idx'] += 1

                                                if req['next_idx'] < num_funcs:
                                                        still_pending.append(req)
                                                else:

                                                        still_pending.append(req)
                                        else:
                                                still_pending.append(req)

                                pending_requests = still_pending


                                for ni in range(num_nodes_global):
                                        used_now = sum(entry[1] for entry in node_active[ni] if entry[0] > cur_t)
                                        ram_area[ni] += used_now
                                        per_node_util[cur_t, ni] = used_now
                        else:

                                for ni in range(num_nodes_global):
                                        node_active[ni] = [entry for entry in node_active[ni] if entry[0] > cur_t]

                                for w in range(num_workflows):
                                        for leaf in range(num_leaves):
                                                n = int(requests[cur_t, w, leaf])
                                                if n <= 0:
                                                        continue
                                                info = perwf.get(w, {}).get(leaf, None)
                                                if info is None:
                                                        continue
                                                Lmat = info['L']
                                                a_mat = perwf[w].get('a', [])
                                                num_paths_local = perwf[w].get('num_paths', 0)
                                                if not stochastic_selection:
                                                        # Deterministic: one fixed plan and base_T per (w, leaf).
                                                        sel_places = info['selected_places']
                                                        durs = info['durations']
                                                        rams = info['rams']
                                                        base_T_paths = []
                                                        for s in range(num_paths_local):
                                                                b_mat = perwf[w].get('b', [])
                                                                func_nums = branch_functions(b_mat, s, perwf[w].get('path_order'))
                                                                tot = 0
                                                                for idx_m in func_nums:
                                                                        tot += durs[idx_m]

                                                                        # Positional successor (see plan_base_T).
                                                                        pos = func_nums.index(idx_m)
                                                                        if pos < len(func_nums) - 1:
                                                                                next_m = func_nums[pos + 1]
                                                                                next_place = sel_places[next_m]
                                                                        else:
                                                                                next_place = leaf + num_other
                                                                        try:
                                                                                tot += int(np.ceil(Lmat[sel_places[idx_m]][next_place]))
                                                                        except Exception:
                                                                                pass
                                                                base_T_paths.append(tot)
                                                        base_T = max(base_T_paths) if base_T_paths else 0
                                                        deployment_id = int(info.get('deployment', w))
                                                for i in range(n):
                                                        if stochastic_selection:
                                                                # Sample this request's deployment from the policy
                                                                # H[leaf,:] and use that deployment's plan/base_T.
                                                                dist = info['dist']
                                                                d_sel = int(rng_sel.choice(len(dist), p=dist))
                                                                plan = info['plans'].get(d_sel)
                                                                if plan is None:
                                                                        continue
                                                                sel_places = plan['selected_places']
                                                                durs = plan['durations']
                                                                rams = plan['rams']
                                                                base_T = plan['base_T']
                                                                deployment_id = d_sel
                                                                sel_counts[w][leaf][d_sel] = sel_counts[w][leaf].get(d_sel, 0) + 1
                                                        arrival = cur_t
                                                        path_finish_times = []
                                                        b_mat = perwf[w].get('b', [])
                                                        num_paths_local = perwf[w].get('num_paths', 0)
                                                        for s in range(num_paths_local):
                                                                func_nums = branch_functions(b_mat, s, perwf[w].get('path_order'))
                                                                curr_time = arrival
                                                                for idx_m in func_nums:
                                                                        node = sel_places[idx_m]
                                                                        dur = durs[idx_m]
                                                                        ram_req = rams[idx_m]

                                                                        while True:

                                                                                # Only executions whose [start, end) interval
                                                                                # overlaps the candidate slot occupy the node -
                                                                                # see the decomposed loop above.
                                                                                slot_end = curr_time + dur
                                                                                overlapping = [entry for entry in node_active[node]
                                                                                               if entry[5] < slot_end and curr_time < entry[0]]
                                                                                used = sum(entry[1] for entry in overlapping)

                                                                                func_busy_ends = [entry[0] for entry in overlapping
                                                                                                  if entry[2] == deployment_id
                                                                                                  and entry[3] == idx_m
                                                                                                  and entry[4] == w]

                                                                                if used + ram_req <= node_ram_max[node] and (not func_busy_ends):
                                                                                        start = curr_time
                                                                                        break


                                                                                future_ends = [entry[0] for entry in node_active[node] if entry[0] > curr_time]
                                                                                if future_ends:
                                                                                        curr_time = int(min(future_ends))
                                                                                else:
                                                                                        start = curr_time
                                                                                        break
                                                                        endt = start + dur

                                                                        node_active[node].append((endt, ram_req, deployment_id, idx_m, w, start))

                                                                        idx_pos = func_nums.index(idx_m)
                                                                        if idx_pos < len(func_nums) - 1:
                                                                                next_m = func_nums[idx_pos + 1]
                                                                                next_place = sel_places[next_m]
                                                                        else:
                                                                                next_place = leaf + num_other
                                                                        try:
                                                                                transfer = int(np.ceil(Lmat[node][next_place]))
                                                                        except Exception:
                                                                                transfer = 0
                                                                        curr_time = endt + transfer
                                                                path_finish_times.append(curr_time)
                                                        req_done = max(path_finish_times) if path_finish_times else arrival
                                                        req_done_list.append(req_done-arrival)
                                                        wait = (req_done - arrival) - base_T
                                                        if wait < 0:
                                                                wait = 0

                                                        waiting_acc[w, leaf] += wait

                                                        req_log = {
                                                                'id': int(req_id_counter),
                                                                'workflow': int(w),
                                                                'leaf': int(leaf),
                                                                'arrival': int(arrival),
                                                                'start': int(arrival + wait),
                                                                'end': int(req_done),
                                                                'base_T': int(base_T),
                                                                'wait': int(wait)
                                                        }
                                                        request_logs.append(req_log)
                                                        req_id_counter += 1

                                for ni in range(num_nodes_global):
                                        # Started and not yet finished - see above.
                                        used_now = sum(entry[1] for entry in node_active[ni]
                                                       if entry[5] <= cur_t < entry[0])
                                        ram_area[ni] += used_now
                                        per_node_util[cur_t, ni] = used_now


                overall_requests=np.zeros((num_workflows,num_leaves))
                for t in range(num_workflows):
                        for k in range(num_leaves):
                                overall_requests[t,k]=sum(requests[:,t,k])

                # Monetary/time cost evaluation of the comparison solution -
                # same cost terms as for the decomposed case, but directly on
                # the flat P/H/L matrices of the centralized placement.
                for t in range(num_workflows):

                        cloud_name = f"requirements_cloud_full.json"
                        sol_name = f"opt_{t}"
                        path_cloud = in_path("cloud", cloud_name)
                        path_workflow = in_path("workflows", f"requirements_workflow_{t}" + "_0.json")
                        path_solution = in_path("placement", sol_name + "_full.json")

                        cloud_config = load_json_cached(path_cloud)
                        workflow_config = load_json_cached(path_workflow)
                        placement_config = load_json_cached(path_solution)
                        num_functions = workflow_config["num_functions"]
                        num_deployment = workflow_config["deployment_number"]
                        num_nodes=cloud_config["num_nodes"]
                        H = placement_config['H']
                        T = placement_config['T']
                        P = placement_config['P']
                        C = placement_config['C']
                        D = placement_config['D']
                        D_in = placement_config['D_in']
                        # Latency from the (perturbable) full cloud config; fall
                        # back to the placement solution L for legacy runs.
                        L = cloud_config.get('L', placement_config['L'])
                        S = placement_config['S']
                        num_paths = workflow_config['paths']['num_paths']
                        # Execution order of each branch (see branch_functions).
                        path_order = path_order_of(workflow_config)
                        b = placement_config['b']
                        a = placement_config['a']
                        select_cost=0
                        invocations=np.zeros((num_nodes))

                        # Monetary cost per (leaf, deployment): deterministic weights
                        # the one-hot deployment by all requests at the leaf; stochastic
                        # weights each deployment by how many requests sampled it
                        # (Monte-Carlo expectation over the routing policy H).
                        if stochastic_selection:
                                work_items = [(k, d, c) for k in range(num_leaves) for d, c in sel_counts[t][k].items()]
                        else:
                                work_items = [(k, H[k].index(1), overall_requests[t, k]) for k in range(num_leaves)]
                        for k, selected_workflow, weight in work_items:

                                pred_k=k+num_other
                                time_cost = 0
                                money_cost = 0

                                T_paths = np.zeros((num_paths, 1))
                                D_func = np.zeros((num_functions, 1))
                                C_func = np.zeros((num_functions, 1))
                                L_func = np.zeros((num_functions, 1))
                                for s in range(num_paths):
                                        pred_k = k + num_other

                                        func_nums = branch_functions(b, s, path_order)

                                        for r,m in enumerate(func_nums):

                                                selected_place = P[selected_workflow * num_functions + m].index(1)
                                                # r is the POSITION of m in func_nums (enumerate), so the
                                                # end-of-branch test must use r, not the function id m.
                                                if r < len(func_nums)-1:

                                                        next_selected_place=P[selected_workflow * num_functions + func_nums[r+1]].index(1)
                                                else:
                                                        next_selected_place=pred_k
                                                curr_selected_workflow = selected_workflow

                                                L_func[m] = L[pred_k][selected_place]

                                                if m == 0:

                                                        pricing_data_sent = cloud_config['providers'][
                                                                f"provider_{selected_place}"][
                                                                'pricing_data_sent']
                                                        D_in_0 = D_in[selected_place]


                                                pricing_data_sent = \
                                                        cloud_config['providers'][
                                                                f"provider_{selected_place}"][
                                                                'pricing_data_sent']
                                                D_func[m] =D[m][selected_place][next_selected_place]


                                                if cloud_config['providers'][f"provider_{selected_place}"]["name"]=="tinyfaas":
                                                        speedup=1
                                                else:
                                                        speedup = workflow_config["functions"][f"function_{m}"][
                                                                'speedup']
                                                ram = workflow_config['functions'][f"function_{m}"]['ram']
                                                time = workflow_config['functions'][f"function_{m}"][
                                                        'time']
                                                T_func = speedup * time
                                                data_dependencies = workflow_config['functions'][f"function_{m}"][
                                                        'data_dependencies']
                                                for key in data_dependencies.keys():
                                                        value = data_dependencies[key]
                                                        pricing_Storage_Transfer = provider_cost_list[key][
                                                                "pricing_Storage_Transfer"]
                                                        if key != cloud_config["providers"][f"provider_{selected_place}"][
                                                                "name"]:

                                                                C_func[m] = C_func[m] + pricing_Storage_Transfer * value
                                                pricing_RAM = \
                                                cloud_config['providers'][f"provider_{selected_place}"][
                                                        'pricing_RAM']
                                                C_func[m] += pricing_RAM * ram  * time

                                                T_paths[s] += (T_func+L_func[m])
                                                pred_k=selected_place
                                                invocations[selected_place]+=1

                                        T_paths[s]+=L[selected_place][k+num_other]


                                time_cost += np.max(T_paths)


                                money_cost += np.sum(C_func) + np.sum(D_func) + D_in_0


                                time_cost_all += weight*time_cost
                                money_cost_all += weight*money_cost

        overall_cost = w_1 * money_cost_all+w_2 * sum(req_done_list)
        print(overall_cost)
        results['overall_cost_comparison'] = overall_cost[0]

        # Persist the comparison run's cost split in a sibling file (results.json
        # itself only carries overall_cost_comparison and is consumed by the
        # visualization scripts, so its content stays unchanged). The comparison
        # is the centralized solve or the all-on-cloud heuristic depending on
        # the run's settings; the flag records which one produced these costs.
        comparison_results = {
                'heuristic': bool(heuristic),
                'overall_cost': float(np.ravel(overall_cost)[0]),
                'money_cost': float(np.ravel(money_cost_all)[0]),
                'time_cost': float(np.ravel(time_cost_all)[0]),
                'complete_time_cost': float(sum(sum(waiting_acc))),
        }
        with open(os.path.join(results_dir, "results_comparison.json"), "w") as f:
                json.dump(comparison_results, f, indent=4)


        # Store the aggregated costs (results.json: used by the visualization
        # scripts) and a detailed per-request / per-node trace of the run.
        with open(results_path, "w") as f:
                json.dump(results, f, indent=4)


        detailed = {
                'per_node_utilization': per_node_util.tolist(),
                'request_logs': request_logs,
                'waiting_acc': waiting_acc.tolist()
        }
        detailed_path = os.path.join(results_dir, f"detailed_simulation_run_{num_run}.json")
        with open(detailed_path, 'w') as f:
                json.dump(detailed, f, indent=2)