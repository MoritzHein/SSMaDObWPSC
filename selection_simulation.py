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
if __name__ == "__main__":


        parser = argparse.ArgumentParser()
        parser.add_argument("--num_run", required=True, type=int)
        args = parser.parse_args()
        num_run = args.num_run

        placement_path=f"placement/run_{num_run}"
        parameter_config_path=f"settings/parameters_{num_run}.json"
        with open(parameter_config_path, "r") as f:
                parameter_general_config = json.load(f)
        provider_cost_list=parameter_general_config["provider_cost_list"]
        general_config_path=f"cloud/run_{num_run}/general_config.json"
        with open(general_config_path, "r") as f:
                cloud_general_config = json.load(f)
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

                        placement_path = f"placement/run_{num_run}/opt_{t}_{pred_k+ind_start_list[r]}.json"
                        with open(placement_path, "r") as f:
                                curr_placement = json.load(f)
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
        np.random.seed(42)
        sim_time = parameter_general_config["simulation_time"]
        time_cost_all = 0
        money_cost_all = 0
        select_cost_all = 0
        time_cost = 0
        money_cost = 0
        request_rates = np.zeros((num_workflows, num_leaves))
        for t in range(num_workflows):
                cloud_name = f"requirements_cloud_full.json"
                path_cloud = os.path.join(f"cloud/run_{num_run}", cloud_name)
                with open(path_cloud, "r") as f:
                        cloud_config = json.load(f)
                for k in range(num_leaves):
                        request_rates[t, k] = cloud_config["providers"][f"provider_{k + num_other}"]["request_rate"]
        requests = np.zeros((sim_time, num_workflows, num_leaves))

        # RAM capacity per physical node: only edge (tinyfaas) nodes are
        # limited; cloud nodes never block due to RAM.
        cloud_full_path = os.path.join(f"cloud/run_{num_run}", "requirements_cloud_full.json")
        with open(cloud_full_path, 'r') as f:
                cloud_full = json.load(f)
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

                wf_path = os.path.join(f"workflows/run_{num_run}", f"requirements_workflow_{w}_0.json")
                try:
                        with open(wf_path, 'r') as f:
                                wf_conf = json.load(f)
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
                full_path = os.path.join(f"placement/run_{num_run}", f"opt_{w}_full.json")
                try:
                        with open(full_path, 'r') as f:
                                full_place_tmp = json.load(f)
                                if 'L' in full_place_tmp:
                                        L_full = full_place_tmp.get('L', [])
                                a_full = full_place_tmp.get('a', [])
                                b_full = full_place_tmp.get('b', [])
                                if len(b_full)>1:
                                        for r in range(1,len(b_full)):
                                                b_full[r][0]=0
                                                b_full[r][-1]=0
                except Exception:
                        pass


                top_path = os.path.join(f"placement/run_{num_run}", f"opt_{w}_0.json")
                try:
                        with open(top_path, 'r') as f:
                                top_place = json.load(f)
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
                                        placement_path = f"placement/run_{num_run}/opt_{w}_{opt_num}.json"
                                        try:
                                                with open(placement_path, 'r') as f:
                                                        curr_placement = json.load(f)
                                        except Exception:
                                                curr_placement = {}

                                        curr_P = curr_placement.get('P', [])
                                        curr_workflow = {}
                                        try:
                                                workflow_path = f"workflows/run_{num_run}/requirements_workflow_{w}_{opt_num}.json"
                                                with open(workflow_path, 'r') as f:
                                                        curr_workflow = json.load(f)
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
                wf_path = os.path.join(f"workflows/run_{num_run}", f"requirements_workflow_{w}_0.json")
                sol_path = os.path.join(f"placement/run_{num_run}", f"opt_{w}_full.json")
                node_path = os.path.join(f"cloud/run_{num_run}", f"requirements_cloud_full.json")
                try:
                        with open(wf_path, 'r') as f:
                                wf_conf = json.load(f)
                        with open(node_path, 'r') as f:
                                node_conf = json.load(f)
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
                for leaf in range(num_leaves):
                        sel_work = 0
                        try:
                                sel_work = int(np.where(np.array(H[leaf]) == 1)[0][0]) if H else 0
                        except Exception:
                                sel_work = 0

                        selected_places = []
                        for m in range(num_funcs):
                                try:
                                        row = P[sel_work * num_funcs + m]
                                        selected_places.append(int(np.where(np.array(row) == 1)[0][0]))
                                except Exception:
                                        selected_places.append(0)


                        durations = [0] * num_funcs
                        rams = [0] * num_funcs
                        for m in range(num_funcs):
                                node = selected_places[m]
                                node_name=node_conf["providers"][f"provider_{node}"]["name"]
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
                                                        speedup=1
                                        dur = int(np.ceil(speedup * base_time))
                                        durations[m] = dur
                                        rams[m] = func_conf.get('ram', 0)
                                except Exception:
                                        durations[m] = 0
                                        rams[m] = 0

                        perwf[w][leaf] = {
                                'selected_places': selected_places,
                                'deployment': int(sel_work),
                                'durations': durations,
                                'rams': rams,
                                'L': L
                        }

                perwf[w]['a'] = a
                perwf[w]['b'] = b
                try:
                        perwf[w]['num_paths'] = wf_conf.get('paths', {}).get('num_paths', 0)
                except Exception:
                        perwf[w]['num_paths'] = 0

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

        rng = np.random.default_rng(42)

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
                                                func_nums = np.where(np.array(b_mat[s])==1)[0].tolist() if isinstance(b_mat, (list, np.ndarray)) and len(b_mat) > s else []

                                                tot = 0
                                                for idx_m in func_nums:
                                                        tot += durs[idx_m]

                                                        if idx_m < len(func_nums) - 1:
                                                                next_m = func_nums[func_nums.index(idx_m) + 1]
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
                                        req_done_list.append(req_done)
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
                                        sel_places = info['selected_places']
                                        durs = info['durations']
                                        rams = info['rams']
                                        Lmat = info['L']
                                        deployment_id = int(info.get('deployment', w))

                                        base_T_paths = []
                                        a_mat = perwf[w].get('a', [])
                                        b_mat = perwf[w].get('b', [])
                                        num_paths_local = perwf[w].get('num_paths', 0)
                                        for s in range(num_paths_local):
                                                func_nums = np.where(np.array(b_mat[s])==1)[0].tolist() if isinstance(b_mat, (list, np.ndarray)) and len(b_mat) > s else []

                                                tot = 0
                                                for idx_m in func_nums:
                                                        tot += durs[idx_m]

                                                        if idx_m < len(func_nums) - 1:
                                                                next_m = func_nums[func_nums.index(idx_m) + 1]
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
                                                arrival = cur_t
                                                path_finish_times = []

                                                a_mat = perwf[w].get('a', [])
                                                b_mat = perwf[w].get('b', [])
                                                num_paths_local = perwf[w].get('num_paths', 0)
                                                for s in range(num_paths_local):
                                                        func_nums = np.where(np.array(b_mat[s])==1)[0].tolist() if isinstance(b_mat, (list, np.ndarray)) and len(b_mat) > s else []

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

                                                                        used = sum(
                                                                                entry[1] for entry in node_active[node]
                                                                                if entry[0] > curr_time)

                                                                        func_busy_ends = [entry[0] for entry in
                                                                                          node_active[node] if
                                                                                          len(entry) >= 4 and entry[
                                                                                                  0] > curr_time and
                                                                                          entry[2] == deployment_id and entry[
                                                                                                  3] == idx_m and entry[4] == w]

                                                                        if used + ram_req <= node_ram_max[node] and (
                                                                        not func_busy_ends):

                                                                                start = curr_time
                                                                                break

                                                                        future_ends_node = [entry[0] for entry in
                                                                                            node_active[node] if
                                                                                            entry[0] > curr_time]
                                                                        future_ends = future_ends_node + func_busy_ends
                                                                        if future_ends:
                                                                                curr_time = int(min(future_ends))
                                                                        else:
                                                                                start = curr_time
                                                                                break
                                                                endt = start + dur

                                                                node_active[node].append((endt, ram_req, deployment_id, idx_m,w))

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

                                used_now = sum(entry[1] for entry in node_active[ni] if entry[0] > cur_t)
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
                workflow_path=f"workflows/run_{num_run}/requirements_workflow_{t}_0.json"
                placement_path=f"placement/run_{num_run}/opt_{t}_0.json"
                with open(workflow_path, "r") as f:
                        workflow_config = json.load(f)
                with open(placement_path, "r") as f:
                        placement_config = json.load(f)
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
                b = placement_config['b']
                a = placement_config['a']
                num_deployment=workflow_config["deployment_number"]
                workflow_overall=np.zeros((num_deployment))
                num_nodes_all=sum(num_nodes_in_level[-1])
                invocations=np.zeros(num_nodes_all)
                placement_overall=0
                for k in range(num_leaves):

                        # Which deployment serves this leaf: read the selection H of
                        # the top-level problem for the leaf's level-1 region.
                        pred_k_list, lat_upward_list,pos_list = upward_trajectory(k+num_other)

                        selected_workflow = H[
                                pred_k_list[1] - num_other_first_level].index(1)

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

                                func_nums = np.where(np.array(b[s])==1)[0].tolist()

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
                                                placement_path=f"placement/run_{num_run}/opt_{t}_{opt_num}.json"
                                                workflow_path = f"workflows/run_{num_run}/requirements_workflow_{t}_{opt_num}.json"
                                                cloud_path = f"cloud/run_{num_run}/requirements_cloud_{r}_{opt_num-ind_start_list[r]}.json"
                                                with open(placement_path, "r") as f:
                                                        curr_placement = json.load(f)
                                                with open(cloud_path, "r") as f:
                                                        curr_cloud = json.load(f)
                                                with open(workflow_path, "r") as f:
                                                        curr_workflow = json.load(f)

                                                curr_P=curr_placement['P']
                                                curr_L = curr_placement['L']
                                                if r==0:

                                                        curr_selected_workflow = selected_workflow
                                                        selected_place = P[selected_workflow * num_functions + m].index(
                                                                1)
                                                else:

                                                        curr_selected_workflow=find_correct_workflow(curr_selected_workflow,curr_workflow["active_workflow_list"])
                                                if curr_P==[]:
                                                        selected_place=0
                                                else:
                                                        selected_place=curr_P[curr_selected_workflow*num_functions+m].index(1)

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

                                                        if m < len(func_nums) - 1:

                                                                next_selected_place = curr_P[
                                                                        selected_workflow * num_functions + func_nums[
                                                                                m + 1]].index(1)
                                                        else:
                                                                next_selected_place = pred_k_list[1]
                                                        D_func[m]=D[m][selected_place][next_selected_place]

                                                else:
                                                        # Lower levels: if the user's trajectory passes through
                                                        # the same region, add the intra-region latency;
                                                        # otherwise route via the head nodes (latency to the
                                                        # own head node plus the user's upward latency).
                                                        if pred_k_list[r]==opt_num-ind_start_list[r]:

                                                                with open(placement_path, "r") as f:
                                                                        curr_placement = json.load(f)


                                                                curr_L = curr_placement['L']
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
                                                placement_path = f"placement/run_{num_run}/opt_{t}_{opt_num}.json"
                                                with open(placement_path, "r") as f:
                                                        curr_placement = json.load(f)
                                                curr_L = curr_placement['L']
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




                        time_cost_all+=overall_requests[t,k]*time_cost

                        money_cost_all+=overall_requests[t,k]*money_cost
                select_cost=np.sum(workflow_overall**2)+np.sum(invocations**2)
                select_cost_all+=select_cost

        # Overall simulated cost C = w_1*M + w_2*(B+W): req_done_list holds the
        # complete per-request response times (basic time + waiting).
        overall_cost = w_1 * money_cost_all+ w_2 * sum(req_done_list)
        print(sum(sum(waiting_acc)))
        print([overall_cost])
        results_path = f"results/run_{num_run}/results.json"
        if not os.path.exists(os.path.join("results",f"run_{num_run}")):
                os.mkdir(os.path.join("results",f"run_{num_run}"))
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
                np.random.seed(42)
                sim_time=parameter_general_config["simulation_time"]
                time_cost_all=0
                money_cost_all=0
                select_cost_all=0
                time_cost=0
                money_cost=0
                request_rates=np.zeros((num_workflows,num_leaves))
                for t in range(num_workflows):
                        cloud_name = f"requirements_cloud_full.json"
                        path_cloud = os.path.join(f"cloud/run_{num_run}", cloud_name)
                        with open(path_cloud, "r") as f:
                                cloud_config = json.load(f)
                        for k in range(num_leaves):
                                request_rates[t,k]=cloud_config["providers"][f"provider_{k+num_other}"]["request_rate"]
                requests=np.zeros((sim_time,num_workflows,num_leaves))
                cloud_full_path = os.path.join(f"cloud/run_{num_run}", "requirements_cloud_full.json")
                with open(cloud_full_path, 'r') as f:
                        cloud_full = json.load(f)
                num_nodes_global = cloud_full.get('num_nodes', 0)
                node_ram_max = [cloud_full['providers'][f'provider_{i}'].get('max_RAM_curr', 1e9) if cloud_full['providers'][f'provider_{i}'].get('name')=="tinyfaas" else np.inf for i in
                        range(num_nodes_global)]

                perwf = {}

                def reconstruct_full_solution(w):

                        wf_path = os.path.join(f"workflows/run_{num_run}", f"requirements_workflow_{w}_0.json")
                        try:
                                with open(wf_path, 'r') as f:
                                        wf_conf = json.load(f)
                        except Exception:
                                wf_conf = {}
                        num_funcs = wf_conf.get('num_functions', 0)

                        num_nodes_all = cloud_full.get('num_nodes', 0)

                        P_full = [[0 for _ in range(num_nodes_all)] for _ in range(num_funcs)]

                        H_full = [[0] for _ in range(num_leaves)]


                        L_full = []
                        full_path = os.path.join(f"placement/run_{num_run}", f"opt_{w}_full.json")
                        try:
                                with open(full_path, 'r') as f:
                                        full_place_tmp = json.load(f)
                                        if 'L' in full_place_tmp:
                                                L_full = full_place_tmp.get('L', [])
                        except Exception:
                                pass


                        top_path = os.path.join(f"placement/run_{num_run}", f"opt_{w}_0.json")
                        try:
                                with open(top_path, 'r') as f:
                                        top_place = json.load(f)
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
                                                placement_path = f"placement/run_{num_run}/opt_{w}_{opt_num}.json"
                                                try:
                                                        with open(placement_path, 'r') as f:
                                                                curr_placement = json.load(f)
                                                except Exception:
                                                        curr_placement = {}

                                                curr_P = curr_placement.get('P', [])
                                                curr_workflow = {}
                                                try:
                                                        workflow_path = f"workflows/run_{num_run}/requirements_workflow_{w}_{opt_num}.json"
                                                        with open(workflow_path, 'r') as f:
                                                                curr_workflow = json.load(f)
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
                        wf_path = os.path.join(f"workflows/run_{num_run}", f"requirements_workflow_{w}_0.json")
                        sol_path = os.path.join(f"placement/run_{num_run}", f"opt_{w}_full.json")
                        node_path = os.path.join(f"cloud/run_{num_run}", f"requirements_cloud_full.json")
                        try:
                                with open(wf_path, 'r') as f:
                                        wf_conf = json.load(f)
                                with open(node_path, 'r') as f:
                                        node_conf = json.load(f)
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
                                        with open(sol_path, 'r') as f:
                                                sol_conf = json.load(f)
                                except Exception:
                                        sol_conf = {}

                                if not sol_conf.get('P'):
                                        try:
                                                sol_conf = reconstruct_full_solution(w)
                                        except Exception:
                                                sol_conf = sol_conf or {}
                        P = sol_conf.get('P', [])
                        H = sol_conf.get('H', [])
                        L = sol_conf.get('L', [])
                        Speed = sol_conf.get('Speed', None)
                        a = sol_conf.get('a', [])
                        b = sol_conf.get('b', [])
                        # Count the shared start/finish functions in only one branch so
                        # they are not executed once per branch in the simulation.
                        if len(b) > 1:
                                for r in range(1, len(b)):
                                        b[r][0] = 0
                                        b[r][-1] = 0
                        num_funcs = wf_conf.get('num_functions', 0)
                        perwf[w] = {}
                        for leaf in range(num_leaves):
                                sel_work = 0
                                try:
                                        sel_work = int(np.where(np.array(H[leaf]) == 1)[0][0]) if H else 0
                                except Exception:
                                        sel_work = 0

                                selected_places = []
                                for m in range(num_funcs):
                                        try:
                                                row = P[sel_work * num_funcs + m]
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

                                perwf[w][leaf] = {
                                        'selected_places': selected_places,
                                        'deployment': int(sel_work),
                                        'durations': durations,
                                        'rams': rams,
                                        'L': L
                                }

                        perwf[w]['a'] = a
                        perwf[w]['b'] = b
                        try:
                                perwf[w]['num_paths'] = wf_conf.get('paths', {}).get('num_paths', 0)
                        except Exception:
                                perwf[w]['num_paths'] = 0

                curr_utilization=np.zeros((num_nodes_global,1))
                func_blocked=np.zeros((num_workflows,num_functions))

                node_active = [[] for _ in range(num_nodes_global)]

                ram_area = [0.0] * num_nodes_global


                waiting_acc = np.zeros((num_workflows, num_leaves))

                per_node_util = np.zeros((sim_time, num_nodes_global), dtype=float)

                request_logs = []
                req_id_counter = 0
                req_done_list=[]
                rng = np.random.default_rng(42)
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
                                                        func_nums = np.where(np.array(b_mat[s])==1)[0].tolist() if isinstance(b_mat, (list, np.ndarray)) and len(b_mat) > s else []
                                                        tot = 0
                                                        for idx_m in func_nums:
                                                                tot += durs[idx_m]

                                                                if idx_m < len(func_nums) - 1:
                                                                        next_m = func_nums[func_nums.index(idx_m) + 1]
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
                                                req_done_list.append(req_done)
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
                                                sel_places = info['selected_places']
                                                durs = info['durations']
                                                rams = info['rams']
                                                Lmat = info['L']

                                                base_T_paths = []
                                                a_mat = perwf[w].get('a', [])
                                                num_paths_local = perwf[w].get('num_paths', 0)
                                                for s in range(num_paths_local):
                                                        b_mat = perwf[w].get('b', [])
                                                        func_nums = np.where(np.array(b_mat[s])==1)[0].tolist() if isinstance(b_mat, (list, np.ndarray)) and len(b_mat) > s else []
                                                        tot = 0
                                                        for idx_m in func_nums:
                                                                tot += durs[idx_m]

                                                                if idx_m < len(func_nums) - 1:
                                                                        next_m = func_nums[func_nums.index(idx_m)+1]
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
                                                        arrival = cur_t
                                                        path_finish_times = []
                                                        b_mat = perwf[w].get('b', [])
                                                        num_paths_local = perwf[w].get('num_paths', 0)
                                                        for s in range(num_paths_local):
                                                                func_nums = np.where(np.array(b_mat[s]) == 1)[0].tolist() if isinstance(b_mat, list) and len(b_mat) > s else []
                                                                curr_time = arrival
                                                                for idx_m in func_nums:
                                                                        node = sel_places[idx_m]
                                                                        dur = durs[idx_m]
                                                                        ram_req = rams[idx_m]

                                                                        while True:

                                                                                used = sum(entry[1] for entry in node_active[node] if entry[0] > curr_time)

                                                                                func_busy_ends = [entry[0] for entry in
                                                                                          node_active[node] if
                                                                                          len(entry) >= 4 and entry[
                                                                                                  0] > curr_time and
                                                                                          entry[2] == deployment_id and entry[
                                                                                                  3] == idx_m and entry[4] == w]

                                                                                if used + ram_req <= node_ram_max[node] and (not func_busy_ends):
                                                                                        start = curr_time
                                                                                        break


                                                                                future_ends_node = [entry[0] for entry in node_active[node] if entry[0] > curr_time]
                                                                                future_ends = future_ends_node + func_busy_ends
                                                                                if future_ends:
                                                                                        curr_time = int(min(future_ends))
                                                                                else:
                                                                                        start = curr_time
                                                                                        break
                                                                        endt = start + dur

                                                                        node_active[node].append((endt, ram_req, deployment_id, idx_m,w))

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
                                        used_now = sum(entry[1] for entry in node_active[ni] if entry[0] > cur_t)
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
                        path_cloud = os.path.join(f"cloud/run_{num_run}", cloud_name)
                        path_workflow = os.path.join(f"workflows/run_{num_run}", f"requirements_workflow_{t}" + "_0.json")
                        path_solution = os.path.join(f"placement/run_{num_run}", sol_name + "_full.json")

                        with open(path_cloud, "r") as f:
                                cloud_config = json.load(f)
                        with open(path_workflow, "r") as f:
                                workflow_config = json.load(f)
                        with open(path_solution, "r") as f:
                                placement_config = json.load(f)
                        num_functions = workflow_config["num_functions"]
                        num_deployment = workflow_config["deployment_number"]
                        num_nodes=cloud_config["num_nodes"]
                        H = placement_config['H']
                        T = placement_config['T']
                        P = placement_config['P']
                        C = placement_config['C']
                        D = placement_config['D']
                        D_in = placement_config['D_in']
                        L = placement_config['L']
                        S = placement_config['S']
                        num_paths = workflow_config['paths']['num_paths']
                        b = placement_config['b']
                        a = placement_config['a']
                        select_cost=0
                        invocations=np.zeros((num_nodes))

                        for k in range(num_leaves):

                                pred_k=k+num_other
                                selected_workflow = H[k].index(1)
                                time_cost = 0
                                money_cost = 0

                                T_paths = np.zeros((num_paths, 1))
                                D_func = np.zeros((num_functions, 1))
                                C_func = np.zeros((num_functions, 1))
                                L_func = np.zeros((num_functions, 1))
                                for s in range(num_paths):
                                        pred_k = k + num_other

                                        func_nums = np.where(np.array(b[s]) == 1)[0].tolist()

                                        for r,m in enumerate(func_nums):

                                                selected_place = P[selected_workflow * num_functions + m].index(1)
                                                if m<len(func_nums)-1:

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


                                time_cost_all += overall_requests[t,k]*time_cost
                                money_cost_all += overall_requests[t,k]*money_cost

        overall_cost = w_1 * money_cost_all+w_2 * sum(req_done_list)
        print(overall_cost)
        results['overall_cost_comparison'] = overall_cost[0]


        # Store the aggregated costs (results.json: used by the visualization
        # scripts) and a detailed per-request / per-node trace of the run.
        with open(results_path, "w") as f:
                json.dump(results, f, indent=4)


        detailed = {
                'per_node_utilization': per_node_util.tolist(),
                'request_logs': request_logs,
                'waiting_acc': waiting_acc.tolist()
        }
        detailed_path = os.path.join(f"results/run_{num_run}", f"detailed_simulation_run_{num_run}.json")
        with open(detailed_path, 'w') as f:
                json.dump(detailed, f, indent=2)