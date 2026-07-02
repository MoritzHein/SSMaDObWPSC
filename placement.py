# Main driver of the decomposed placement approach (Sec. III-C of the paper).
#
# Usage: placement.py --num_run=<N>
#
# Reads the experiment description settings/parameters_<N>.json, then
#   1. generates a random hierarchical serverless architecture (cloud/run_<N>/)
#      and a set of random workflows (workflows/run_<N>/),
#   2. solves the placement problem level by level: the level-0 problem places
#      functions on region/provider head nodes, and every lower-level
#      subproblem refines the placement of its predecessor's head node within
#      one region (placement/run_<N>/opt_<workflow>_<subproblem>.json),
#   3. optionally solves the full centralized problem (or the all-on-cloud
#      heuristic) on the flattened architecture for comparison,
#   4. stores all wall-clock solve times in results/run_<N>/time.json.
import argparse
import os.path
from cloud_workflow_setup import *
from optimization import *
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script that adds 3 numbers from CMD"
    )
    parser.add_argument("--num_run", required=False, type=int, default=None)
    args = parser.parse_args()
    num_run=args.num_run
    if num_run is None:
        while True:
            try:
                num_run = int(input("Please enter num_run: "))
                break
            except ValueError:
                print("num_run must be an integer. Please try again.")

    # Load the experiment parameters: objective weights w_1..w_3 (Eq. (1)),
    # architecture size (num_levels), number of workflows, provider pricing
    # and the sampling bounds for random workflows/clouds (Table 2).
    parameter_config_path=f"settings/parameters_{num_run}.json"
    with open(parameter_config_path, "r") as f:
        parameter_general_config = json.load(f)
    seed= parameter_general_config["seed"]
    w_1=parameter_general_config["w_1"]
    w_2=parameter_general_config["w_2"]
    w_3=parameter_general_config["w_3"]
    num_levels=parameter_general_config["num_levels"]
    num_workflows=parameter_general_config["num_workflows"]
    provider_list=parameter_general_config["provider_list"]#["aws","tinyfaas"]
    num_tiny_first=parameter_general_config["num_tiny_first"]
    comparison=parameter_general_config["comparison"]
    provider_cost_list=parameter_general_config["provider_cost_list"]
    provider_dev=parameter_general_config["provider_cost_deviation"]
    comparison_gap=parameter_general_config["comparison_gap"]
    own_gap=parameter_general_config["own_gap"]
    config_random_workflows=parameter_general_config["config_random_workflows"]
    config_random_cloud=parameter_general_config["config_random_cloud"]
    # Stage 1 of the stochastic-selection relaxation: when enabled, the
    # centralized (comparison) solve treats H as a continuous routing policy.
    # Absent => False => unchanged deterministic behavior.
    stochastic_selection=parameter_general_config.get("stochastic_selection", False)
    # Sample the hierarchical serverless architecture: one cloud config file per
    # (level, region) pair plus a general_config.json describing the hierarchy.
    generate_cloud(num_levels,num_run=num_run,provider_name=provider_list,provider_dev=provider_dev,num_tiny_first=num_tiny_first,provider_cost_list=provider_cost_list,config_random_cloud=config_random_cloud,seed=seed)
    general_config_path = f"cloud/run_{num_run}/general_config.json"
    with open(general_config_path, "r") as f:
        cloud_general_config = json.load(f)

    ges_nodes=cloud_general_config["ges_nodes"]              # total number of subproblems over all levels
    num_nodes_in_level=cloud_general_config["num_nodes_in_level"]  # nodes per region for each level
    predecessor_list=cloud_general_config["predecessor_list"]      # parent subproblem of each region
    num_leaves_ges=cloud_general_config["num_leaves"]
    result_dict={}

    # Sample the workflows (DAG with parallel branches, function attributes);
    # one copy of each workflow file is written per subproblem because the
    # configs are modified in place while descending the hierarchy.
    workflow_name_list=generate_random_workflows(num_workflows=num_workflows,config_random_workflows=config_random_workflows,provider_name=provider_list, num_opts=ges_nodes,num_run=num_run,seed=seed)
    # load the deployment configuration
    t1=time.time()
    comp_time=[]
    # ---- Decomposed solution: workflows are placed one after the other, each
    # one solved top-down through the hierarchy (level k=0 first). num_prob is
    # the running subproblem index used in all file names.
    for d,workflow_name in enumerate(workflow_name_list):
        #P_list = []
        num_prob = 0
        comp_time.append([])
        for k in range(num_levels):
            curr_node=0   # index of the parent region in level k-1
            ind=0         # node index within the parent region
            comp_time[-1].append([])
            for s in range(len(num_nodes_in_level[k])):
                curr_leave_k = -1
                if ind==num_nodes_in_level[k-1][curr_node]:
                    ind=0
                    curr_node+=1
                cloud_name = f"requirements_cloud_{k}_{s}.json"
                sol_name=f"opt_{d}"
                if not os.path.exists(os.path.join("placement", f"run_{num_run}")):
                    os.mkdir(os.path.join("placement", f"run_{num_run}"))

                path_cloud = os.path.join(f"cloud/run_{num_run}", cloud_name)
                path_workflow = os.path.join(f"workflows/run_{num_run}", workflow_name+"_"+str(num_prob)+".json")
                path_solution = os.path.join(f"placement/run_{num_run}", sol_name+"_"+str(num_prob)+".json")

                with open(path_cloud, "r") as f:
                    requirements_cloud_config = json.load(f)
                with open(path_workflow, "r") as f:#requirements_workflow_0.json
                    requirements_workflow_config = json.load(f)

                # Below the top level, inherit the parent's decisions: the
                # parent's placement P determines which functions/deployments
                # remain in this region (others become virtual placements on
                # the head node) and its selection H fixes the workflow choice
                # of this region's users (Eqs. (3)-(6) of the paper).
                if k>0:
                    path_predecessor = os.path.join(f"placement/run_{num_run}", sol_name + "_" + str(predecessor_list[k][s]) + ".json")
                    with open(path_workflow, "r") as f:  # requirements_workflow_0.json
                        requirements_workflow_config = json.load(f)
                    with open(path_predecessor, "r") as f:  # requirements_workflow_0.json
                        solution = json.load(f)
                    # Parent region's cloud config (level k-1, region curr_node): its
                    # per-leaf rates + selections feed the stochastic flow-conservation
                    # RHS and the head-node outside-routing distribution.
                    parent_cloud_path = os.path.join(f"cloud/run_{num_run}", f"requirements_cloud_{k-1}_{curr_node}.json")
                    with open(parent_cloud_path, "r") as f:
                        parent_cloud_config = json.load(f)
                    # Parent's workflow config: carries the parent's own outside
                    # stream (outside_rate / outside_routing), which is folded
                    # recursively into this region's outside stream (levels >= 2).
                    parent_workflow_config = None
                    if stochastic_selection:
                        path_parent_workflow = os.path.join(f"workflows/run_{num_run}", workflow_name + "_" + str(predecessor_list[k][s]) + ".json")
                        if os.path.exists(path_parent_workflow):
                            with open(path_parent_workflow, "r") as f:
                                parent_workflow_config = json.load(f)
                    if requirements_cloud_config["providers"]["provider_0"]["name"]=="tinyfaas":
                        curr_leave_k+=1
                    requirements_workflow_config=update_requirements_config(requirements_workflow_config,np.array(solution['P']),np.array(solution['H']),ind,curr_leave_k,requirements_cloud_config["num_nodes"],seed=seed,parent_cloud_config=parent_cloud_config,stochastic_selection=stochastic_selection,parent_workflow_config=parent_workflow_config)#P_list[predecessor_list[k][s]]
                    with open(path_workflow, "w") as f:
                        json.dump(requirements_workflow_config, f, indent=4)
                # Solve the regional placement subproblem (Eq. (1)) with Gurobi
                # and store all solution matrices and model parameters as JSON
                # so that selection_simulation.py can re-evaluate the costs.
                model,P,H,L,T,C,D,D_in,Time,a,b,Speed,S,t,M=solve_opt(requirements_workflow_config,requirements_cloud_config,gap=own_gap,w_1=w_1,w_2=w_2,w_3=w_3,provider_name_list=provider_list,provider_cost_list=provider_cost_list,relax_selection=stochastic_selection)
                solution=dict()
                solution['P']=P.tolist()
                solution['H']=H.tolist()
                solution['L'] = L.tolist()
                solution['T'] = T.tolist()
                solution['C'] = C.tolist()
                solution['D'] = D.tolist()
                solution['D_in'] = D_in.tolist()
                solution['time'] = Time.tolist()
                solution['a'] = a.tolist()
                solution['b'] = b.tolist()
                solution['Speed'] = Speed.tolist()
                solution['S']=S.tolist()
                comp_time[-1][-1].append(t)
                with open(path_solution, "w") as f:
                    json.dump(solution, f, indent=4)
                #P_list.append(P)
                num_prob += 1
                ind+=1

                # Remember the utilization this workflow adds to each node, so
                # that the next workflow's placement sees the occupied capacity.
                requirements_cloud_config=update_cloud_config(requirements_cloud_config,requirements_workflow_config,P,M)
                # Persist the region latency matrix L (a node-to-node parameter,
                # identical across workflows) alongside the node parameters so the
                # simulation reads latency from the cloud config and the placement
                # solution files stay immutable. See selection_simulation.py.
                requirements_cloud_config["L"] = L.tolist()
                with open(path_cloud, "w") as f:
                    json.dump(requirements_cloud_config, f, indent=4)
    t2=time.time()
    overall_time = [t2 - t1]
    # ---- Comparison run (Sec. IV-B/IV-C): solve the same workflows on the
    # flattened architecture containing every physical node, either with the
    # centralized optimization or with the all-on-provider-1 heuristic.
    if comparison:
        heuristic=parameter_general_config["heuristic"]
        comparison_time=[]
        # Build requirements_cloud_full.json: all physical (last-level) nodes
        # with end-to-end latencies reconstructed through the hierarchy.
        generate_full_cloud(cloud_general_config,num_run,provider_cost_list,num_levels)
        t1 = time.time()
        for s,workflow_name in enumerate(workflow_name_list):
            cloud_name = f"requirements_cloud_full.json"
            sol_name = f"opt_{s}"
            path_cloud = os.path.join(f"cloud/run_{num_run}", cloud_name)
            path_workflow = os.path.join(f"workflows/run_{num_run}", workflow_name + "_0.json")
            path_solution = os.path.join(f"placement/run_{num_run}", sol_name + "_full.json")

            with open(path_cloud, "r") as f:
                requirements_cloud_config = json.load(f)
            with open(path_workflow, "r") as f:  # requirements_workflow_0.json
                requirements_workflow_config = json.load(f)

            # Stochastic selection applies to the centralized optimization only
            # (the heuristic forces a trivial one-hot H, so it is left untouched).
            model,P,H,L,T,C,D,D_in,Time,a,b,Speed,S,t,M = solve_opt(requirements_workflow_config, requirements_cloud_config,comparison_gap,w_1,w_2,w_3,provider_list,provider_cost_list,heuristic,relax_selection=(stochastic_selection and not heuristic))
            solution = dict()
            solution['P'] = P.tolist()
            solution['H'] = H.tolist()
            solution['L'] = L.tolist()
            solution['T'] = T.tolist()
            solution['C'] = C.tolist()
            solution['D'] = D.tolist()
            solution['D_in'] = D_in.tolist()
            solution['time'] = Time.tolist()
            solution['a'] = a.tolist()
            solution['b'] = b.tolist()
            solution['Speed'] = Speed.tolist()
            solution['S'] = S.tolist()
            requirements_cloud_config = update_cloud_config(requirements_cloud_config, requirements_workflow_config, P, M)
            # Persist the global (flattened) latency matrix L into the full cloud
            # config so the simulation reads latency from here instead of the
            # immutable placement solution file.
            requirements_cloud_config["L"] = L.tolist()
            with open(path_cloud, "w") as f:
                json.dump(requirements_cloud_config, f, indent=4)
            comparison_time.append(t)
            with open(path_solution, "w") as f:
                json.dump(solution, f, indent=4)
        t2=time.time()

    # Persist the computation times: time_list[workflow][level][region] holds
    # the per-subproblem Gurobi solve times of the decomposed approach,
    # comparison_time the per-workflow times of the centralized/heuristic run.
    result_dict["time_list"]=comp_time#time_list
    result_dict["overall_time"]=overall_time
    if comparison:
        result_dict["comparison_time"] = comparison_time
    if not os.path.exists(os.path.join("results", f"run_{num_run}")):
        os.mkdir(os.path.join("results", f"run_{num_run}"))
    results_path = f"results/run_{num_run}/time.json"
    with open(results_path, "w") as f:
        json.dump(result_dict,f, indent=4)
