# Generators and helpers for the simulated case study (Sec. IV-A of the paper):
#  - generate_random_workflows(): random workflow DAGs with parallel branches
#    and function attributes sampled within the bounds of Table 2,
#  - generate_cloud(): a random hierarchical serverless architecture (levels of
#    regions/providers, Fig. 3a) written as one JSON config per (level, region),
#  - generate_full_cloud(): the flattened single-level architecture of all
#    physical nodes used by the centralized comparison (latencies recomposed
#    through the hierarchy as in Algorithm 1),
#  - update_*_config(): pass the decisions of an upper decomposition level down
#    to the subproblems of the next level.
import numpy as np
import json
import networkx as nx
import os
import re
# Add the utilization M_i caused by the just-placed workflow to the per-node
# "utilization" field, so the next workflow's placement starts from M_{i,0} > 0.
def update_cloud_config(requirements_cloud_config, requirements_workflow_config, P,M):
    for i in range(M.shape[0]):
        requirements_cloud_config["providers"][f"provider_{i}"]["utilization"]=requirements_cloud_config["providers"][f"provider_{i}"]["utilization"]+M[i][0]
    #num_functions = requirements_workflow_config["num_functions"]
    #for i in range(P.shape[1]):
    #    ram_add = 0
    #    for k in range(P.shape[0]):
    #        m = int(k % num_functions)
    #        ram_add += P[k, i] * requirements_workflow_config["functions"][f"function_{m}"]["ram"] #* \
    #                   #requirements_workflow_config["functions"][f"function_{m}"]["prob_request"]
    #    requirements_cloud_config["providers"][f"provider_{i}"]["max_Ram_curr"] = \
    #    requirements_cloud_config["providers"][f"provider_{i}"]["max_RAM_curr"] - ram_add
    return requirements_cloud_config


# Prepare the workflow config of a lower-level subproblem from the parent
# solution (decomposition, Sec. III-C): P/H are the parent's placement and
# selection matrices and current_node is this region's index (= its head node)
# within the parent problem. Deployments whose functions were all placed
# elsewhere are pruned, the parent's workflow selection is fixed via
# "selection_list" (Eq. (5)), and functions placed outside this region are
# recorded per function in "function_set" so solve_opt can pin them to the
# head node as virtual placements (Eq. (3)).
# Stochastic decomposition (Stage 2): given the parent selection matrix H
# (rows = parent leaves, cols = parent deployments), the parent node this region
# refines (current_node) and the local->parent deployment map (n_list), return
# per-local-deployment targets:
#   flow_rhs[l]        = lambda_rep * H_parent[rep, n_list[l]] -- the own-user
#                        demand for deployment n that this region's real leaves
#                        must reproduce (0 if the parent node is not a user leaf);
#   outside_routing[l] = rate-weighted routing of the outside/virtual demand to
#                        deployment n (a distribution over l).
# NOTE: the parent-deployment column index used here is n_list[l]; the exact
# cross-level deployment numbering must be validated empirically before the
# optimizer consumes these (Stage 2a).
def _stochastic_selection_targets(H, current_node, n_list, parent_cloud):
    num_local = len(n_list)
    flow_rhs = [0.0] * num_local
    outside_routing = [0.0] * num_local
    # head_has_own = the parent node this region refines is itself a user leaf, so
    # the region's head node carries real own-user demand to conserve. If it is a
    # non-leaf (e.g. a cloud node), the head's rate is phantom => own demand is 0
    # and its whole rate is virtual (handled in solve_opt).
    if parent_cloud is None or H is None or getattr(H, "shape", (0,))[0] == 0:
        return flow_rhs, outside_routing, True
    providers = parent_cloud["providers"]
    ncols = H.shape[1]

    def own_rate(j):
        p = providers[f"provider_{j}"]
        return float(p.get("request_rate", 0.0)) - float(p.get("virtual_request_rate", 0.0))

    parent_leaves = [j for j in range(parent_cloud["num_nodes"])
                     if providers[f"provider_{j}"].get("leave", False)]
    leaf_pos = {j: i for i, j in enumerate(parent_leaves)}
    head_has_own = current_node in leaf_pos

    if current_node in leaf_pos:
        rep = leaf_pos[current_node]
        lam_rep = own_rate(current_node)
        for l, n in enumerate(n_list):
            if 0 <= rep < H.shape[0] and 0 <= n < ncols:
                flow_rhs[l] = lam_rep * float(H[rep][n])

    others = [j for j in parent_leaves if j != current_node]
    tot = sum(own_rate(j) for j in others)
    if tot > 0:
        for l, n in enumerate(n_list):
            if 0 <= n < ncols:
                outside_routing[l] = sum(
                    own_rate(j) * float(H[leaf_pos[j]][n]) for j in others
                    if leaf_pos[j] < H.shape[0]) / tot
    return flow_rhs, outside_routing, head_has_own


def update_requirements_config(requirements_workflow_config, P,H, current_node,current_leave_k, num_nodes,seed=42,parent_cloud_config=None,stochastic_selection=False):
    np.random.seed(seed)
    num_functions = requirements_workflow_config["num_functions"]
    num_workflows = requirements_workflow_config["deployment_number"]
    n_list = []
    selection_list=[]
    if len(P) == 0:
        num_workflows = 0
    else:
        # Which deployments stay active in this region. Deterministic: only those
        # with a function on the head node. Stochastic: additionally keep every
        # deployment any parent leaf routes to (H>0), so the flow-conservation
        # policy is representable even if its functions all live elsewhere
        # (they then become virtual placements via function_set below).
        num_orig = num_workflows
        for n in range(num_orig):
            has_func_here = not all(P[n * num_functions:(n + 1) * num_functions, current_node] == 0)
            selected = (H.shape[0] > 0 and sum(H[:, n]) > 0)
            if has_func_here or (stochastic_selection and selected):
                n_list.append(n)
            else:
                num_workflows -= 1
        if stochastic_selection:
            # Stochastic policy propagation (Stage 2): instead of pinning a single
            # deployment for the whole region, pass the per-deployment demand the
            # real users of this region must reproduce (flow_rhs) and the routing
            # of the outside/virtual demand (outside_routing). Both are indexed by
            # the local (n_list) deployment order.
            flow_rhs, outside_routing, head_has_own = _stochastic_selection_targets(
                H, current_node, n_list, parent_cloud_config)
            requirements_workflow_config["flow_rhs"] = flow_rhs
            requirements_workflow_config["outside_routing"] = outside_routing
            requirements_workflow_config["head_has_own"] = bool(head_has_own)
            requirements_workflow_config["selection_list"] = {}
        else:
            # Deterministic: deployments selected by at least one upper-level leaf
            # stay selected for all users of this region (regional coupling).
            for n in range(num_workflows):
                if sum(H[:,n_list[n]])>0:
                    selection_list.append(n)
            requirements_workflow_config["selection_list"]=selection_list
        requirements_workflow_config["active_workflow_list"]=n_list
        # function_set[m] = deployments whose function m was NOT placed on this
        # region's head node, i.e. it runs in another region and only appears
        # here virtually (communication enters through the head node).
        for m in range(num_functions):
            liste = []
            for n in range(num_workflows):
                num = n_list[n]
                if P[num * num_functions + m, current_node] == 0:
                    liste.append(n)
            requirements_workflow_config["functions"][f"function_{m}"]["function_set"] = liste
    for m in range(num_functions):
        if requirements_workflow_config["functions"][f"function_{m}"]["data_dependencies"] != {}:
            key = \
            [key for key in requirements_workflow_config["functions"][f"function_{m}"]["data_dependencies"].keys()][0]
            #provider = np.random.randint(num_nodes)
            value = requirements_workflow_config["functions"][f"function_{m}"]["data_dependencies"][key]
            #requirements_workflow_config["functions"][f"function_{m}"]["data_dependencies"]={}
            #match = re.search(r'provider_(\d+)', key)
            #number = int(match.group(1))
            #if number==current_node:
            requirements_workflow_config["functions"][f"function_{m}"]["data_dependencies"][
            key] = value#f"provider_{provider}"
    requirements_workflow_config["deployment_number"] = num_workflows

    return requirements_workflow_config


# Build a random workflow DAG (Fig. 2 of the paper): functions 0 and
# n_nodes-1 are the common start/finish, the remaining functions are split
# into n_parallel_paths disjoint chains (the "branches"). Returns the graph
# and the list of branches (each a list of function indices).
def generate_connected_graph_with_paths(n_nodes, n_parallel_paths=2, seed=42):
    # assert n_nodes >= 4, "Need at least 4 nodes to have interesting parallel paths"
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    G = nx.DiGraph()

    nodes = list(range(n_nodes))
    start, finish = 0, n_nodes - 1
    G.add_nodes_from(nodes)

    # Add parallel paths
    available_nodes = set(nodes[1:-1])
    path_nodes = list(available_nodes)

    rng.shuffle(path_nodes)
    chunks = np.array_split(path_nodes, n_parallel_paths)
    path_list = []

    for chunk in chunks:
        # for m,c in enumerate(chunk):
        #    c= int(c)
        #    chunk[m]=c
        path = [start] + chunk.tolist() + [finish]
        path_list.append(path)
        for u, v in zip(path[:-1], path[1:]):
            G.add_edge(u, v)

    # Ensure connectivity by adding extra random edges
    n_extra_edges = rng.integers(low=1, high=n_nodes)
    for _ in range(n_extra_edges):
        u, v = rng.choice(nodes, 2, replace=False)
        if not G.has_edge(u, v) and u != v:
            G.add_edge(u, v)

    return G, path_list


# Sample num_workflows random workflows with attributes uniformly distributed
# within the bounds of config_random_workflows (Table 2): number of
# deployments, number of functions, per-function execution time, cloud speedup
# factor, RAM usage, output data size and (with probability
# data_dependencies_prob) a data dependency stored at one cloud provider.
# Each workflow is written num_opts times (one copy per decomposition
# subproblem, since the copies are modified independently while descending).
def generate_random_workflows(num_workflows,num_opts, config_random_workflows,provider_name, num_run=0,seed=42):
    np.random.seed(seed)
    if not os.path.exists(os.path.join("workflows",f"run_{num_run}")):
        os.mkdir(os.path.join("workflows",f"run_{num_run}"))
    deploy_min=config_random_workflows["deploy_min"]
    deploy_max=config_random_workflows["deploy_max"]
    func_min=config_random_workflows["func_min"]
    func_max = config_random_workflows["func_max"]
    input_data_max = config_random_workflows["input_data_max"]
    time_min  = config_random_workflows["time_min"]
    time_max  = config_random_workflows["time_max"]
    speedup_min  = config_random_workflows["speedup_min"]
    speedup_max  = config_random_workflows["speedup_max"]
    ram_min  = config_random_workflows["ram_min"]
    ram_max  = config_random_workflows["ram_max"]
    #prob_min  = config_random_workflows["prob_min"]
    #prob_max  = config_random_workflows["prob_max"]
    data_sent_min  = config_random_workflows["data_sent_min"]
    data_sent_max  = config_random_workflows["data_sent_max"]
    data_dependencies_prob  = config_random_workflows["data_dependencies_prob"]
    data_dependencies_max = config_random_workflows["data_dependencies_max"]
    data_dependecies_list = ['None', 0]

    workflow_name_list = []
    for i in range(num_workflows):
        workflow_name = f"requirements_workflow_{i}"
        workflow_name_list.append(workflow_name)
        config = dict()
        config["deployment_number"] = np.random.randint(deploy_min, deploy_max)
        num_functions = np.random.randint(func_min, func_max)
        config["num_functions"] = num_functions
        config["num_requests"] = {}
        config["selection_list"] = {}
        # for node in range(num_nodes):
        #    name=f"provider_{node}"
        #    value=np.random.uniform(0,1)
        #    config["num_requests"][name]=value
        # config["num_requests"]={}
        n_parallel_paths = np.random.randint(1, num_functions - 1)
        G, paths = generate_connected_graph_with_paths(num_functions, n_parallel_paths=n_parallel_paths, seed=i)
        config["functions"] = {}
        config["paths"] = {
            "num_paths": n_parallel_paths,
        }
        config["input_data"] = np.random.uniform(0, input_data_max)
        config["input_functions"] = ["function_0"]
        config["output_functions"] = [f"function_{num_functions - 1}"]
        for m in range(len(paths)):
            config["paths"].update({str(m): paths[m]})

        # Sample each function's attributes. With probability
        # data_dependencies_prob the function depends on data stored at one of
        # the cloud providers ("tinyfaas" = no dependency); accessing it from a
        # different provider later costs pricing_Storage_Transfer per MB.
        for k in range(num_functions):
            p_list=[(data_dependencies_prob)/(len(provider_name)-1) for k in range(len(provider_name)-1)]+[1-data_dependencies_prob]
            value=np.random.choice(len(provider_name), 1, p=p_list)[0]
            string = provider_name[value]#f"provider_{value}"
            if provider_name[value]=="tinyfaas":
                config["functions"][f"function_{k}"] = {
                    "name": f"function_{k}",
                    "time": np.random.uniform(time_min, time_max),
                    "speedup": np.random.uniform(speedup_min, speedup_max),
                    "ram": np.random.uniform(ram_min, ram_max),
                    #"prob_request": np.random.uniform(prob_min, prob_max),
                    "data_send": np.random.uniform(data_sent_min, data_sent_max),
                    "data_dependencies": {},  # np.random.uniform(1,3)
                    "function_set": {}
                }
            else:
                config["functions"][f"function_{k}"] = {
                    "name": f"function_{k}",
                    "time": np.random.uniform(time_min, time_max),
                    "speedup": np.random.uniform(speedup_min, speedup_max),
                    "ram": np.random.uniform(ram_min, ram_max),
                    #"prob_request": np.random.uniform(prob_min, prob_max),
                    "data_send": np.random.uniform(data_sent_min, data_sent_max),
                    "data_dependencies": {string: np.random.uniform(0, data_dependencies_max)},  # np.random.uniform(1,3)
                    "function_set": {}
                }

        # One identical copy per decomposition subproblem; copy k is later
        # specialized by update_requirements_config for subproblem k.
        for k in range(num_opts):
            path = os.path.join(f"workflows/run_{num_run}", f"requirements_workflow_{i}_{k}.json")
            with open(path, "w") as f:
                json.dump(config, f, indent=4)
    config_requirements={}

    config_requirements["num_workflows"]=num_workflows
    config_requirements["workflow_list"]=workflow_name_list
    path = os.path.join(f"workflows/run_{num_run}", f"workflow_config.json")
    with open(path, "w") as f:
        json.dump(config_requirements, f, indent=4)
    return workflow_name_list


# Sample the hierarchical serverless architecture (Fig. 3a / Fig. 5):
#  - Level 0 contains the top-level representatives: one head node per cloud
#    provider plus num_tiny_first edge ("tinyfaas") regions.
#  - On every following level each node of the previous level expands into a
#    region of nodes_min..nodes_max nodes of the same provider; node 0 of each
#    region is its head node (the link to the level above).
#  - Edge nodes are leaves (users send requests from them) and have finite
#    RAM; cloud nodes have infinite RAM but cost money. RAM and request rates
#    of a region are split among its children with a Dirichlet distribution so
#    that the totals are preserved across levels (cf. Eq. (2)).
#  - Latencies within a region shrink by base_decrease per level
#    (L_l,upper bound = L_nodes * b_decrease^-l, Sec. IV-A).
# Writes one requirements_cloud_<level>_<region>.json per subproblem and a
# general_config.json describing the whole hierarchy.
def generate_cloud(num_levels=2, num_run=0,provider_name=["aws","google", "tinyfaas"],num_tiny_first=2,provider_cost_list=[],provider_dev=[],config_random_cloud=[],seed=42):
    np.random.seed(42)
    if not os.path.exists(os.path.join("cloud",f"run_{num_run}")):
        os.mkdir(os.path.join("cloud",f"run_{num_run}"))
    speedup_min=config_random_cloud["speedup_min"]
    speedup_max=config_random_cloud["speedup_max"]
    speedup_deviation=config_random_cloud["speedup_dev"]
    nodes_min=config_random_cloud["nodes_min"]
    nodes_max= config_random_cloud["nodes_max"]
    lat_min= config_random_cloud["lat_min"]
    lat_max= config_random_cloud["lat_max"]
    ram_min=config_random_cloud["ram_min"]
    ram_max=config_random_cloud["ram_max"]
    user_request_rate_min=config_random_cloud["user_request_rate_min"]
    user_request_rate_max=config_random_cloud["user_request_rate_max"]

    base_decrease=config_random_cloud["base_decrease"]
    num_nodes_in_level = []
    num_leaves_in_level = []
    provider_list = []
    leave_list = []
    predecessor_list = []

    # ---- First pass: build the bookkeeping lists of the hierarchy.
    # num_nodes_in_level[l][region] = number of nodes in that region,
    # provider_list[l] = provider of every node on level l (flattened),
    # leave_list[l]    = True where requests can originate (head nodes / edge),
    # predecessor_list[l][region] = index of the parent subproblem.
    num_opt = 0
    node_list=[]
    for r in range(num_levels):
        if r == 0:
            num_tiny = num_tiny_first
            num_nodes = num_tiny_first+len(provider_name)-1
            node_list.append([[0]])
            num_nodes_in_level.append([num_nodes])
            provider_list.append([provider_name[k] for k in range(num_nodes - num_tiny)] + ["tinyfaas" for k in range(num_tiny)])
            num_leaves_in_level.append([num_tiny])
            leave_list.append([False for k in range(num_nodes - num_tiny)] + [True for k in range(num_tiny)])
            predecessor_list = [[-1]]
            num_opt=0
            curr_node=1

        else:
            # Every node of the previous level becomes one region with a random
            # number of nodes; non-edge regions expose only their head node as
            # "leaf" (it collects the requests of the levels above), while
            # every node of an edge (tinyfaas) region is a user-facing leaf.
            node_list.append([])
            num_nodes_in_level.append([])
            num_leaves_in_level.append([])
            provider_list.append([])
            leave_list.append([])
            predecessor_list.append([])
            ind = 0
            for k, num_node in enumerate(num_nodes_in_level[-2]):

                predecessor_list[-1] = predecessor_list[-1] + [k + num_opt for d in range(num_node)]
                node_list[-1] =node_list[-1]+ [d+curr_node for d in range(num_node)]


                for s in range(num_node):
                    value = np.random.randint(nodes_min, nodes_max)
                    num_nodes_in_level[-1].append(value)
                    #print(s)
                    if provider_list[-2][ind] != "tinyfaas":
                        num_leaves_in_level[-1].append(1)
                    else:
                        num_leaves_in_level[-1].append(value)

                    for m in range(num_nodes_in_level[-1][-1]):
                        provider_list[-1].append(provider_list[-2][ind])
                        if m == 0:
                            leave_list[-1].append(True)
                        else:
                            if provider_list[-1][-1] != "tinyfaas":
                                leave_list[-1].append(False)

                            else:
                                leave_list[-1].append(True)
                    ind += 1
                curr_node += num_node
            num_opt+=len(num_nodes_in_level[-2])

    curr_node = 0
    # Physical user leaves = all edge nodes of the last (physical) level.
    num_leaves_ges=0
    for provider in provider_list[-1]:
        if provider == "tinyfaas":
            num_leaves_ges+=1
    # ---- Second pass: write the per-region cloud config files with sampled
    # latencies, RAM budgets, request rates, speedups and provider pricing.
    latencies_list=[[]]
    speedup_list=[]
    for k in range(num_levels):

        if k == 0:

            # Level 0: one single region holding all top-level head nodes.
            # Latencies are sampled symmetrically (L[r,t] = L[t,r]).
            num_leaves = num_leaves_in_level[k][0]
            num_clouds_in_level = 1
            num_nodes = num_nodes_in_level[k][0]  # np.random.randint(3,10)
            L = np.zeros((num_nodes, num_nodes))
            rand_latencies = np.random.uniform(lat_min / (base_decrease ** (k)), lat_max / (base_decrease ** (k)), (int(num_nodes * (num_nodes - 1) / 2), 1))
            ind = 0

            for r in range(num_nodes):
                for t in range(r, num_nodes):
                    if r != t:
                        L[r, t] = rand_latencies[ind]
                        L[t, r] = rand_latencies[ind]
                        ind += 1

            #predecessor_list = ["None" for k in range(num_nodes)]
            nodes_list = [[k for k in range(num_nodes)]]
            name_list = [["aws"]]
            # curr_node+=num_nodes
            cloud_name = f"requirements_cloud_{k}_{0}.json"
            path = os.path.join(f"cloud/run_{num_run}", cloud_name)
            speedup_list.append([np.random.uniform(speedup_min, speedup_max) for k in range(num_nodes)])
            # var_prob=np.random.choice(0,1,p=[0.2])
            config = {
                "name": f"cloud_{k}_{0}",
                "num_nodes": num_nodes,
                "num_leaves": num_leaves,
                "providers": {
                    "provider_" + str(0): {
                        "name": provider_list[0][0],
                        "leave": leave_list[0][0],
                        "estimated_latency": list(L[0, :]),
                        "max_RAM_curr": np.random.uniform(ram_min,ram_max),#np.inf,
                        "request_rate": np.random.uniform(user_request_rate_min,user_request_rate_max),
                        "utilization":0,
                        "speedup":speedup_list[-1][0],
                        "pricing_Storage_Transfer": provider_cost_list[provider_list[0][0]]["pricing_Storage_Transfer"],
                        "pricing_RAM": provider_cost_list[provider_list[0][0]]["pricing_RAM"],  # 2.5e-06,
                        "pricing_data_sent": provider_cost_list[provider_list[0][0]]["pricing_data_sent"],
                    }
                },
            }
            latencies_list[-1].append(list(L[0, :]))
            curr_node += 1
            ram_list = [[np.inf]]
            request_list=[[0]]
            for s in range(num_nodes - 1):
                #var_prob = np.random.choice(2, 1, p=[0, 1.0])[0]
                name_list[0].append("tinyfaas")
                #if var_prob == 0:
                request_list[0].append(np.random.uniform(user_request_rate_min,user_request_rate_max))
                ram_list[0].append(np.random.uniform(ram_min,ram_max))#np.inf)
                #else:
                #    ram_list[0].append(np.random.uniform(2000, 10000))
                config["providers"].update({
                    "provider_" + str(s + 1): {
                        "name": provider_list[0][s + 1],
                        "estimated_latency": list(L[s + 1, :]),  # np.random.uniform(0.01, 0.1),
                        "leave": leave_list[0][s + 1],  # provider_name[var_prob] == "tinyfaas",
                        "max_RAM_curr": ram_list[0][s + 1],
                        "request_rate":request_list[0][s+1],
                        "utilization": 0,
                        "speedup":speedup_list[-1][s+1],
                        "pricing_Storage_Transfer": provider_cost_list[provider_list[0][s+1]]["pricing_Storage_Transfer"],
                        "pricing_RAM": provider_cost_list[provider_list[0][s+1]]["pricing_RAM"],  # 2.5e-06,
                        "pricing_data_sent": provider_cost_list[provider_list[0][s+1]]["pricing_data_sent"],
                    }
                })
                curr_node += 1
                latencies_list[-1].append(list(L[s + 1, :]),)
            with open(path, "w") as f:
                json.dump(config, f, indent=4)

        else:
            # Lower levels: one config file per region m. Latency bounds shrink
            # with base_decrease**k (connections become increasingly local).
            num_clouds_in_level = sum(num_nodes_in_level[k - 1])  # len(nodes_list[-1])
            name_list.append([])
            ram_list.append([])
            request_list.append([])
            latencies_list.append([])
            speedup_list.append([])
            curr_node_level = 0
            for m in range(sum(num_nodes_in_level[k - 1])):
                num_leaves = num_leaves_in_level[k][m]
                num_nodes = num_nodes_in_level[k][m]  # np.random.randint(3,10)
                L = np.zeros((num_nodes, num_nodes))
                for r in range(num_nodes):
                    speedup_list[-1].append(speedup_list[-2][m]+np.random.uniform(-speedup_deviation,speedup_deviation))
                rand_latencies = np.random.uniform(lat_min / (base_decrease ** (k)), lat_max / (base_decrease ** (k)),
                                                   (int(num_nodes * (num_nodes - 1) / 2), 1))
                ind = 0
                for r in range(num_nodes):
                    for t in range(r, num_nodes):
                        if r != t:
                            L[r, t] = rand_latencies[ind]
                            L[t, r] = rand_latencies[ind]
                            ind += 1
                # predecessor_list.append([m for k in range(num_nodes)])
                # Split the parent's RAM budget and request rate randomly among
                # the region's nodes (Dirichlet keeps the totals constant);
                # cloud regions (infinite RAM) generate no user requests.
                request_overall = sum(request_list[-2])
                if ram_list[-2][m] == np.inf:
                    ram = [np.inf for k in range(num_nodes)]
                    request=[0 for k in range(num_nodes)]
                else:
                    request_sum = request_list[-2][m]

                    request = (request_sum * np.random.dirichlet(np.ones(num_nodes))).tolist()
                    ram_sum = ram_list[-2][m]
                    ram = (ram_sum * np.random.dirichlet(np.ones(num_nodes))).tolist()
                ram_list[-1] = ram_list[-1].copy() + ram.copy()
                request_list[-1]=request_list[-1].copy()+request.copy()
                # name=name_list[-2][m]

                # name_list[-1].append([name for k in range(num_nodes)])
                nodes_list.append([k + curr_node for k in range(num_nodes)])
                # curr_node+=num_nodes
                cloud_name = f"requirements_cloud_{k}_{m}.json"
                path = os.path.join(f"cloud/run_{num_run}", cloud_name)
                # provider_0 is the region's head node; it additionally carries
                # the request rate of all other regions (request_overall),
                # acting as placeholder for outside requests (Eq. (7)).
                config = {
                    "name": f"cloud_{k}_{m}",
                    "num_nodes": num_nodes,
                    "num_leaves": num_leaves,
                    "providers": {
                        "provider_" + str(0): {
                            "name": provider_list[k][curr_node_level],  # name,
                            "leave": leave_list[k][curr_node_level],
                            "estimated_latency": list(L[:, 0]),
                            "speedup":speedup_list[-1][m],
                            "max_RAM_curr": ram[0],
                            "utilization": 0,
                            "request_rate":request[0]+request_overall,
                            # Split of the head-node rate into the region's OWN user
                            # share (request[0]) and the virtual/outside demand
                            # (request_overall, the Eq. 7 placeholder). Stored so the
                            # stochastic decomposition can treat node 0 as a normal
                            # leaf for its own users and the virtual stream separately.
                            "virtual_request_rate":request_overall,
                            "pricing_Storage_Transfer": provider_cost_list[provider_list[k][curr_node_level]]["pricing_Storage_Transfer"]+np.random.uniform(-provider_dev["pricing_Storage_Transfer"],provider_dev["pricing_Storage_Transfer"]),
                            "pricing_RAM": provider_cost_list[provider_list[k][curr_node_level]]["pricing_RAM"]+np.random.uniform(-provider_dev["pricing_RAM"],provider_dev["pricing_RAM"]),  # 2.5e-06,
                            "pricing_data_sent": provider_cost_list[provider_list[k][curr_node_level]]["pricing_data_sent"]+np.random.uniform(-provider_dev["pricing_data_sent"],provider_dev["pricing_data_sent"]),
                        }
                    },
                }
                latencies_list[-1].append(list(L[:, 0]))
                curr_node += 1
                curr_node_level += 1
                for s in range(num_nodes - 1):
                    config["providers"].update({
                        "provider_" + str(s + 1): {
                            "name": provider_list[k][curr_node_level],
                            "estimated_latency": list(L[:, s + 1]),
                            "leave": leave_list[k][curr_node_level],
                            "speedup":speedup_list[-1][m],
                            # name == "tinyfaas" and s+1>=num_nodes-num_leaves,
                            "max_RAM_curr": ram[s + 1],
                            "utilization": 0,
                            "request_rate":request[s+1],
                            "pricing_Storage_Transfer": provider_cost_list[provider_list[k][curr_node_level]]["pricing_Storage_Transfer"]+np.random.uniform(-provider_dev["pricing_Storage_Transfer"],provider_dev["pricing_Storage_Transfer"]),
                            "pricing_RAM": provider_cost_list[provider_list[k][curr_node_level]]["pricing_RAM"]+np.random.uniform(-provider_dev["pricing_RAM"],provider_dev["pricing_RAM"]),  # 2.5e-06,
                            "pricing_data_sent": provider_cost_list[provider_list[k][curr_node_level]]["pricing_data_sent"]+np.random.uniform(-provider_dev["pricing_data_sent"],provider_dev["pricing_data_sent"])
                        }
                    })
                    curr_node += 1
                    curr_node_level += 1
                    latencies_list[-1].append(list(L[:, s+1]))
                with open(path, "w") as f:
                    json.dump(config, f, indent=4)
    # Store the hierarchy description ("ges" = gesamt, i.e. total) used by
    # placement.py, selection_simulation.py and generate_full_cloud().
    ges_nodes = 0
    for k in range(num_levels):
        for value in num_nodes_in_level[k]:
            ges_nodes += value
    cloud_general_config={}
    cloud_general_config["num_leaves"]=num_leaves_ges
    cloud_general_config["ges_nodes"] = ges_nodes
    cloud_general_config["node_list"] = node_list
    cloud_general_config["predecessor_list"] = predecessor_list
    cloud_general_config["leave_list"] = leave_list
    cloud_general_config["ram_list"]=ram_list
    cloud_general_config["request_list"]=request_list
    cloud_general_config["speedup_list"] = speedup_list
    cloud_general_config["provider_list"] = provider_list
    cloud_general_config["latencies_list"] = latencies_list
    cloud_general_config["num_nodes_in_level"] = num_nodes_in_level
    cloud_general_config["num_leaves_in_level"] = num_leaves_in_level
    path_general_config=f"cloud/run_{num_run}/general_config.json"
    with open(path_general_config, "w") as f:
        json.dump(cloud_general_config, f, indent=4)

# Flatten the hierarchy into one architecture containing only the physical
# (last-level) nodes - the input of the centralized comparison and of the
# selection simulation. End-to-end latencies between nodes of different
# regions are recomposed by walking up the predecessor tree until the two
# nodes share a region, summing the head-node link latencies on the way
# (this is Algorithm 1 of the paper).
def generate_full_cloud(config,num_run,provider_cost_list,num_levels):
    path=os.path.join(f"cloud/run_{num_run}", "requirements_cloud_full.json")
    full_config={}
    predecessor_list=config["predecessor_list"]
    num_nodes=sum(config["num_nodes_in_level"][-1])
    full_config["num_nodes"]=num_nodes
    full_config["num_leaves"] = config["num_leaves"]
    full_config["providers"]={}
    ram_list=config["ram_list"][-1]
    request_list=config["request_list"][-1]
    speedup_list=config["speedup_list"][-1]
    leave_list=config["leave_list"][-1]
    provider_list=config["provider_list"][-1]
    num_nodes_in_level=config["num_nodes_in_level"]
    # k iterates over all physical nodes; ind_k is the region of node k on the
    # last level and curr_k its local index within that region.
    curr_len_k=num_nodes_in_level[-1][0]
    curr_k=0
    ind_k=0
    for k in range(len(provider_list)):
        if curr_k==curr_len_k:
            ind_k+=1
            curr_len_k=num_nodes_in_level[-1][ind_k]
            curr_k=0
        full_config["providers"][f"provider_{k}"]={}
        full_config["providers"][f"provider_{k}"]["name"]=provider_list[k]
        full_config["providers"][f"provider_{k}"]["leave"] = (leave_list[k] and (provider_list[k]=="tinyfaas"))
        full_config["providers"][f"provider_{k}"]["max_RAM_curr"] = ram_list[k]
        full_config["providers"][f"provider_{k}"]["request_rate"] = request_list[k]
        full_config["providers"][f"provider_{k}"]["speedup"] =  speedup_list[k]
        full_config["providers"][f"provider_{k}"]["utilization"] = 0
        full_config["providers"][f"provider_{k}"]["pricing_Storage_Transfer"] = provider_cost_list[provider_list[k]]["pricing_Storage_Transfer"]
        full_config["providers"][f"provider_{k}"]["pricing_RAM"] = provider_cost_list[provider_list[k]]["pricing_RAM"]
        full_config["providers"][f"provider_{k}"]["pricing_data_sent"] = provider_cost_list[provider_list[k]]["pricing_data_sent"]
        level_start=[]
        num_opt=0
        for r in range(num_levels):
            level_start.append(num_opt)
            num_opt+=len(predecessor_list[r])

        # Latency from node k to every other physical node m (Algorithm 1):
        # same region -> direct intra-region latency; otherwise climb both
        # predecessor chains in lockstep, adding each node's latency to its
        # region's head node (index 0), until a common region is found, where
        # the remaining gap is bridged by that region's internal latency.
        latencies_list=[]
        curr_len_m = num_nodes_in_level[-1][0]
        curr_m = 0
        ind_m = 0
        for m in range(len(provider_list)):
            if curr_m == curr_len_m:
                ind_m += 1
                curr_len_m = num_nodes_in_level[-1][ind_m]
                curr_m=0
            latency=0
            pred_k=ind_k
            pred_m=ind_m
            if pred_k==pred_m:
                latency=config["latencies_list"][-1][k][curr_m]
            else:
                latency+=config["latencies_list"][-1][m][0]
                latency+=config["latencies_list"][-1][k][0]
                next_ind_k=ind_k
                next_ind_m=ind_m
                pred_k=predecessor_list[-1][ind_k]  - level_start[-2]
                pred_m=predecessor_list[-1][ind_m]  - level_start[-2]

                for r in range(num_levels-1):

                    if pred_k==pred_m:
                        find_m=next_ind_m
                        for num_node in num_nodes_in_level[num_levels-r-2]:
                           find_m=find_m-num_node
                           if find_m<0:
                               find_m+=num_node
                               break
                        latency += config["latencies_list"][num_levels -r - 2][next_ind_k][find_m]
                        break
                    else:
                        next_ind_k=pred_k
                        next_ind_m=pred_m
                        pred_k=predecessor_list[num_levels -r -2][next_ind_k]  - level_start[num_levels - r - 2]
                        pred_m=predecessor_list[num_levels -r -2][next_ind_m]  - level_start[num_levels - r - 2]
                        latency += config["latencies_list"][num_levels - r - 1][next_ind_k][0]
                        latency += config["latencies_list"][num_levels - r - 1][next_ind_m][0]
            curr_m+=1
            latencies_list.append(latency)
        curr_k+=1
        full_config["providers"][f"provider_{k}"]["estimated_latency"]=latencies_list
    with open(path, "w") as f:
        json.dump(full_config, f, indent=4)