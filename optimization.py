# %%
"""Core optimization module: builds and solves the workflow placement problem.

This implements the mixed-integer nonlinear program (MINLP) of Eq. (1) in
"Large-scale workflow placement in serverless computing using integer
nonlinear programming" (Adamek et al.). One call of solve_opt() places a
single workflow (with all its deployments) on one set of nodes - either a
regional subproblem of the decomposition (Sec. III-C) or the full
centralized architecture, depending on the cloud config that is passed in.
"""
from gurobipy import *
import numpy as np
import os
from typing import Any, Dict
import time

# %%
def solve_opt(deployment_workflow: Dict[str, Any],deployment_cloud: Dict[str, Any], gap=0,w_1=1,w_2=1,w_3=1,provider_name_list=[],provider_cost_list=[],heuristic=False,relax_selection=False):
    """Solve the placement + selection problem (Eq. (1) of the paper) with Gurobi.

    Parameters
    ----------
    deployment_workflow : dict
        Workflow description (functions with time/ram/speedup/data attributes,
        parallel paths/branches, number of deployments, input/output functions,
        possibly a fixed "selection_list" and "function_set" inherited from the
        upper decomposition level).
    deployment_cloud : dict
        Node description ("providers"): per node its provider name, latency row,
        RAM capacity, request rate, current utilization and pricing.
    gap : float
        Gurobi MIPGap (0 = solve to optimality).
    w_1, w_2, w_3 : float
        Weights of the monetary cost M, basic time cost B and utilization cost U
        in the weighted-sum objective of Eq. (1).
    provider_name_list, provider_cost_list :
        Global provider names and their pricing (used for data-dependency
        storage-transfer costs that reference providers by name).
    heuristic : bool
        If True, skip optimization and use the baseline heuristic that places
        every function (and selects every workflow) on node 0, i.e. the first
        cloud provider (Sec. IV-C).

    Returns
    -------
    model, P, H, L, T, C, D, D_in, Time, a, b, Speed, S, t, M
        The Gurobi model, the solution matrices (placement P, selection H,
        branch times T, bilinear data-transfer costs S, node utilizations M),
        the model parameters (latencies L, execution costs C, data-transfer
        costs D, input-data costs D_in, execution times "Time", successor
        matrix a, branch membership b, speedup factors Speed) and the wall-clock
        solve time t.
    """

    workflow_functions = list(deployment_workflow['functions'].keys())
    providers = list(deployment_cloud['providers'].keys())
    num_paths = deployment_workflow['paths']['num_paths']
    num_leaves = deployment_cloud['num_leaves']
    num_workflows = deployment_workflow['deployment_number']
    num_functions = len(deployment_workflow['functions'])
    # Count cloud nodes vs. edge nodes ("tinyfaas"). Edge nodes are free but
    # RAM-limited; cloud nodes cost money but offer a speedup (Sec. II-B).
    num_nodes_cloud = 0
    num_nodes_tiny = 0
    for key in deployment_cloud['providers'].keys():
        if deployment_cloud['providers'][key]['name'] != 'tinyfaas':
            num_nodes_cloud += 1
        else:
            num_nodes_tiny += 1
    num_nodes = num_nodes_cloud + num_nodes_tiny
 


    model= Model("FaaS")
    if gap>0:
        model.setParam('MIPGap', gap)
    # Model parameters (cf. Table 1 of the paper):
    C=np.zeros((num_functions,num_nodes))         # C_{m,i}: cost to execute function m on node i
    D=np.zeros((num_functions, num_nodes,num_nodes))  # D_{m,i,j}: cost to send output of m from node i to node j
    L=np.zeros((num_nodes, num_nodes))            # L_{i,j}: latency between nodes i and j
    b=np.zeros((num_paths,num_functions))         # b_{c,m}: function m belongs to branch/path c
    a=np.zeros((num_functions,num_functions))     # a_{m,m~}: function m~ directly succeeds function m
    Speed=np.ones((num_functions, num_nodes))     # x_{i,m}: execution-time factor of m on node i (<1 = speedup on cloud)
    ram_max=np.ones((num_nodes,1))*1e8            # RAM_max,i: node RAM capacity (effectively inf for cloud nodes)
    request=np.zeros((num_nodes,1))               # lambda_k: request rate of each leaf (user) node
    ram=np.zeros((num_functions,1))               # RAM_m: memory usage of function m
    Time=np.zeros((num_functions,1))              # t_func,m: base execution time of function m
    #prob_request=np.zeros((num_functions,1))
    D_in=np.zeros((num_nodes,1))                  # cost of receiving the workflow input data d_in on node i
    # Indices of the first (input-receiving) and last (output-sending) functions
    # of the workflow; they incur the user<->node latency and data costs.
    input_list=deployment_workflow['input_functions']
    m_start=[list(deployment_workflow['functions']).index(func) for func in input_list]
    end_list=deployment_workflow['output_functions']
    m_end=[list(deployment_workflow['functions']).index(func) for func in end_list]
    leave_nodes=[]
    # Cloud nodes execute functions faster: apply the function's speedup factor
    # (edge/tinyfaas nodes keep the base time, factor 1).
    for m in range(num_functions):
        for i in range(num_nodes):
            if deployment_cloud['providers'][f"provider_{i}"]['name']!="tinyfaas":
                Speed[m,i]=deployment_workflow["functions"][f"function_{m}"]['speedup']#+deployment_cloud["providers"][f"provider_{i}"]['speedup']
    # Leaf nodes are the user-facing nodes from which requests originate.
    for r in range(len(deployment_cloud['providers'])):
        if deployment_cloud['providers'][f"provider_{r}"]['leave']:
            leave_nodes.append(r)
    #num_leaves=len(leave_nodes)
    #for m in range(num_functions):
    #    func_name=workflow_functions[m]#deployment_workflow['functions'][m]
    #    list_func=deployment_workflow['workflow_dependencies'][func_name]
    #    indices = [list(deployment_workflow['functions']).index(func) for func in list_func]
    #    a[m,indices]=1
    # Build branch membership b and successor matrix a from the parallel paths
    # of the workflow DAG (Fig. 2 in the paper).
    for s in range(num_paths):
        list_func=deployment_workflow['paths'][str(s)]
        indices= [list(deployment_workflow['functions']).index(f"function_{func}") for func in list_func]
        b[s,indices]=1
        for r in range(len(list_func)-1):
            a[list_func[r],list_func[r+1]]=1
    # Only edge (tinyfaas) nodes have a finite RAM budget left.
    for i in range(num_nodes):
        if deployment_cloud['providers'][f"provider_{i}"]["name"]=='tinyfaas':

            ram_max[i]=deployment_cloud['providers'][f"provider_{i}"]['max_RAM_curr']
    # Request rate lambda_k of every leaf; in subproblems of the decomposition
    # this already aggregates the rates of the underlying region (Eqs. (2)/(7)).
    for k in range(num_leaves):
        request[k]=deployment_cloud['providers'][f"provider_{leave_nodes[k]}"]["request_rate"]
    # ---- Stage 2 stochastic decomposition. When a lower-level subproblem carries
    # an inherited flow policy (flow_rhs from update_requirements_config), node 0
    # (leaf 0, the head) is treated as a normal leaf for its OWN users only (its
    # rate is de-conflated by subtracting the virtual/outside demand), and the
    # virtual demand is added separately as a fixed pseudo-leaf routed by
    # outside_routing. Absent flow_rhs (top-level decomposed / centralized) =>
    # unchanged Stage-1 behavior.
    stochastic_decomp = relax_selection and ("flow_rhs" in deployment_workflow)
    head_has_own = deployment_workflow.get("head_has_own", True)
    flow_rhs = deployment_workflow.get('flow_rhs', [0.0] * num_workflows)
    outside_routing = deployment_workflow.get('outside_routing', [0.0] * num_workflows)
    # True outside-user demand of this region (sum of the OTHER parent leaves' own
    # rates = the normalizer of outside_routing). Used as the virtual rate so the
    # inside:outside weighting is correct: virtual_rate * outside_routing[n] equals
    # the actual rate of outside users choosing deployment n.
    outside_rate = float(deployment_workflow.get('outside_rate', 0.0))
    head_virtual = float(deployment_cloud['providers']['provider_0'].get('virtual_request_rate', 0.0))
    # Head node (leaf 0) de-conflation. head_has_own (set by
    # _stochastic_selection_targets) is True only if the parent node this region
    # refines is a PHYSICAL user leaf (leave flag AND tinyfaas): then the head's
    # own share is (head_rate - request_overall). Cloud-region heads with
    # leave=True are mere request collectors -- their rate is phantom demand the
    # generation parked on non-leaf nodes (it never reaches a physical user
    # leaf and is never simulated), so own = 0. This keeps Sum_k own_k =
    # Sum_n flow_rhs exact and the flow conservation feasible everywhere.
    # The outside load is weighted by the true outside demand `outside_rate`
    # (recursively aggregated over all levels, NOT the Eq. 7 placeholder
    # request_overall), so inside:outside is weighted exactly.
    if stochastic_decomp:
        virtual_rate = outside_rate
        own_head = (request[0][0] - head_virtual) if head_has_own else 0.0
    else:
        virtual_rate = 0.0
        own_head = 0.0
    # Effective per-leaf rate for the H-weighted objective/utilization terms.
    lam = [(own_head if (stochastic_decomp and leave_nodes[k] == 0) else request[k][0])
           for k in range(num_leaves)]
    # Assemble the per-function cost parameters C, D and D_in from the node
    # pricing (cf. Sec. II-B: C_{m,i} = p_RAM,i*R_func,m + p_data,i*d_data,m and
    # D_{m,i,j} = (p_send,i + p_send,j)*d_sent,m).
    for j in range( num_functions):
        ram[j] =  deployment_workflow['functions'][ workflow_functions[j]]['ram']
        Time[j] =  deployment_workflow['functions'][ workflow_functions[j]]['time']
        #requests =  deployment_workflow['functions'][ workflow_functions[j]]['requests']
        data_dependencies =  deployment_workflow['functions'][ workflow_functions[j]]['data_dependencies']
        #prob_request[j]= deployment_workflow['functions'][ workflow_functions[j]]['prob_request']
        data_send = deployment_workflow['functions'][ workflow_functions[j]]['data_send']
        for k in range( num_nodes):
            # Receiving the user's input data is free on edge nodes, priced per
            # MB on cloud nodes.
            if deployment_cloud['providers'][f"provider_{k}"]["name"]=='tinyfaas':
                #D[j,k]=0
                D_in[k]=0
            else:
                pricing_data_sent = deployment_cloud['providers'][providers[k]]['pricing_data_sent']
                #D[j,k]=prob_request[j]*data_send*pricing_data_sent
                D_in[k]=deployment_workflow['input_data']*pricing_data_sent
            # Data transfer between two nodes of different providers costs the
            # sender's plus the receiver's egress price (free for tinyfaas).
            for m in range(num_nodes):

                    if deployment_cloud['providers'][f"provider_{k}"]["name"]!=deployment_cloud['providers'][f"provider_{m}"]["name"]:
                        if deployment_cloud['providers'][f"provider_{k}"]["name"]!='tinyfaas':
                            pricing_data_sent = deployment_cloud['providers'][providers[k]][
                            'pricing_data_sent']
                            D[j,k,m]=D[j,k,m]+data_send*pricing_data_sent# *prob_request[j]
                        if deployment_cloud['providers'][f"provider_{m}"]["name"]!='tinyfaas':
                            pricing_data_sent = deployment_cloud['providers'][providers[m]][
                            'pricing_data_sent']
                            D[j,k,m]=D[j,k,m]+data_send*pricing_data_sent #*prob_request[j]
            # Dependency data stored at provider "key" must be transferred if
            # function j runs on a node of a different provider
            # (p_dependencies * d_dependencies, Sec. II-A).
            for key in data_dependencies.keys():
                value = data_dependencies[key]
                #provider_name=provider_name_list[int(key[9:])]
                pricing_Storage_Transfer =  provider_cost_list[key]["pricing_Storage_Transfer"]#deployment_cloud['providers'][key][
                    #'pricing_Storage_Transfer']
                if key!=  deployment_cloud["providers"][providers[k]]["name"]:
            #        D[j,k]=data_send
                    C[j,k]=C[j,k]+pricing_Storage_Transfer*value
            # Direct execution cost: RAM-seconds (R_func,m = RAM_m * t_func,m)
            # priced per provider; execution on edge nodes is free.
            if deployment_cloud['providers'][f"provider_{k}"]["name"]=='tinyfaas':
                C[j,k]+=0
            else:
                pricing_RAM =  deployment_cloud['providers'][ providers[k]]['pricing_RAM']
                #pricing_StartRequest =  deployment_cloud['providers'][ providers[k]]['pricing_StartRequest']
                C[j,k]+=pricing_RAM*ram[j]*Time[j]#Speed#prob_request[j]*#+pricing_StartRequest*requests

    # Pairwise node latencies L_{i,j} as estimated in the cloud config.
    for k in range( num_nodes):
        for m in range( num_nodes):
            L[k,m]= deployment_cloud['providers'][f"provider_{k}"]['estimated_latency'][m]
    #for j in range( num_functions):
    #    for k in range( num_nodes):
    #        if k>= num_nodes_tiny:
    #            C[j,k]=1
    #        if k!=1:
    #            D[j,k]=2
    #for k in range( num_nodes):
    #    for m in range( num_nodes):
    #        if k!=m:
    #            L[k,m]=1
    #w_1=1
    #w_2=1
    #w_3=1
    if heuristic:
        # Baseline heuristic (Sec. IV-C): place every function of every
        # deployment on node 0 (the first cloud provider) and let every leaf
        # select the first deployment. No optimization is solved.
        t1=time.time()
        P_list = []
        # for model in  model_list:

        P = np.zeros((num_functions * num_workflows, num_nodes))
        S = np.zeros((num_functions * num_workflows, num_nodes))
        H = np.zeros((num_leaves, num_workflows))
        T = np.zeros((num_workflows, 1))
        M=np.zeros((num_nodes,1))
        p = np.zeros((num_workflows,num_functions))

        for l in range(len(P)):
            P[l,0]=1
        for l in range(len(H)):
            H[l,0]=1
        t2=time.time()
        t=t2-t1

    else:
        # Decision variables of Eq. (1) (deployments are indexed n, functions m,
        # nodes i; rows of P/S/p are flattened as m + n*num_functions):
        #   P[m+n*F, i] : binary placement P_{n,m,i} - function m of deployment n on node i
        #   S[m+n*F, i] : continuous bilinear helper S_{n,m,i} - monetary cost of sending
        #                 the output of function m from node i to its successors' nodes
        #   H[k, n]     : binary selection H_{k,n} - leaf (user) k requests deployment n
        #   T[n]        : evaluation time of deployment n (>= every branch time T_{n,c})
        #   M[i]        : RAM utilization M_i of node i (penalized quadratically)
        #   p[m+n*F]    : function utilization u_{n,m} (penalized quadratically)
        P=model.addVars( num_functions* num_workflows, num_nodes,vtype=GRB.BINARY, name="P")
        S = model.addVars(num_functions * num_workflows, num_nodes, vtype=GRB.CONTINUOUS, name="S")
        # Selection H_{k,n}: binary one-hot by default; when relax_selection is
        # set, H becomes a continuous stochastic policy in [0,1] (probability that
        # leaf k routes a request to deployment n). The objective is already an
        # expectation over H, so only the variable type changes; the w_3
        # utilization penalty is what keeps the relaxed policy non-degenerate.
        if relax_selection:
            H=model.addVars(num_leaves,num_workflows,lb=0,ub=1,vtype=GRB.CONTINUOUS, name="H")
            model.setParam('NonConvex', 2)
        else:
            H=model.addVars(num_leaves,num_workflows,vtype=GRB.BINARY, name="H")
        T=model.addVars(num_workflows,lb=0, ub=float('inf'),vtype=GRB.CONTINUOUS, name="T")
        M = model.addVars(num_nodes, vtype=GRB.CONTINUOUS, name="M")
        p = model.addVars(num_functions*num_workflows,vtype=GRB.CONTINUOUS,name="p")
        obj=0
        # Expected-cost part of the objective: every term is weighted by the
        # request rate lambda_k and the selection H_{k,n} of the requesting leaf.
        # w_2 * T[n] is the basic time cost B; w_1 * (C + S) is the monetary
        # cost M of executing functions and transferring data between them.
        for n in range( num_workflows):
            for k in range( num_leaves):
                obj=obj+lam[k]*H[k,n]*w_2*T[n]
                for m in range( num_functions):
                    for i in range(num_nodes):
                        obj=obj+lam[k]*H[k,n]*(w_1*(P[m+n* num_functions,i]*C[m,i]+S[m+n* num_functions,i]))#S[m+n* num_functions,i]*D[m,i]
                        #for m_s in range(num_functions):
                            #for j in range(num_nodes):
                                #obj=obj+H[k,n]*P[m+n*num_functions,i]*P[m_s+n*num_functions,j]*D[m,i]*a[m,m_s]
            # Fixed virtual/outside demand for deployment n: same time + money
            # cost terms weighted by the pinned outside routing (no H variable).
            if stochastic_decomp:
                vw = virtual_rate * outside_routing[n]
                obj = obj + vw*w_2*T[n]
                for m in range(num_functions):
                    for i in range(num_nodes):
                        obj = obj + vw*(w_1*(P[m+n*num_functions,i]*C[m,i]+S[m+n*num_functions,i]))
        # "Virtual placements" of the decomposition (Eq. (3) of the paper):
        # functions that the upper level placed OUTSIDE this region are pinned
        # to node 0 (the head node), because all communication to other regions
        # is routed through it. They are remembered in already_placed so that
        # they can be excluded from the head node's utilization below (Eq. (4)).
        already_placed=np.zeros((num_workflows,num_functions))
        for m in range(num_functions):
            func_set=deployment_workflow["functions"][f"function_{m}"]["function_set"]
            for n in func_set:
                model.addConstr(P[n*num_functions+m,0]==1,"Already placed workflow functions")
                already_placed[n,m]+=1
        # User <-> workflow communication terms: the first function(s) m_start
        # receive the input data d_in from the requesting leaf (cost D_in plus
        # leaf-to-node latency), the last function(s) m_end send the result back
        # (cost D plus node-to-leaf latency). Cf. the P_{n,m_start,i} D_in,i and
        # P_{n,m_end,i} D_{m,i,k} / L terms in Eq. (1).
        for n in range( num_workflows):
            for k in range( num_leaves):
                for i in range(num_nodes):
                    for m in m_start:
                        obj=obj+lam[k]*H[k,n]*( P[n*num_functions+m,i]*w_1*D_in[i]+w_2*L[leave_nodes[k],i]*P[n*num_functions+m,i])# P[n*num_functions+m,i]*w_1*D_in[i]
                    for m in m_end:
                        obj= obj+lam[k]*H[k,n]*(w_1*D[m,i,leave_nodes[k]]*P[n*num_functions+m, i]+w_2*L[leave_nodes[k], i] * P[n*num_functions+m, i])
            # Fixed virtual/outside demand: enters and exits through the head
            # node 0 (all outside communication is routed through it).
            if stochastic_decomp:
                vw = virtual_rate * outside_routing[n]
                for i in range(num_nodes):
                    for m in m_start:
                        obj = obj + vw*( P[n*num_functions+m,i]*w_1*D_in[i]+w_2*L[0,i]*P[n*num_functions+m,i])
                    for m in m_end:
                        obj = obj + vw*(w_1*D[m,i,0]*P[n*num_functions+m,i]+w_2*L[0,i]*P[n*num_functions+m,i])


        # Function utilization u_{n,m} = sum_k lambda_k H_{k,n} P_{n,m,i} t_{m,i}
        # (expected busy time of one function deployment); its square is added
        # to the objective with weight w_3 to discourage funnelling all requests
        # through the same function instance (reduces blocking/waiting time).
        for n in range(num_workflows):
            for m in range(num_functions):
                virt_u = (virtual_rate*outside_routing[n])*quicksum(P[n*num_functions+m,i]*Time[m]*Speed[m][i] for i in range(num_nodes)) if stochastic_decomp else 0
                model.addConstr(p[n*num_functions+m]==quicksum(quicksum(lam[k]*H[k,n]*P[n*num_functions+m,i]*Time[m]*Speed[m][i] for k in range(num_leaves)) for i in range(num_nodes))+virt_u)
                obj+=w_3*p[n*num_functions+m]**2
        # Node utilization M_i (share of RAM_max,i expected to be occupied,
        # on top of the utilization left by previously placed workflows).
        # For the head node i=0 the RAM of virtually placed functions is
        # subtracted again, since they actually run in another region (Eq. (4)).
        for i in range(num_nodes):
            curr_util=deployment_cloud["providers"][f"provider_{i}"]["utilization"]

            if i ==0:
                virt_M = quicksum((virtual_rate*outside_routing[n])*(quicksum(P[n * num_functions + m, i]*Time[m]*ram[m] for m in range(num_functions))-sum([already_placed[n,m]*Time[m]*ram[m] for m in range(num_functions)])) for n in range(num_workflows)) if stochastic_decomp else 0
                model.addConstr(M[i] == (quicksum(quicksum(lam[k]*H[k,n]*(quicksum(P[n * num_functions + m, i]*Time[m]*ram[m] for m in range(num_functions))-sum([already_placed[n,m]*Time[m]*ram[m] for m in range(num_functions)])) for k in range(num_leaves)) for n in range(num_workflows))+virt_M)/ram_max[i]+curr_util,
                                "Aux cons")
            else:
                virt_M = quicksum((virtual_rate*outside_routing[n])*quicksum(P[n * num_functions + m, i]*Time[m]*ram[m] for m in range(num_functions)) for n in range(num_workflows)) if stochastic_decomp else 0
                model.addConstr(M[i] == (quicksum(quicksum(
                    quicksum(lam[k]*H[k, n] * P[n * num_functions + m, i]*Time[m]*ram[m] for m in range(num_functions)) for k in
                    range(num_leaves)) for n in range(num_workflows))+virt_M)/ram_max[i]+curr_util,
                                "Aux cons")
            obj+=w_3*M[i]**2#quicksum(H[r,n]*M[i,n] for r in range(num_leaves))
        # The three penalty/cost groups above form the utilization cost U of Eq. (1).
        #for k in range(num_nodes):
        #    obj+=w_3*quicksum(quicksum(P[n*num_functions+m,k] for m in range(num_functions)) for n in range(num_workflows))**2
        #for n in range(num_workflows):
        #    obj+=w_3*quicksum(H[k,n] for k in range(num_leaves))**2
        model.setObjective(obj,GRB.MINIMIZE)
        #T_path=np.zeros((num_workflows,num_paths))
        # Placement constraint of Eq. (1): every function of every deployment
        # is placed on exactly one node (sum_i P_{n,m,i} = 1).
        for n in range( num_workflows):
            for m in range( num_functions):
                model.addConstr(quicksum(P[m+n* num_functions,i] for i in range( num_nodes))==1,"only one placed")

        # Selection constraints: in lower-level subproblems the workflow choice
        # was already made by the upper level (Eqs. (5)/(6)), so the inherited
        # selection_list pins H; otherwise (top level / centralized problem)
        # every leaf freely selects exactly one deployment (sum_n H_{k,n} = 1).
        if deployment_workflow["selection_list"]!={}:
            #if deployment_workflow["selection_list"] != [-1]:
            #    for k in range(num_leaves-1):
            #        for n in range(num_workflows):
            #            model.addConstr(H[k+1,n]==deployment_workflow["selection_list"][n])
            #model.addConstr(quicksum(H[0,n] for n in range(num_workflows))==num_workflows)
            for n in  deployment_workflow["selection_list"]:
                for k in range(num_leaves):
                    model.addConstr(H[k,n]==1)
        elif num_workflows > 0:
            # (num_workflows == 0: a fully pruned region has no deployments to
            # select -- adding the simplex constraint would be an unsatisfiable
            # 0 == 1.)
            for k in range(num_leaves):
                model.addConstr(quicksum(H[k, n] for n in range(num_workflows)) == 1, "only one selected")
            # Stage 2: real-user demand conservation. Each leaf freely distributes
            # (simplex above), but the region's OWN-rate aggregate per deployment
            # must reproduce the parent's allocation flow_rhs[n] = lambda_rep *
            # H_parent[rep,n]. Node 0 contributes only its own share (lam[0]); the
            # virtual demand is handled by the fixed outside_routing terms above.
            # With own demand defined on PHYSICAL user leaves only (see
            # head de-conflation above), sum_k lam_k == sum_n flow_rhs holds
            # structurally at every level: edge regions conserve the Dirichlet
            # split exactly, and phantom-head cloud regions have 0 == 0. The
            # guard below is therefore a pure floating-point tolerance (the
            # rates are (a+b)-b reconstructions), not a feasibility escape; if
            # it ever fails, demand bookkeeping is broken and the constraint
            # must not be silently dropped.
            if stochastic_decomp and abs(sum(lam) - sum(flow_rhs)) < 1e-6:
                # Rescale flow_rhs so its sum equals sum(lam) EXACTLY. The
                # residual FP noise (~1e-8, from the parent's H rows summing to
                # 1 only within solver tolerance) is otherwise aggregated by
                # Gurobi's presolve across the simplex + flow equalities into
                # an exact 0 == eps contradiction, and the subproblem is
                # declared infeasible even though the noise is far below
                # FeasibilityTol (run_134, speedup x1.25, opt_0_4).
                if sum(flow_rhs) > 0:
                    rhs_scale = sum(lam) / sum(flow_rhs)
                    flow_rhs = [x * rhs_scale for x in flow_rhs]
                for n in range(num_workflows):
                    model.addConstr(quicksum(lam[k]*H[k, n] for k in range(num_leaves)) == flow_rhs[n], "flow_conservation")
            elif stochastic_decomp:
                print(f"WARNING [{deployment_workflow.get('name','?')}]: flow conservation "
                      f"skipped, own demand {sum(lam):.8f} != parent allocation {sum(flow_rhs):.8f} "
                      f"-- demand bookkeeping is inconsistent.")
        #for n in range( num_workflows):
        #    model.addConstr(quicksum(H[k,n] for k in range( num_leaves))>=1,"place differently")
        #for i in range( num_nodes):
        #    model.addConstr(quicksum(quicksum(P[n*num_functions+m,i]*ram[m] for m in range(num_functions)) for n in range(num_workflows))<= ram_max[i] #*prob_request , "RAM_constraint")
        # Maximum-branch-time constraint of Eq. (1): for every branch c the
        # evaluation time T_n must cover the branch's execution times
        # (Speed*Time) plus the latencies between consecutively placed
        # functions. The product P*P of two placement variables makes the
        # problem bilinear (hence an integer *nonlinear* program).
        for n in range(num_workflows):
            for s in range(num_paths):
                model.addConstr(T[n]>=quicksum(quicksum(quicksum(quicksum(P[n*num_functions+m,i]*P[n*num_functions+m_s,j]*L[i,j]*a[m,m_s]*b[s,m] for m_s in range(num_functions)) for j in range(num_nodes))+P[n*num_functions+m,i]*b[s,m]*Speed[m,i]*Time[m] for m in range(num_functions)) for i in range(num_nodes)),'Time constraint')#=quicksum(quicksum(L[leave_nodes[k],i]*P[n*num_functions+m,i] for k in range(num_leaves)) for m in m_start)+quicksum(quicksum(L[leave_nodes[k],i]*P[n*num_functions+m,i] for k in range(num_leaves)) for m in m_end)#quicksum(quicksum(P[n*num_functions+m,i]*P[n*num_functions+m_s,j]*L[i,j]*a[m,m_s] for m_s in range(num_functions)) for j in range(num_nodes))*b[s,m]+

        # Monetary bilinear helper of Eq. (1): S_{n,m,i} equals the data-transfer
        # cost D_{m,i,j} to every direct successor m_s of m placed on node j.
        for n in range(num_workflows):
            for m in range(num_functions):
                for i in range(num_nodes):
                    model.addConstr(S[n*num_functions+m,i]==quicksum(quicksum(
                        P[n * num_functions + m, i] * P[n * num_functions + m_s, j] *
                        a[m, m_s]*D[m,i,j] for m_s in
                        range(num_functions)) for j in range(num_nodes)),'Adding aux variable')#

        #if num>0:
        #    model.addConstr(quicksum(P[j,num-1] for j in range( num_functions))==0,"node_0_constr")
        model.setParam('OutputFlag', 1)
        model.update()
        # Diagnostic hook: OPTIFAAS_DEBUG_DUMP=<dir> dumps every model
        # (overwriting) right before optimize, so after a native solver crash
        # the last dump is the model that killed the process. No effect when
        # the variable is unset.
        dump_dir = os.environ.get("OPTIFAAS_DEBUG_DUMP")
        if dump_dir:
            model.setParam("LogFile", os.path.join(dump_dir, "gurobi_last.log"))
            model.write(os.path.join(dump_dir, "sub_last.mps"))
            model.write(os.path.join(dump_dir, "sub_last.lp"))
        P_list = []
        # for model in  model_list:
        t1=time.time()
        try:
            # Solve the MINLP and copy the flat Gurobi variable vector back into
            # numpy arrays. Variables appear in the order they were added to the
            # model (P, S, H, T, M, p), so the running index r/s walks through
            # each block row-by-row.
            model.optimize()
            # INF_OR_UNBD (4) is an ambiguous presolve verdict; re-solving with
            # DualReductions=0 is Gurobi's documented way to force a definite
            # status. It is also load-bearing: Gurobi 13.0.2 misdeclares some
            # feasible nonconvex subproblems as 4 (run_134 request_rate x0.75,
            # workflow 9), and the retry then finds the true optimum. A truly
            # infeasible model still ends at status 3 and raises below.
            if model.status == GRB.INF_OR_UNBD:
                print("Status INF_OR_UNBD -- re-solving with DualReductions=0 "
                      "for a definite status.")
                model.setParam('DualReductions', 0)
                model.optimize()
            # An infeasible/unbounded subproblem means the decomposition's
            # demand/pruning bookkeeping is inconsistent -- never write a
            # degenerate solution for it (that silently corrupts the run).
            if model.status != GRB.OPTIMAL:
                raise RuntimeError(
                    f"Subproblem {deployment_workflow.get('name', '?')}: Gurobi "
                    f"finished with status {model.status} (2 = OPTIMAL). The "
                    f"model is infeasible/unbounded or the solve failed; the "
                    f"decomposition bookkeeping for this region is inconsistent.")

            t2=time.time()
            t=t2-t1
            P = np.zeros((num_functions * num_workflows, num_nodes))
            S = np.zeros((num_functions * num_workflows, num_nodes))
            H = np.zeros((num_leaves, num_workflows))
            T = np.zeros((num_workflows, 1))
            M=  np.zeros((num_nodes,1))
            p= np.zeros((num_workflows*num_functions,1))
            r = 0
            s = 0
            for index,v in enumerate(model.getVars()):
                if index<num_functions*num_workflows*num_nodes:
                    P[s, r] = round(v.x)
                    r = r + 1
                    if r == num_nodes:
                        s = s + 1
                        r = 0
                    if s == num_functions*num_workflows:
                        s=0

                elif index<2*num_functions*num_workflows*num_nodes:
                    S[s, r] = v.x
                    r = r + 1
                    if r == num_nodes:
                        s = s + 1
                        r = 0
                    if s == num_functions*num_workflows:
                        s=0
                elif index<2*num_functions*num_workflows*num_nodes+num_leaves*num_workflows:
                    # Keep the continuous probability when relaxed; otherwise the
                    # binary selection is recovered by rounding the solver value.
                    H[s, r] = v.x if relax_selection else round(v.x)
                    r = r + 1
                    if r == num_workflows:
                        s = s + 1
                        r = 0
                    if s == num_leaves:
                        s=0
                elif index<2*num_functions*num_workflows*num_nodes+num_leaves*num_workflows+num_workflows:
                    T[s] = v.x
                    s = s + 1
                    if s==num_workflows:
                        s=0
                elif index<2*num_functions*num_workflows*num_nodes+num_leaves*num_workflows+num_workflows+num_nodes:
                    M[s] =v.x
                    s=s+1
                    if s==num_nodes:
                        s=0
                else:
                    p[s]=v.x
                    s=s+1
            obj=model.getObjective()
        except:
            # Never swallow solver errors: a silently returned zero solution
            # corrupts the whole downstream run (see run_132 diagnosis).
            raise
    P_list.append(P)
    #print(obj)
    print(P)

    print(H)
    print(M)
    print(p)
    return model,P,H,L,T,C,D,D_in,Time,a,b,Speed,S,t,M



