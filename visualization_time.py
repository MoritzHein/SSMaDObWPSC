# Visualization of the decomposed-vs-CENTRALIZED scaling study (Sec. IV-C,
# Table 3 of the paper).
#
# Usage: visualization_time.py --num_run=<N>  (figures saved to results/run_<N>/)
#
# Reads runs 126..128 (increasing node count, run via run_time_case_study.bat)
# and plots (a) the computation time of the decomposed approach - summed over
# its subproblems - against the centralized solve time and (b) the relative
# cost gap g between both solutions.
import scienceplots
import matplotlib.pyplot as plt
import sys
import os
import json
import argparse
import numpy as np
plt.style.use('science')

sys.path.append('./../plotTEX')
import config_mpl

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
    )
    parser.add_argument("--num_run", required=True, type=int)
    args = parser.parse_args()
    num_run = args.num_run

    # setup export path
    export_path = os.path.join('results',f'run_{num_run}')
    time=[]
    time_comparison=[]
    costs=[]
    costs_comparison=[]
    num_weights=3
    num_workflows=10
    heuristic=False
    # Average per-workflow solve times and simulated costs over the scaling
    # runs: time_list[workflow][level][region] from placement.py is summed
    # over all subproblems, comparison_time is the centralized solve time.
    for k in range(num_weights):
        import_path_time = os.path.join('results', f'run_{k+126}','time.json')
        with open(import_path_time, "r") as f:  # requirements_workflow_0.json
            results_time = json.load(f)
        import_path_time = os.path.join('results', f'run_{k + 126}', 'results.json')
        with open(import_path_time, "r") as f:  # requirements_workflow_0.json
            results_cost = json.load(f)
        time.append(sum([sum([sum(results_time['time_list'][r][k]) for k in range(len(results_time['time_list'][0]))]) for r in range(num_workflows)])/num_workflows)#results_time["overall_time"][0])
        time_comparison.append(sum([results_time["comparison_time"][r] for r in range(len(results_time['comparison_time']))])/num_workflows)
        costs.append(results_cost["overall_cost"]/num_workflows)
        costs_comparison.append(results_cost["overall_cost_comparison"]/num_workflows)
    if not os.path.exists(export_path):
        raise Exception(f'The folder {export_path} where the figures will be saved does not exist.')
    else:
        print(f'Saving to folder {export_path}.')
    format_type = 'pgf'

    # Computation time of both approaches over the number of nodes per region.
    fig,ax=plt.subplots(1,1,figsize=(config_mpl.textwidth, config_mpl.textwidth / config_mpl.golden_ratio),sharex="col")
    nodes=[k+2 for k in range(num_weights)]#
    #nodes=[30 * 10 ** k for k in range(num_weights)]
    ax.plot(nodes,time,label="decomposed")
    if heuristic:
        ax.plot(nodes, time_comparison, label="heuristic")
    else:
        ax.plot(nodes,time_comparison, label="centralized")
    ax.legend()
    ax.set_xlabel(r'Number of nodes')
    ax.set_ylabel(r'Computation time $[s]$')
    ax.set_xticks([2,3,4])

    fig.savefig(os.path.join(export_path, f'time_results_{num_run}.' + format_type), format=format_type)

    format_type = 'svg'
    fig.savefig(os.path.join(export_path, f'time_results_{num_run}.' + format_type), format=format_type)
    fig, ax = plt.subplots(1, 1, figsize=(config_mpl.textwidth, config_mpl.textwidth / config_mpl.golden_ratio),
                           sharex="col")
    # Relative cost gap g [%] of the decomposed against the centralized cost.
    #nodes = [50*10**k for k in range(num_weights)]
    ax.plot(nodes, 100*(np.array(costs)-np.array(costs_comparison))/np.array(costs_comparison), label="decomposed")

    ax.set_xlabel(r'Number of nodes')
    ax.set_ylabel(r'Cost Gap g [\%]')
    ax.set_xticks([2,3,4])
    fig.savefig(os.path.join(export_path, f'cost_results_{num_run}.' + format_type), format=format_type)

    format_type = 'svg'
    fig.savefig(os.path.join(export_path, f'cost_results_{num_run}.' + format_type), format=format_type)