# Generate the parameter files for the weight-sensitivity study (Sec. IV-D of
# the paper): starting from settings/parameters_1.json, an 11x11 grid of
# weight combinations is written to settings/parameters_2.json ...
# parameters_122.json. For each of 11 utilization weights w_3 in [0, 10],
# the monetary weight w_1 sweeps [0, 1] with w_2 = 1 - w_1 (the Pareto-front
# weighted-sum scan of Fig. 8 / Table 5). run_case_study.bat then runs
# placement + simulation for all of these runs.
import json
import numpy as np
if __name__ == "__main__":
    parameter_config_path="settings/parameters_1.json"
    weights_config_path="settings/weights_2.json"
    # First build a simplex grid of (w1, w2, w3) triples summing to one; it is
    # only saved to weights_2.json for reference (visualization.py loads it).
    num_weights=10
    weights=[]
    for i in range(num_weights + 1):
        for j in range(num_weights + 1 - i):
            w1 = i / num_weights
            w2 = j / num_weights
            w3 = (1 - w1 - w2)
            weights.append([w1, w2, w3])

    # Actual experiment grid: w_3 in {0,1,...,10} x w_1 in {0,0.1,...,1},
    # w_2 = 1 - w_1. The small +0.0001 offsets avoid exactly-zero weights.
    num_weights_util=11
    num_weights=11
    with open(parameter_config_path, "r") as f:
        parameter_general_config = json.load(f)
    w_3=np.linspace(0,10,num_weights_util)+0.0001
    w_1 = np.linspace(0, 1, num_weights)+0.0001
    w_2 = (1.0001-w_1)+0.0001
    # Write one parameter file per (w_3, w_1) pair; run numbers start at 2
    # (parameters_1.json is the unmodified template).
    for k in range(len(w_3)):
        for m in range(len(w_2)):
            parameter_general_config["w_1"]=w_1[m]
            parameter_general_config["w_2"]=w_2[m]
            parameter_general_config["w_3"] = w_3[k]
            parameter_config_path=f"settings/parameters_{k*num_weights_util+m+2}.json"
            with open(parameter_config_path, "w") as f:
                json.dump(parameter_general_config, f, indent=4)
    with open(weights_config_path, "w") as f:
        json.dump(weights, f, indent=4)
