# Generate the parameter files for the scalability study (Sec. IV-C of the
# paper): starting from settings/parameters_1.json, three runs with an
# increasing number of hierarchical levels (2, 3, 4) are written to
# settings/parameters_123.json ... parameters_125.json. Since each level
# multiplies the node count, this scales the architecture to 40/400/4000
# physical nodes (Table 4). run_time_case_study_heuristic.bat executes runs
# 123-125 (comparison against the cloud-only heuristic) and
# run_time_case_study.bat runs 126-129 (comparison against the centralized
# optimization; create those parameter files analogously).
import json
import numpy as np
if __name__ == "__main__":
    parameter_config_path="settings/parameters_1.json"
    num_weights=4 # 4 when time case study, 3 when heuristic study
    with open(parameter_config_path, "r") as f:
        parameter_general_config = json.load(f)
    # Alternative scaling knob (disabled): grow the nodes per region instead
    # of the number of levels.
    nodes=[2+k for k in range(num_weights)]#np.linspace(3,100,num_weights)
    for k in range(num_weights):
        parameter_general_config["config_random_cloud"]["nodes_min"]=nodes[k] # Use these for runs 126-129
        parameter_general_config["config_random_cloud"]["nodes_max"]=nodes[k]+1
        #parameter_general_config["num_levels"]=k+2 # Use this line for experiments 123-125
        parameter_config_path=f"settings/parameters_{k+126}.json"
        with open(parameter_config_path, "w") as f:
            json.dump(parameter_general_config, f, indent=4)
