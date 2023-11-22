import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
map_name = "esp"
from f1tenth_sim.data_tools.plotting_utils import *

def make_computation_table():
    vehicle_id = {
                  "FollowTheGap_Std": "Follow \nthe gap", 
                  "EndToEnd_TestSAC": "End-to-end \nSAC", 
                  "EndToEnd_TestTD3": "End-to-end \nTD3", 
                    "FullStackPP_mu60": "Global \ntwo-stage", 
                  "FullStackMPCC_mu60": "Global \nMPCC", 
                  "LocalMapPP_mu60": "Local \ntwo-stage", 
                  "LocalMPCC_mu60": "Local \nMPCC"}

    perception_id = {
                  "FollowTheGap_Std": None, 
                  "EndToEnd_TestSAC": None, 
                  "EndToEnd_TestTD3": None, 
                "FullStackPP_mu60": "localise", 
                  "FullStackMPCC_mu60": "localise", 
                  "LocalMapPP_mu60": "generate_line_local_map", 
                  "LocalMPCC_mu60": "generate_line_local_map"}

    planning_times = {}
    perception_times = {}

    for k in vehicle_id.keys():
        planner_name = k.split('_')[0]
        test_id = k.split('_')[1]
        path = f"Logs/{planner_name}/RawData_{test_id}/"

        df = pd.read_csv(path + f"Profile_{map_name}_{test_id}.csv")
        df["cumtime"] = df["cumtime"].astype(float, errors="ignore")  # Convert cumtime column to float
        df["ncalls"] = pd.to_numeric(df["ncalls"], errors='coerce')  # Convert cumtime column to float

        planning = df.loc[df["func"] == "plan"]
        if k[0] == "L":
            ct = df.loc[df["func"] == "generate_line_local_map"].cumtime.values[0]
            planning_time = (planning.cumtime.values[0] -  ct) / planning.ncalls.values[0]
        else:
            planning_time = planning.cumtime.values[0] / planning.ncalls.values[0]
            # planning_time = planning["cumtime"] / planning["ncalls"]
        planning_times[vehicle_id[k]] = planning_time

        if perception_id[k] is not None:
            perception = df.loc[df["func"] == perception_id[k]]
            perception_time = perception["cumtime"].values[0] / perception["ncalls"].values[0]
            perception_times[vehicle_id[k]] = perception_time
        else:
            perception_times[vehicle_id[k]] = 0

    planning_times = pd.DataFrame.from_dict(planning_times, orient="index", columns=["PlanningTime"])
    perception_times = pd.DataFrame.from_dict(perception_times, orient="index", columns=["PerceptionTime"])

    # Merge the data frames into one
    computation_table = pd.concat([perception_times, planning_times], axis=1)
    print(computation_table)

    computation_table.plot(kind='bar', use_index=True, stacked=True, logy=True, color=[sweedish_green, nartjie], figsize=(5, 2.), rot=0)
    # plt.plot(plt.xlim(), [0.04, 0.04], 'k--', linewidth=1, label="25 Hz")
    # plt.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.3), fontsize=8)
    plt.legend(["Perception", "Planning"], ncol=2, loc="center", bbox_to_anchor=(0.28, 0.9), fontsize=8)
    # plt.legend(ncol=1, loc="center", bbox_to_anchor=(0.4, 0.8), fontsize=8)
    plt.plot(plt.xlim(), [0.04, 0.04], 'k--', linewidth=1)
    plt.gca().set_xticklabels(plt.gca().get_xticklabels(), fontsize=7)
    # plt.gca().set_yticklabels(plt.gca().get_yticklabels(), fontsize=8)
    plt.yticks(fontsize=9)
    plt.ylabel("Computation time (s)", fontsize=8)

    plt.tight_layout()
    plt.grid()

    plt.savefig(f"Data/LocalMapRacing/computation_times.svg", bbox_inches='tight', pad_inches=0)
    plt.savefig(f"Data/LocalMapRacing/computation_times.pdf", bbox_inches='tight', pad_inches=0)

make_computation_table()



