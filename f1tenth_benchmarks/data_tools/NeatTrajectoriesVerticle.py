import os 
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import csv

from f1tenth_benchmarks.data_tools.plotting_utils import *
from f1tenth_benchmarks.utils.MapData import MapData
# from F1TenthRacingDRL.DataTools.MapData import MapData
# from F1TenthRacingDRL.DataTools.plotting_utils import *

root_path = "/home/benjy/sim_ws/src/F1TenthRacingROS/Data/"


def make_agent_trajectory(agent_name, label_name, folder, run_n, cbar=True):
    map_data = MapData("CornerHall") 

    folder = root_path + folder + f"{agent_name}/Run_{run_n}"
    with open(folder + f"/Run_{run_n}_states_{run_n}.csv") as file:
        state_reader = csv.reader(file, delimiter=',')
        state_list = []
        for row in state_reader:
            state_list.append(row)
    states = np.array(state_list[1:]).astype(float)
        
    plt.figure(1)
    plt.clf()
    map_data.plot_map_img()
    # map_data.plot_map_img_transpose()

    xs, ys = map_data.xy2rc(states[:, 0], states[:, 1])
    points = np.array([xs, ys]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    norm = plt.Normalize(1, 8)
    # norm = plt.Normalize(1, 5)
    lc = LineCollection(segments, cmap='jet', norm=norm)
    lc.set_array(states[:, 3])
    lc.set_linewidth(3)
    line = plt.gca().add_collection(lc)
    if cbar:
        cb = plt.colorbar(line, fraction=0.08, pad=0.04, shrink=0.5, label="Speed (m/s)")
        cb.ax.tick_params(labelsize=14)
        for t in cb.ax.get_yticklabels():
            t.set_fontsize(14)

    plt.text(50, 80, label, fontsize=16, color='black', ha="center", va="center")
    # plt.text(8, 60, label, fontsize=16, color='black')
    plt.ylim(35, 380)
    plt.xlim(5, 135)
    plt.margins(x=0, y=0)
    plt.xticks([])
    plt.yticks([])
    plt.gca().set_axis_off()
    label_name = label_name.replace("-", "_")
    label_name = label_name.replace("m/s", "")
    label_name = label_name.replace(" ", "")
    label_name = label_name.replace("\n", "")

    name = f"Data/Sim2RealImgs/TrajectoryV_{label_name}"
    std_img_saving(name)



agent_name = "FollowTheGap"
label = "Follow \nthe \ngap"
folder = "SimNov23_1/"
run_n = 0
make_agent_trajectory(agent_name, label, folder, run_n, False)

agent_name = "LocalPlanning"
label = "Local \nplanning"
folder = "SimNov23_1/"
run_n = 0
make_agent_trajectory(agent_name, label, folder, run_n, True)


agent_name = "PurePursuit"
label = "Global \nplanning"
folder = "SimNov23_1/"
run_n = 0
make_agent_trajectory(agent_name, label, folder, run_n, False)




