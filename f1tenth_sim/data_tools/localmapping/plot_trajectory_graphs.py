import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 
from matplotlib.collections import LineCollection
from f1tenth_sim.data_tools.plotting_utils import *
from f1tenth_sim.utils.MapData import MapData


def plot_speed_profiles(planner1, planner2, test_id, map_name, test_lap, name=""):
    save_path = f"Data/LocalMapRacing/"
    
    root = f"Logs/{planner1}/"
    logs1 = np.load(root + f"RawData_{test_id}/SimLog_{map_name}_{test_lap}.npy")
    root = f"Logs/{planner2}/"
    logs2 = np.load(root + f"RawData_{test_id}/SimLog_{map_name}_{test_lap}.npy")


    plt.figure()
    plt.clf()

    ax1 = plt.subplot(2, 1, 1)
    ax2 = plt.subplot(2, 1, 2)

    ax1.plot(logs1[:, 3], color='blue', linewidth=3)
    ax1.plot(logs2[:, 3], color='orange', linewidth=3)




    name = f"SpeedComparison_{test_id}_ESP_{test_lap}"

    plt.savefig(save_path + name + ".svg", bbox_inches="tight", pad_inches=0)
    plt.savefig(save_path + name + ".pdf", bbox_inches="tight", pad_inches=0)







map_name = "esp"
lap_n = 0

plot_speed_profiles("FullStackPP", "LocalMapPP", "mu60", map_name, lap_n)



