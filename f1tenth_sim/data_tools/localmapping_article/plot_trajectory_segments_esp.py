import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 
from matplotlib.collections import LineCollection
from f1tenth_sim.data_tools.plotting_utils import *
from f1tenth_sim.utils.MapData import MapData

def make_trajectory_imgs(planner_name, test_id, map_name, test_lap, name="", cbar=False):
    root = f"Logs/{planner_name}/"
    save_path = f"Data/LocalMapRacing/Trajectories_{map_name.upper()}/"
    ensure_path_exists(save_path)

    map_data = MapData(map_name)
    logs = np.load(root + f"RawData_{test_id}/SimLog_{map_name}_{test_lap}.npy")

    positions = logs[:, :2]
    speeds = logs[:, 3]

    plt.figure()
    plt.clf()
    map_data.plot_map_img()

    xs, ys = map_data.pts2rc(positions)
    points = np.concatenate([xs[:, None], ys[:, None]], axis=1)
    points = points.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    norm = plt.Normalize(0, 8)
    lc = LineCollection(segments, cmap='jet', norm=norm)
    lc.set_array(speeds)
    lc.set_linewidth(5)
    line = plt.gca().add_collection(lc)
    plt.gca().set_aspect('equal', adjustable='box')

    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
        
    esp_left_limits()
    plt.text(150, 410, name, fontsize=20)

    plt.tight_layout()

    if cbar:
        cbar = plt.colorbar(line, fraction=0.046, pad=0.04, shrink=0.9)
        cbar.ax.tick_params(labelsize=20)
        name = f"{planner_name}_{test_id}_ESP_{test_lap}_left"

        plt.rcParams['pdf.use14corefonts'] = True
        plt.savefig(save_path + name + ".svg", bbox_inches="tight", pad_inches=0)
        plt.savefig(save_path + name + ".pdf", bbox_inches="tight", pad_inches=0)
    else:
        name = f"{planner_name}_{test_id}_ESP_{test_lap}_left_noC"

        plt.rcParams['pdf.use14corefonts'] = True
        plt.savefig(save_path + name + ".svg", bbox_inches="tight", pad_inches=0)
        plt.savefig(save_path + name + ".pdf", bbox_inches="tight", pad_inches=0)


def esp_left_limits():

    # plt.xlim(20, 620)
    plt.xlim(20, 420)
    plt.ylim(50, 520)







map_name = "esp"
lap_n = 0

make_trajectory_imgs("LocalMPCC", "mu60", map_name, lap_n, "Local MPCC", True)
make_trajectory_imgs("FollowTheGap", "Std", map_name, lap_n, "Follow the gap", False)
make_trajectory_imgs("EndToEnd", "TestTD3", map_name, lap_n, "End-to-end TD3", False)


# make_trajectory_imgs("LocalMapPP", "mu60", map_name, lap_n)
# make_trajectory_imgs("FullStackMPCC", "mu60", map_name, lap_n)




