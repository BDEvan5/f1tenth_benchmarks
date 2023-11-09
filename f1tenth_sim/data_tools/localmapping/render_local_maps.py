import numpy as np 
import matplotlib.pyplot as plt
import os
from f1tenth_sim.data_tools.plotting_utils import *

def ensure_path_exists(path):
    if not os.path.exists(path): 
        os.mkdir(path)

def render_local_maps(planner_name, test_id, map_name="aut"):
    root = f"Logs/{planner_name}/"
    scan_data_path = root + f"ScanData_{test_id}/"
    localmap_data_path = root + f"LocalMapData_{test_id}/"
    logs = np.load(root + f"RawData_{test_id}/SimLog_{map_name}_0.npy")
    save_path = root + f"LocalMapGeneration_{test_id}/"
    ensure_path_exists(save_path)
    # in the future, I could load the scans from here and not hae to save them seperately....

    angles = np.linspace(-2.35619449615, 2.35619449615, 1080)
    coses = np.cos(angles)
    sines = np.sin(angles)

    for i in range(len(logs)):
        scan = np.load(scan_data_path + f"Scan_{i}.npy")
        scan_xs, scan_ys = scan * np.array([coses, sines])

        local_track = np.load(localmap_data_path + f"local_map_{i}.npy")
        line_1 = np.load(localmap_data_path + f"line1_{i}.npy")
        line_2 = np.load(localmap_data_path + f"line2_{i}.npy")
        boundaries = np.load(localmap_data_path + f"boundaries_{i}.npy")
        boundary_extension= np.load(localmap_data_path + f"boundExtension_{i}.npy") 

        # Plot local map
        plt.figure(1)
        plt.clf()

        plt.plot(scan_xs, scan_ys, '.', color='#45aaf2', alpha=0.5)
        plt.plot(0, 0, '*', markersize=12, color='red')

        plt.plot(line_1[:, 0], line_1[:, 1], '-', color=minty_green, linewidth=3)
        plt.plot(line_2[:, 0], line_2[:, 1], '-', color=minty_green, linewidth=3)

        plt.plot(local_track[:, 0], local_track[:, 1], '-X', color='orange', markersize=10)

        for z in range(len(boundaries)):
            xs = [boundaries[z, 0], boundaries[z, 2]]
            ys = [boundaries[z, 1], boundaries[z, 3]]
            plt.plot(xs, ys, '-o', color='black', markersize=8)
        
        if len(boundary_extension) > 0:
            for z in range(boundary_extension.shape[0]):
                xs = [boundary_extension[z, 0], boundary_extension[z, 2]]
                ys = [boundary_extension[z, 1], boundary_extension[z, 3]]
                plt.plot(xs, ys, '-o', color='pink', markersize=8)

        plt.axis('equal')
        plt.tight_layout()
        plt.axis('off')
        name = save_path + f"LocalMapGeneration_{i}"
        plt.savefig(name + ".svg", bbox_inches="tight")
        # plt.show()
        # break

if __name__ == '__main__':
    render_local_maps("LocalMapPlanner", "devel")



