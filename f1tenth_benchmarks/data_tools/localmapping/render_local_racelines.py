import numpy as np 
import matplotlib.pyplot as plt
import os
from f1tenth_benchmarks.data_tools.plotting_utils import *
from matplotlib.collections import LineCollection
import trajectory_planning_helpers as tph

def ensure_path_exists(path):
    if not os.path.exists(path): 
        os.mkdir(path)

def render_local_maps(planner_name, test_id, map_name="aut"):
    root = f"Logs/{planner_name}/"
    raceline_data_path = root + f"RacingLineData_{test_id}/"
    localmap_data_path = root + f"RawData_{test_id}/LocalMapData_{test_id}/"
    Logs = np.load(root + f"RawData_{test_id}/SimLog_{map_name}_0.npy")
    raceline_img_path = root + f"LocalRaceline_{test_id}/"
    ensure_path_exists(raceline_img_path)
    # in the future, I could load the scans from here and not hae to save them seperately....

    # for i in range(0, 100):
    for i in range(len(Logs)-50, len(Logs)):
    # for i in range(len(Logs)):
        local_track = np.load(localmap_data_path + f"local_map_{i}.npy")
        raceline = np.load(raceline_data_path + f"LocalRaceline_{i}.npy")

        plt.figure(1)
        plt.clf()

        el_lengths = np.linalg.norm(np.diff(local_track[:, :2], axis=0), axis=1)
        psi, kappa = tph.calc_head_curv_num.calc_head_curv_num(local_track, el_lengths, False)
        nvecs = tph.calc_normal_vectors_ahead.calc_normal_vectors_ahead(psi)

        plt.plot(local_track[:, 0], local_track[:, 1], '--', linewidth=2, color='black')

        vs = raceline[:, 2]
        points = raceline[:, :2].reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        norm = plt.Normalize(2, 8)
        lc = LineCollection(segments, cmap='jet', norm=norm)
        lc.set_array(vs)
        lc.set_linewidth(3)
        line = plt.gca().add_collection(lc)
        plt.colorbar(line)
        plt.gca().set_aspect('equal', adjustable='box')

        l1 = local_track[:, :2] + nvecs * local_track[:, 2][:, None]
        l2 = local_track[:, :2] - nvecs * local_track[:, 3][:, None]
        plt.plot(l1[:, 0], l1[:, 1], color='green')
        plt.plot(l2[:, 0], l2[:, 1], color='green')

        plt.plot(0, 0, '*', markersize=10, color='red')
        plt.title(f"Action: ({Logs[i, 7]:.3f}, {Logs[i, 8]:.1f})")

        plt.tight_layout()
        plt.gca().set_aspect('equal', adjustable='box')
        plt.axis('off')

        plt.savefig(raceline_img_path + f"Raceline_{i}.svg")


if __name__ == '__main__':
    # render_local_maps("LocalMapPlanner", "r1")
    render_local_maps("LocalMapPP", "r1", "mco")




