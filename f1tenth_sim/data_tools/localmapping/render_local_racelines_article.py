import numpy as np 
import matplotlib.pyplot as plt
import os
from f1tenth_sim.data_tools.plotting_utils import *
from matplotlib.collections import LineCollection
import trajectory_planning_helpers as tph
from f1tenth_sim.utils.MapData import MapData

def ensure_path_exists(path):
    if not os.path.exists(path): 
        os.mkdir(path)


def reoreintate_pts(pts, position, theta):
    # pts = pts - position
    rotation_mtx = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    pts = np.matmul(pts, rotation_mtx.T) + position

    return pts

def render_local_maps(planner_name, test_id, map_name="aut"):
    root = f"Logs/{planner_name}/"
    raceline_data_path = root + f"RawData_{test_id}/RacingLineData_{test_id}/"
    localmap_data_path = root + f"RawData_{test_id}/LocalMapData_{test_id}/"
    logs = np.load(root + f"RawData_{test_id}/SimLog_{map_name}_0.npy")
    ensure_path_exists(root + f"Images_{test_id}/")
    data_img_path = f"Data/LocalMapRacing/LocalRacelines/" 
    raceline_img_path = root + f"Images_{test_id}/" + f"LocalRaceline_{test_id}/"
    ensure_path_exists(raceline_img_path)
    # in the future, I could load the scans from here and not hae to save them seperately....

    map_data = MapData(map_name)
    
    # for i in range(0, 100):
    # for i in range(len(logs)-50, len(logs)):
    for i in [345, 20, 457, 30]:
    # for i in range(len(logs)):
        local_track = np.load(localmap_data_path + f"local_map_{i}.npy")
        raceline = np.load(raceline_data_path + f"LocalRaceline_{i+1}.npy")

        plt.figure(1)
        plt.clf()
        map_data.plot_map_img_light()

        position = logs[i+1, :2]
        orientation = logs[i+1, 4]


        pts = reoreintate_pts(local_track[:, :2], position, orientation)
        xs, ys = map_data.pts2rc(pts)
        plt.plot(xs, ys, '--', linewidth=2, color='black')

        vs = raceline[:, 2]
        pts = reoreintate_pts(raceline[:, :2], position, orientation)
        xs, ys = map_data.pts2rc(pts)
        points = np.concatenate([xs[:, None], ys[:, None]], axis=1)
        points = points.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        norm = plt.Normalize(2, 8)
        lc = LineCollection(segments, cmap='jet', norm=norm)
        lc.set_array(vs)
        lc.set_linewidth(5)
        line = plt.gca().add_collection(lc)

        x1s, y1s = map_data.pts2rc(position[None, :])
        l = 12
        plt.arrow(x1s[0], y1s[0], np.cos(orientation)*l, np.sin(orientation)*l, color="#8854d0", zorder=10, width=3, head_width=6, head_length=5)


        b = 20
        x_min = np.min(xs) - b
        x_max = np.max(xs) + b
        y_min = np.min(ys) - b
        y_max = np.max(ys) + b
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)

        plt.tight_layout()
        plt.gca().set_aspect('equal', adjustable='box')
        plt.axis('off')

        plt.savefig(raceline_img_path + f"Raceline_{i}.svg")
        plt.savefig(data_img_path + f"Raceline_{i}.pdf", bbox_inches="tight", pad_inches=0.05)
        plt.savefig(data_img_path + f"Raceline_{i}.svg", bbox_inches="tight", pad_inches=0.05)
        cbar = plt.colorbar(line, shrink=0.5)
        plt.gca().set_aspect('equal', adjustable='box')
        cbar.ax.tick_params(labelsize=20)
        plt.savefig(raceline_img_path + f"Raceline_{i}_c.svg")
        plt.savefig(data_img_path + f"Raceline_{i}_c.pdf", bbox_inches="tight", pad_inches=0.05)
        plt.savefig(data_img_path + f"Raceline_{i}_c.svg", bbox_inches="tight", pad_inches=0.05)


if __name__ == '__main__':
    # render_local_maps("LocalMapPlanner", "r1")
    render_local_maps("LocalMapPP", "test", "aut")
    # render_local_maps("LocalMapPP", "mu60", "aut")




