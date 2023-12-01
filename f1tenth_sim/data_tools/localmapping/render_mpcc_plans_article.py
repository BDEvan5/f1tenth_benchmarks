import numpy as np 
import matplotlib.pyplot as plt
import os
from f1tenth_sim.data_tools.plotting_utils import *
from matplotlib.collections import LineCollection
import trajectory_planning_helpers as tph
from f1tenth_sim.utils.track_utils import *

def ensure_path_exists(path):
    if not os.path.exists(path): 
        os.mkdir(path)

def render_local_maps(planner_name, test_id, map_name="aut"):
    root = f"Logs/{planner_name}/"
    mpcc_data_path = root + f"RawData_{test_id}/mpcc_data/"
    logs = np.load(root + f"RawData_{test_id}/SimLog_{map_name}_0.npy")
    ensure_path_exists(root + f"Images_{test_id}/")
    mpcc_img_path = root + f"Images_{test_id}/mpcc_{test_id}/"
    ensure_path_exists(mpcc_img_path)
    
    track = CentreLine(map_name)

    for i in range(475, 485):
    # for i in range(35, 45):
    # for i in range(0, 100):
    # for i in range(len(logs)-100, len(logs)-50):
    # for i in range(len(logs)-50, len(logs)):
    # for i in range(len(logs)):
        states = np.load(mpcc_data_path + f"States_{i}.npy")
        controls = np.load(mpcc_data_path + f"Controls_{i}.npy")

        # fig = plt.figure(1)
        fig = plt.figure(figsize=(5, 2.))
        # fig = plt.figure(figsize=(6, 3))
        plt.clf()
        a1 = plt.subplot2grid((2, 2), (0, 0), rowspan=2)
        a2 = plt.subplot2grid((2, 2), (0, 1))
        a3 = plt.subplot2grid((2, 2), (1, 1))

        xs = np.interp(states[:, 3], track.s_path, track.path[:, 0])
        ys = np.interp(states[:, 3], track.s_path, track.path[:, 1])
        # a1.plot(xs, ys, 'o', color='orange', markersize=5)

        for z in range(len(states)):
            x_line = [states[z, 0], xs[z]]
            y_line = [states[z, 1], ys[z]]
            a1.plot(x_line, y_line, '--', color='gray', linewidth=1)

        a1.plot(track.path[:, 0], track.path[:, 1], '--', linewidth=2, color='black')
        l1 = track.path[:, :2] + track.nvecs * track.widths[:, 0][:, None]
        l2 = track.path[:, :2] - track.nvecs * track.widths[:, 1][:, None]
        a1.plot(l1[:, 0], l1[:, 1], color='green')
        a1.plot(l2[:, 0], l2[:, 1], color='green')

        l = 0.6
        a1.arrow(logs[i, 0], logs[i, 1], np.cos(logs[i, 4])*l, np.sin(logs[i, 4])*l, color="#8854d0", zorder=2, width=0.3, head_width=0.45, head_length=0.45)
        # a1.plot(logs[i, 0], logs[i, 1], '*', markersize=10, color='red')
        a1.axis('off')

        b = 1.25
        # b = 0.8
        a1.set_xlim(np.min(xs)-b, np.max(xs)+b)
        a1.set_ylim(np.min(ys)-b, np.max(ys)+b)


        vs = controls[:, 1]
        points = states[:, :2].reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        norm = plt.Normalize(2, 8)
        lc = LineCollection(segments, cmap='jet', norm=norm)
        lc.set_array(vs)
        lc.set_linewidth(4)
        line = a1.add_collection(lc)
        plt.sca(a1)
        plt.colorbar(line, shrink=0.6)
        a1.set_aspect('equal', adjustable='box')


        a2.plot(controls[:, 1], linewidth=2, color=sunset_orange)
        # a2.plot(controls[:, 2])
        a2.set_ylabel("Speed")
        a2.grid(True)
        a2.set_ylim(1, 9.5)
        a2.xaxis.set_major_locator(plt.MaxNLocator(5))

        a3.plot(controls[:, 0], linewidth=2, color=sunset_orange)
        a3.set_ylabel("Steering")
        a3.grid(True)
        a3.set_ylim(-0.4, 0.4)
        a3.xaxis.set_major_locator(plt.MaxNLocator(5))


        plt.tight_layout()

        plt.rcParams['pdf.use14corefonts'] = True
        plt.savefig(mpcc_img_path + f"Raceline_{i}.svg", bbox_inches="tight", pad_inches=0.05)
        plt.savefig(f"Data/LocalMapRacing/mpcc_imgs/" + f"Raceline_{i}.svg", bbox_inches="tight", pad_inches=0.05)
        plt.savefig(f"Data/LocalMapRacing/mpcc_imgs/" + f"Raceline_{i}.pdf", bbox_inches="tight", pad_inches=0.05)

        plt.close(fig)


if __name__ == '__main__':
    # render_local_maps("LocalMapPlanner", "r1")
    # render_local_maps("LocalMPCC2", "r1", "aut")
    render_local_maps("FullStackMPCC", "t1", "aut")




