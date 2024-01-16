import numpy as np 
import matplotlib.pyplot as plt
import os
from f1tenth_sim.data_tools.plotting_utils import *
from matplotlib.collections import LineCollection
import trajectory_planning_helpers as tph
from f1tenth_sim.utils.track_utils import CentreLine

def ensure_path_exists(path):
    if not os.path.exists(path): 
        os.mkdir(path)

def render_local_maps(planner_name, test_id, map_name="aut"):
    root = f"Logs/{planner_name}/"
    mpcc_data_path = root + f"RawData_{test_id}/MPCCData_{test_id}/"
    localmap_data_path = root + f"RawData_{test_id}/LocalMapData_{test_id}/"
    Logs = np.load(root + f"RawData_{test_id}/SimLog_{map_name}_0.npy")
    mpcc_img_path = root + f"Images_{test_id}/LocalMPCC_{test_id}/"
    ensure_path_exists(mpcc_img_path)
    # in the future, I could load the scans from here and not hae to save them seperately....

    centre_line = CentreLine(map_name)

    for i in range(1, 60):
    # for i in range(20, 60):
    # for i in range(0, 100):
    # for i in range(len(Logs)-100, len(Logs)-50):
    # for i in range(len(Logs)-50, len(Logs)):
    # for i in range(1, len(Logs)):
        # local_track = np.load(localmap_data_path + f"local_map_{i}.npy")
        states = np.load(mpcc_data_path + f"States_{i}.npy")
        controls = np.load(mpcc_data_path + f"Controls_{i}.npy")

        # fig = plt.figure(1)
        fig = plt.figure(figsize=(15, 10))
        plt.clf()
        a1 = plt.subplot2grid((4, 2), (0, 0), rowspan=3)
        ae = plt.subplot2grid((4, 2), (3, 0))
        a2 = plt.subplot2grid((4, 2), (0, 1))
        a3 = plt.subplot2grid((4, 2), (1, 1))
        a4 = plt.subplot2grid((4, 2), (2, 1))
        a5 = plt.subplot2grid((4, 2), (3, 1))

        xs = np.interp(states[:, 3], centre_line.s_path, centre_line.path[:, 0])
        ys = np.interp(states[:, 3], centre_line.s_path, centre_line.path[:, 1])
        a1.plot(xs, ys, 'o', color='orange', markersize=10)

        for z in range(len(states)):
            x_line = [states[z, 0], xs[z]]
            y_line = [states[z, 1], ys[z]]
            a1.plot(x_line, y_line, '--', color='gray', linewidth=1)

        a1.plot(centre_line.path[:, 0], centre_line.path[:, 1], '--', linewidth=2, color='black')
        l1 = centre_line.path + centre_line.nvecs * centre_line.widths[:, 0][:, None]
        l2 = centre_line.path - centre_line.nvecs * centre_line.widths[:, 1][:, None]
        a1.plot(l1[:, 0], l1[:, 1], color='green')
        a1.plot(l2[:, 0], l2[:, 1], color='green')

        a1.plot(Logs[i, 0], Logs[i, 1], '*', markersize=10, color='red')
        a1.set_title(f"Action: ({Logs[i+1, 7]:.3f}, {Logs[i+1, 8]:.1f})")
        a1.axis('off')

        b = 1
        x_min = np.min(xs) - b 
        x_max = np.max(xs) + b
        y_min = np.min(ys) - b
        y_max = np.max(ys) + b
        a1.set_xlim(x_min, x_max)
        a1.set_ylim(y_min, y_max)

        # t_angle = np.interp(states[:, 3], local_ss, psi)
        # lag = np.sum(-np.cos(t_angle) * (states[:, 0] - xs) - np.sin(t_angle) * (states[:, 1] - ys)) * 10
        # contour = np.sum(np.sin(t_angle) * (states[:, 0] - xs) - np.cos(t_angle) * (states[:, 1] - ys)) * 200
        steer = np.sum(controls[:, 0] **2) * 10
        progress = -np.sum(controls[:, 2]) * 0.1
        base = -5
        # a1.text(3, base, f"Lag o: {lag:.2f}")
        # a1.text(3,  base - 0.5, f"Contour o: {contour:.2f}")
        # a1.text(3, base - 1, f"Steer o: {steer:.2f}")
        # a1.text(3, base - 1.5, f"Progress o: {progress:.2f}")

        # ae.bar(np.arange(4), [lag, contour, steer, progress])
        # ae.set_xticks(np.arange(4))
        # ae.set_xticklabels(["Lag", "Contour", "Steer", "Progress"])
        # ae.set_ylim([-8, 8])
        # ae.grid(True)

        vs = controls[:, 1]
        points = states[:, :2].reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        norm = plt.Normalize(2, 8)
        lc = LineCollection(segments, cmap='jet', norm=norm)
        lc.set_array(vs)
        lc.set_linewidth(3)
        line = a1.add_collection(lc)
        plt.sca(a1)
        plt.colorbar(line)
        a1.set_aspect('equal', adjustable='box')


        a2.plot(controls[:, 1])
        a2.plot(controls[:, 2])
        a2.set_ylabel("Speed action")
        a2.grid(True)
        a2.set_ylim(2, 9.5)

        a3.plot(controls[:, 0])
        a3.set_ylabel("Steering action")
        a3.grid(True)
        a3.set_ylim(-0.4, 0.4)

        forces = controls[:, 1] ** 2 / 0.33 * np.tan(np.abs(controls[:, 0])) * 3.71

        a4.plot(forces, '-o', color='red')
        a4.set_ylabel('Lateral Force')
        a4.set_ylim([0, 40])
        a4.grid(True)

        dv = np.diff(controls[:, 1])
        dv = np.insert(dv, 0, controls[0, 1]- Logs[i+1, 3])
        a5.plot(dv, '-o', color='red')
        a5.set_ylabel('Acceleration')
        a5.grid(True)

        plt.tight_layout()

        plt.savefig(mpcc_img_path + f"Raceline_{i}.svg")

        plt.close(fig)


if __name__ == '__main__':
    # render_local_maps("LocalMapPlanner", "r1")
    # render_local_maps("LocalMPCC2", "r1", "aut")
    render_local_maps("FullStackMPCC3", "m3u70", "aut")




