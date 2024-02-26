import numpy as np 
import matplotlib.pyplot as plt
import os
from f1tenth_benchmarks.data_tools.plotting_utils import *
from f1tenth_benchmarks.utils.MapData import MapData

def ensure_path_exists(path):
    if not os.path.exists(path): 
        os.mkdir(path)

def render_local_maps(planner_name, test_id, map_name="aut"):
    root = f"Logs/{planner_name}/"
    localmap_data_path = root + f"RawData_{test_id}/LocalMapData_{test_id}/"
    try:
        Logs = np.load(root + f"RawData_{test_id}/SimLog_{map_name}_0.npy")
        scans = np.load(root + f"RawData_{test_id}/ScanLog_{map_name}_0.npy")
    except:
        Logs, scans = None, None
    ensure_path_exists(root + f"Images_{test_id}")
    save_path = root + f"Images_{test_id}/LocalMapGeneration_{test_id}/"
    ensure_path_exists(save_path)

    angles = np.linspace(-2.35619449615, 2.35619449615, 1080)
    coses = np.cos(angles)
    sines = np.sin(angles)

    map_data = MapData(map_name)

    n = 10
    start = 410
    # start = 300
    # for i in range(start, start+n):
    # for i in range(len(Logs)-50, len(Logs)):
    # for i in range(490, 510):
    # for i in range(250,  350):
    # for i in range(200, 300):
    # for i in range(100, 200):
    for i in range(len(Logs)-1):
        # if scans:
        scan_xs, scan_ys = scans[i+1] * np.array([coses, sines])
        position = Logs[i+1, :2]
        orientation = Logs[i+1, 4]
        
        if i % 5 != 0:
            continue

        local_track = np.load(localmap_data_path + f"local_map_{i}.npy")
        line_1 = np.load(localmap_data_path + f"line1_{i}.npy")
        line_2 = np.load(localmap_data_path + f"line2_{i}.npy")
        boundaries = np.load(localmap_data_path + f"boundaries_{i}.npy")
        boundary_extension= np.load(localmap_data_path + f"boundExtension_{i}.npy") 

        # Plot local map
        plt.figure(1)
        plt.clf()
        map_data.plot_map_img_light()

        scan_pts = np.vstack([scan_xs, scan_ys]).T
        scan_pts = reoreintate_pts(scan_pts, position, orientation)
        scan_xs, scan_ys = map_data.pts2rc(scan_pts)
        plt.plot(scan_xs, scan_ys, '.', color=free_speech, alpha=0.5)

        # plt.plot(0, 0, '*', markersize=12, color='red')


        # plt.plot(local_track[:, 0], local_track[:, 1], '-X', color='orange', markersize=10)

        boundary1 = reoreintate_pts(boundaries[:, :2], position, orientation)
        xs1, ys1 = map_data.pts2rc(boundary1)
        boundary2 = reoreintate_pts(boundaries[:, 2:], position, orientation)
        xs2, ys2 = map_data.pts2rc(boundary2)


        for z in range(len(boundaries)):
            xs = np.array([xs1[z], xs2[z]])
            ys = np.array([ys1[z], ys2[z]])

            # xs = [boundaries[z, 0], boundaries[z, 2]]
            # ys = [boundaries[z, 1], boundaries[z, 3]]
            plt.plot(xs, ys, '-o', color=sweedish_green, markersize=5)
            # plt.plot(xs, ys, '-o', color='black', markersize=5)
        
        if len(boundary_extension) > 0:
            be1 = reoreintate_pts(boundary_extension[:, :2], position, orientation)
            xss1, yss1 = map_data.pts2rc(be1)
            be2 = reoreintate_pts(boundary_extension[:, 2:], position, orientation)
            xss2, yss2 = map_data.pts2rc(be2)
            for z in range(boundary_extension.shape[0]):
                xs = np.array([xss1[z], xss2[z]])
                ys = np.array([yss1[z], yss2[z]])
                # xs = [boundary_extension[z, 0], boundary_extension[z, 2]]
                # ys = [boundary_extension[z, 1], boundary_extension[z, 3]]
                plt.plot(xs, ys, '-o', color=fresh_t, markersize=5)
                # plt.plot(xs, ys, '-o', color='pink', markersize=5)
        else:
            xss1, xss2 = [xs1[0]], [xs2[0]]
            yss1, yss2 = [ys1[0]], [ys2[0]]
        # plt.plot(local_track[:, 0], local_track[:, 1], '-', color='orange', linewidth=3)
        # plt.plot(line_1[:, 0], line_1[:, 1], '-', color=minty_green, linewidth=2)
        # plt.plot(line_2[:, 0], line_2[:, 1], '-', color=minty_green, linewidth=2)

        xs, ys = map_data.pts2rc(position[None, :])
        # plt.plot(xs, ys, '*', color=sweedish_green, markersize=15)
        l = 12
        plt.arrow(xs[0], ys[0], np.cos(orientation)*l, np.sin(orientation)*l, color="#8854d0", zorder=10, width=3, head_width=6, head_length=5)

        try:
            b = 10
            x_min = min(np.min(xs1), np.min(xs2), np.min(xss1), np.min(xss2)) - b
            x_max = max(np.max(xs1), np.max(xs2), np.max(xss1), np.max(xss2)) + b
            y_min = min(np.min(ys1), np.min(ys2), np.min(yss1), np.min(yss2)) - b 
            y_max = max(np.max(ys1), np.max(ys2), np.max(yss1), np.max(yss2)) + b
            plt.xlim(x_min, x_max)
            plt.ylim(y_min, y_max)

        except:
            pass

        # plt.axis('equal')
        plt.tight_layout()
        plt.axis('off')
        name = save_path + f"LocalMapGeneration_{i}"
        plt.savefig(name + ".svg", bbox_inches="tight")
        # plt.savefig(f"Data/LocalMapRacing/LocalMaps/LocalGeneration_{i}_aut.pdf", bbox_inches="tight", pad_inches=0.05)
        # plt.show()
        # break

def reoreintate_pts(pts, position, theta):
    rotation_mtx = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    pts = np.matmul(pts, rotation_mtx.T) + position

    return pts

if __name__ == '__main__':
    render_local_maps("LocalMapPP", "c1")
    # render_local_maps("LocaleMPCC", "mu60", "aut")
    # render_local_maps("LocalMapPlanner", "r1", "mco")



