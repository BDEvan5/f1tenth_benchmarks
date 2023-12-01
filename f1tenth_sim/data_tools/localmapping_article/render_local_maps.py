import numpy as np 
import matplotlib.pyplot as plt
import os
from f1tenth_sim.data_tools.plotting_utils import *
import trajectory_planning_helpers as tph
from scipy import interpolate
from f1tenth_sim.utils.MapData import MapData

def ensure_path_exists(path):
    if not os.path.exists(path): 
        os.mkdir(path)

def calculate_nvecs(line):
    el_lengths = np.linalg.norm(np.diff(line, axis=0), axis=1)
    psi, kappa = tph.calc_head_curv_num.calc_head_curv_num(line, el_lengths, False)
    nvecs = tph.calc_normal_vectors_ahead.calc_normal_vectors_ahead(psi)

    return nvecs

def interpolate_4d_track(track, point_seperation_distance=0.8, s=0):
    el_lengths = np.linalg.norm(np.diff(track[:, :2], axis=0), axis=1)
    ss = np.insert(np.cumsum(el_lengths), 0, 0)
    order_k = min(3, len(track) - 1)
    tck = interpolate.splprep([track[:, 0], track[:, 1],], u=ss, k=order_k, s=s)[0]
    tck_w = interpolate.splprep([track[:, 2], track[:, 3]], u=ss, k=order_k, s=0)[0]
    n_points = int(ss[-1] / point_seperation_distance  + 1)
    track = np.array(interpolate.splev(np.linspace(0, ss[-1], n_points), tck)).T
    ws = np.array(interpolate.splev(np.linspace(0, ss[-1], n_points), tck_w)).T

    track = np.concatenate((track, ws), axis=1)

    return track

def interpolate_2d_track(track, point_seperation_distance=0.8, s=0):
    el_lengths = np.linalg.norm(np.diff(track[:, :2], axis=0), axis=1)
    ss = np.insert(np.cumsum(el_lengths), 0, 0)
    order_k = min(3, len(track) - 1)
    tck = interpolate.splprep([track[:, 0], track[:, 1],], u=ss, k=order_k, s=s)[0]
    n_points = int(ss[-1] / point_seperation_distance  + 1)
    track = np.array(interpolate.splev(np.linspace(0, ss[-1], n_points), tck)).T

    return track

def interpolate_2d_track_n(track, n_points, s=0):
    el_lengths = np.linalg.norm(np.diff(track[:, :2], axis=0), axis=1)
    ss = np.insert(np.cumsum(el_lengths), 0, 0)
    order_k = min(3, len(track) - 1)
    tck = interpolate.splprep([track[:, 0], track[:, 1],], u=ss, k=order_k, s=s)[0]
    track = np.array(interpolate.splev(np.linspace(0, ss[-1], n_points), tck)).T

    return track

def render_local_maps(planner_name, test_id, map_name="aut"):
    root = f"Logs/{planner_name}/"
    localmap_data_path = root + f"RawData_{test_id}/LocalMapData_{test_id}/"
    logs = np.load(root + f"RawData_{test_id}/SimLog_{map_name}_0.npy")
    scans = np.load(root + f"RawData_{test_id}/ScanLog_{map_name}_0.npy")
    ensure_path_exists(root + f"Images_{test_id}")
    save_path = root + f"Images_{test_id}/LocalMapImgs_{test_id}/"
    ensure_path_exists(save_path)

    map_data = MapData(map_name)

    # for i in range(490, 510):
    # for i in range(250,  350):
    # for i in range(0, 50):
    start = 690
    # start = 580
    n = 10
    # for i in range(len(logs)):
    for i in range(start, start + n):
        local_track = np.load(localmap_data_path + f"local_map_{i}.npy")

        boundaries = np.load(localmap_data_path + f"boundaries_{i}.npy")
        boundary_extension = np.load(localmap_data_path + f"boundExtension_{i}.npy") 

        nvecs = calculate_nvecs(local_track)
        b1 = local_track[:, :2] + nvecs * np.expand_dims(local_track[:, 2], 1)
        b2 = local_track[:, :2] - nvecs * np.expand_dims(local_track[:, 3], 1)
        
        # Plot local map
        plt.figure(1)
        plt.clf()
        map_data.plot_map_img_light()

        # plt.plot(boundaries[:, 0], boundaries[:, 1], '-', color=periwinkle)
        # plt.plot(boundaries[:, 2], boundaries[:, 3], '-', color=periwinkle)
        # if len(boundary_extension) > 0:
        #     plt.plot(boundary_extension[:, 0], boundary_extension[:, 1], '-', color=fresh_t)
        #     plt.plot(boundary_extension[:, 2], boundary_extension[:, 3], '-', color=fresh_t)

        position = logs[i+1, :2]
        orientation = logs[i+1, 4]


        # plt.plot(xs, ys, '-', color='orange', linewidth=3)
        pts = reoreintate_pts(b1, position, orientation)
        xs1, ys1 = map_data.pts2rc(pts)
        # plt.plot(xs1, ys1, '-', color='black', linewidth=3)
        pts = reoreintate_pts(b2, position, orientation)
        xs2, ys2 = map_data.pts2rc(pts)
        # plt.plot(xs2, ys2, '-', color='black', linewidth=3)
        for z in range(b1.shape[0]):
            xs = np.array([xs1[z], xs2[z]])
            ys = np.array([ys1[z], ys2[z]])
            plt.plot(xs, ys, '-', color=good_grey, linewidth=3)
            
        pts = reoreintate_pts(local_track[:, :2], position, orientation)
        xs, ys = map_data.pts2rc(pts)
        plt.plot(xs, ys, '-', color=red_orange, linewidth=3)

        xs, ys = map_data.pts2rc(position[None, :])
        # plt.plot(xs, ys, '*', color=sweedish_green, markersize=15)
        l = 12
        plt.arrow(xs[0], ys[0], np.cos(orientation)*l, np.sin(orientation)*l, color="#8854d0", zorder=10, width=3, head_width=6, head_length=5)
        # plt.arrow(xs[0], ys[0], np.cos(orientation)*l, np.sin(orientation)*l, color=fresh_t, zorder=10, width=3, head_width=6, head_length=5)
        
        # plt.axis('equal')
        plt.tight_layout()
        plt.axis('off')
        b = 10
        x_min = min(np.min(xs1), np.min(xs2)) - b
        x_max = max(np.max(xs1), np.max(xs2)) + b
        y_min = min(np.min(ys1), np.min(ys2)) - b 
        y_max = max(np.max(ys1), np.max(ys2)) + b
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)

        name = save_path + f"LocalMap_{i}"
        plt.savefig(name + ".svg", bbox_inches="tight")
        plt.savefig(f"Data/LocalMapRacing/LocalMaps/Localmap_{i}_{map_name}.pdf", bbox_inches="tight", pad_inches=0.05)
        # plt.show()
        # break


def reoreintate_pts(pts, position, theta):
    # pts = pts - position
    rotation_mtx = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    pts = np.matmul(pts, rotation_mtx.T) + position

    return pts

def check_normals_crossing_complete(track):
    crossing_horizon = min(5, len(track)//2 -1)

    el_lengths = np.linalg.norm(np.diff(track[:, :2], axis=0), axis=1)
    s_track = np.insert(np.cumsum(el_lengths), 0, 0)
    psi, kappa = tph.calc_head_curv_num.calc_head_curv_num(track, el_lengths, False)
    nvecs = tph.calc_normal_vectors_ahead.calc_normal_vectors_ahead(psi-np.pi/2)

    crossing = tph.check_normals_crossing.check_normals_crossing(track, nvecs, crossing_horizon)

    return crossing

if __name__ == '__main__':
    render_local_maps("LocalMapPP", "c1")



