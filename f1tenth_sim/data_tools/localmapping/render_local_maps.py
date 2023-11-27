import numpy as np 
import matplotlib.pyplot as plt
import os
from f1tenth_sim.data_tools.plotting_utils import *
import trajectory_planning_helpers as tph
from scipy import interpolate

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

    # for i in range(490, 510):
    # for i in range(250,  350):
    # for i in range(0, 50):
    for i in range(len(logs)):
        local_track = np.load(localmap_data_path + f"local_map_{i}.npy")


        boundaries = np.load(localmap_data_path + f"boundaries_{i}.npy")
        boundary_extension= np.load(localmap_data_path + f"boundExtension_{i}.npy") 

        nvecs = calculate_nvecs(local_track)
        b1 = local_track[:, :2] + nvecs * np.expand_dims(local_track[:, 2], 1)
        b2 = local_track[:, :2] - nvecs * np.expand_dims(local_track[:, 3], 1)
        
        b1_half = local_track[:, :2] - nvecs * np.expand_dims(local_track[:, 3], 1) * 0.5
        # b1_smooth = interpolate_2d_track_n(b1_half, len(b1_half)*4, 1)
        b1_smooth = interpolate_2d_track_n(b1_half, len(b1_half), 1)


        # Plot local map
        plt.figure(1)
        plt.clf()
        plt.plot(b1_smooth[:, 0], b1_smooth[:, 1], '-', color='black', linewidth=3)
        # plt.plot(b1_half[:, 0], b1_half[:, 1], '-', color='black', linewidth=3)

        # plt.plot(local_track[:, 0], local_track[:, 1], '-', color=sunset_orange, linewidth=3)
        # plt.plot(b1[:, 0], b1[:, 1], '-', color=disco_ball, linewidth=3)
        # plt.plot(b2[:, 0], b2[:, 1], '-', color=disco_ball, linewidth=3)
        # for z in range(b1.shape[0]):
        #     xs = [b1[z, 0], b2[z, 0]]
        #     ys = [b1[z, 1], b2[z, 1]]
        #     plt.plot(xs, ys, '-', color=sweedish_green, linewidth=3)

        plt.plot(boundaries[:, 0], boundaries[:, 1], '-', color=periwinkle)
        plt.plot(boundaries[:, 2], boundaries[:, 3], '-', color=periwinkle)
        if len(boundary_extension) > 0:
            plt.plot(boundary_extension[:, 0], boundary_extension[:, 1], '-', color=fresh_t)
            plt.plot(boundary_extension[:, 2], boundary_extension[:, 3], '-', color=fresh_t)

        # local_track = interpolate_4d_track(local_track, 0.8, 0)
        # local_track = interpolate_4d_track(local_track, 0.6, 50)
        o_nvecs = calculate_nvecs(local_track)
        nvecs = calculate_nvecs(b1_smooth)
        # nvecs[0] = o_nvecs[0]
        # nvecs[-1] = o_nvecs[-1]
        # local_map_smooth =  b1_smooth + nvecs * np.expand_dims(local_track[:, 2], 1) * 0.5
        ob1 = local_track[:, :2] + o_nvecs * np.expand_dims(local_track[:, 2], 1)
        ob2 = local_track[:, :2] - o_nvecs * np.expand_dims(local_track[:, 3], 1)
        b1 = b1_smooth + nvecs * np.expand_dims(local_track[:, 2], 1) * 1.5
        b2 = b1_smooth - nvecs * np.expand_dims(local_track[:, 3], 1) * 0.5

        b1[0] = ob1[0]
        b2[-1] = ob2[-1]

        # if np.linalg.norm(b2[0] - b2[1]) > 1: # add in an additional point if needed here.
            # mid_pt2 = (b2[0] + b2[1]) / 2
            # b2 = np.insert(b2, mid_pt2, 1, axis=0)
            # mid_pt1 = (b1[0] + b1[1]) / 2
            # b1 = np.insert(b1, mid_pt1, 1, axis=0)

        plt.plot(local_track[:, 0], local_track[:, 1], '-', color='orange', linewidth=3)
        plt.plot(b1[:, 0], b1[:, 1], '-', color='black', linewidth=3)
        plt.plot(b2[:, 0], b2[:, 1], '-', color='black', linewidth=3)
        for z in range(b1.shape[0]):
            xs = [b1[z, 0], b2[z, 0]]
            ys = [b1[z, 1], b2[z, 1]]
            plt.plot(xs, ys, '-', color=minty_green, linewidth=3)

        plt.axis('equal')
        plt.tight_layout()
        plt.axis('off')
        name = save_path + f"LocalMap_{i}"
        plt.savefig(name + ".svg", bbox_inches="tight")
        # plt.show()
        # break


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



