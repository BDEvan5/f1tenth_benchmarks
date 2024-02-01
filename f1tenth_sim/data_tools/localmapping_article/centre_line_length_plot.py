import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np
import pandas as pd 

import os
from f1tenth_sim.data_tools.plotting_utils import *
from f1tenth_sim.utils.track_utils import CentreLine

def ensure_path_exists(path):
    if not os.path.exists(path): 
        os.mkdir(path)

def load_data(planner_name, test_id, map_name="aut"):
    root = f"Logs/{planner_name}/"
    localmap_data_path = root + f"RawData_{test_id}/LocalMapData_{test_id}/"
    try:
        Logs = np.load(root + f"RawData_{test_id}/SimLog_{map_name}_0.npy")
    except:
        Logs, scans = None, None
    save_path = root + f"LocalMapGeneration_{test_id}/"
    ensure_path_exists(save_path)

    lengths = []
    for i in range(len(Logs)):
        local_track = np.load(localmap_data_path + f"local_map_{i}.npy")
        el_lengths = np.linalg.norm(np.diff(local_track[:, :2], axis=0), axis=1)
        lengths.append(np.sum(el_lengths))

    return np.array(lengths)

def load_boundary_data(planner_name, test_id, map_name="aut"):
    root = f"Logs/{planner_name}/"
    localmap_data_path = root + f"RawData_{test_id}/LocalMapData_{test_id}/"
    try:
        Logs = np.load(root + f"RawData_{test_id}/SimLog_{map_name}_0.npy")
    except:
        Logs, scans = None, None
    save_path = root + f"LocalMapGeneration_{test_id}/"
    ensure_path_exists(save_path)

    calculated_lengths = []
    projected_lengths = []
    for i in range(len(Logs)):

        boundaries = np.load(localmap_data_path + f"boundaries_{i}.npy")
        boundary_extension= np.load(localmap_data_path + f"boundExtension_{i}.npy") 

        calculated_line = (boundaries[:, :2] + boundaries[:, 2:]) /2
        calculated_el_lengths = np.linalg.norm(np.diff(calculated_line, axis=0), axis=1)
        calculated_lengths.append(np.sum(calculated_el_lengths, axis=0))
        if boundary_extension.shape[0] > 0:
            projected_line = (boundary_extension[:, :2] + boundary_extension[:, 2:]) /2
            projected_el_lengths = np.linalg.norm(np.diff(projected_line, axis=0), axis=1)
            projected_lengths.append(np.sum(projected_el_lengths, axis=0))
        else:
            projected_lengths.append(0)

    return np.array(calculated_lengths), np.array(projected_lengths), Logs[:, -1]


def make_violin_plot(test_id):
    map_list = ["aut", "esp", "gbr", "mco"]
    data = {}
    data_dict = []
    for map_name in map_list:
        lengths = load_data("LocalMapPP", test_id, map_name)
        # data[map_name] = lengths
        for z in lengths:
            data_dict.append({"map": map_name, "length": z})

    df = pd.DataFrame.from_records(data_dict)

    plt.figure(figsize=(5, 1.9))

    ax = sns.violinplot(x='map', y="length", data=df)
    ax.set(xlabel=None)
    # plt.title('Violin Plot of Values Frequency')
    # plt.xlabel()
    plt.ylabel('Centre line \nlength (m)')
    plt.gca().set_ylim(bottom=0)
    plt.gca().yaxis.set_major_locator(plt.MaxNLocator(6))
    # plt.xticks(map_list, ["AUT", "ESP", "GBR", "MCO"])
    plt.xticks(range(4), ["AUT", "ESP", "GBR", "MCO"])
    plt.grid(axis='y', alpha=0.8)
    
    plt.tight_layout()
    plt.savefig(f"Data/LocalMapRacing/LengthViolin_{test_id}.svg", bbox_inches='tight', pad_inches=0.01)
    plt.savefig(f"Data/LocalMapRacing/LengthViolin_{test_id}.pdf", bbox_inches='tight', pad_inches=0.01)


def make_length_progress_plot(map_name, test_id):
    calculated, projected, progresses = load_boundary_data("LocalMapPP", test_id, map_name)
    progresses *= 100

    centre = CentreLine(map_name)

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(5, 2.5), gridspec_kw={'height_ratios': [2, 1]})
    # fig = plt.figure(figsize=(5, 2.5))
    # ax2 = plt.subplot(2, 1, 2, width)
    # ax1 = plt.subplot(2, 1, 1)

    ax1.fill_between(progresses, 0, calculated, color=sweedish_green, label="Calculated", alpha=0.5)
    ax1.fill_between(progresses, calculated, calculated+projected, color=disco_ball, alpha=0.5, label="Projected")

    s_norm = centre.s_path / centre.s_path[-1] * 100
    curvature_deg = abs(np.rad2deg(centre.kappa))
    ax2.plot(s_norm, curvature_deg, color='black', label="Curvature", linewidth=2)
    ax2.yaxis.set_major_locator(plt.MaxNLocator(3))

    ax1.yaxis.set_major_locator(plt.MaxNLocator(5))
    # ax1.set_xticklabels([])
    ax1.set_ylim(-0.5, 23)
    ax2.set_xlabel("Track progress [%]")
    ax1.set_ylabel("Centre line \nlength [m]")
    ax2.set_ylabel("Curvature \n[deg/m]")
    ax1.legend(ncol=2)
    plt.tight_layout()
    ax1.grid()
    ax2.grid()
    plt.rcParams['pdf.use14corefonts'] = True

    plt.savefig(f"Data/LocalMapRacing/LengthProgress_{map_name}_{test_id}.svg", bbox_inches='tight', pad_inches=0.01)
    plt.savefig(f"Data/LocalMapRacing/LengthProgress_{map_name}_{test_id}.pdf", bbox_inches='tight', pad_inches=0.01)

# make_violin_plot("centre")

make_length_progress_plot("aut", "centre")

