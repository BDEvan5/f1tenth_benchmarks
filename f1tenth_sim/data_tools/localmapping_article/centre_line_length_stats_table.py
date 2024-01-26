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
        raise Exception("No Logs found")
    save_path = root + f"LocalMapGeneration_{test_id}/"
    ensure_path_exists(save_path)

    lengths = []
    for i in range(len(Logs)-1):
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
    for i in range(len(Logs)-1):

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


def make_extraction_table(test_id):
    map_list = ["aut", "esp", "gbr"]
    data = {}
    data_dict = []
    for map_name in map_list:
        lengths = load_data("LocalMapPP", test_id, map_name)
        # data[map_name] = lengths
        for z in lengths:
            data_dict.append({"map": map_name, "length": z})

    df = pd.DataFrame.from_records(data_dict)

    descriptions = df.groupby("map").describe(percentiles=[])
    descriptions = descriptions.drop(columns=[("length", "count")]).T
    descriptions = descriptions.round(2)
    descriptions.to_latex("Data/LocalMapRacing/LengthTable.tex", float_format="%.2f", na_rep="-")


make_extraction_table("c1")

