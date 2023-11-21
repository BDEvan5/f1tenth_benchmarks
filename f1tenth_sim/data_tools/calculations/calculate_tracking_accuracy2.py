import csv
import numpy as np
from numba import njit
from scipy.interpolate import splev, splprep
from scipy.optimize import fmin
from scipy.spatial import distance 
import glob, os
import pandas as pd
from matplotlib import pyplot as plt
from f1tenth_sim.utils.track_utils import RaceTrack, CentreLine

def calculate_cross_track(track_line, positions):
    s_points = np.zeros(len(positions))
    for i in range(len(positions)):
        s_points[i] = track_line.calculate_progress_percent(positions[i])

    closest_pts = np.array(splev(s_points, track_line.tck, ext=3)).T
    cross_track_errors = np.linalg.norm(positions - closest_pts, axis=1)

    return s_points, cross_track_errors, closest_pts


def calculate_tracking_accuracy(planner_name, test_id, centerline=False):
    agent_path = f"Logs/{planner_name}/"
    print(f"Planner name: {planner_name}")
    old_df = pd.read_csv(agent_path + f"Results_{planner_name}.csv")

    testing_logs = glob.glob(f"{agent_path}RawData_{test_id}/Sim*.npy")
    if len(testing_logs) == 0: raise ValueError("No logs found")
    for test_log in testing_logs:
        test_folder_name = test_log.split("/")[-1]
        test_log_key = "_".join(test_folder_name.split(".")[0].split("_")[1:])
        file_name = f"{agent_path}RawData_{test_id}/TrackingAccuracy_{test_log_key}.npy"
        lap_num = int(test_folder_name.split("_")[-1].split(".")[0])
        # if os.path.exists(file_name): continue

        print(f"Analysing log: {test_folder_name} ")

        testing_map = test_folder_name.split("_")[1]
        if not centerline:
            std_track = RaceTrack(testing_map, test_id)
            std_track.init_track()
        else:
            std_track = CentreLine(testing_map)

        states = np.load(test_log)[:, :7]
        progresses, cross_track, points = calculate_cross_track(std_track, states[:, 0:2]) 

        df_idx = old_df.loc[(old_df["Lap"] == lap_num) & (old_df["TestMap"] == testing_map)].index[0]
        old_df.at[df_idx, "MeanCT"] = np.mean(cross_track)
        old_df.at[df_idx, "MaxCT"] = np.max(cross_track)

        save_data = np.column_stack((progresses, cross_track, points))
        np.save(file_name, save_data)

    old_df = old_df.sort_values(by=["TestMap", "Lap"])
    old_df.to_csv(f"{agent_path}Results_{planner_name}.csv", index=False, float_format='%.4f')


if __name__ == "__main__":
    calculate_tracking_accuracy("GlobalPlanPP", "mu70")

