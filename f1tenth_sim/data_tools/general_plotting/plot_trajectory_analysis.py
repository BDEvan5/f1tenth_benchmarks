from matplotlib import pyplot as plt
plt.rcParams['pdf.use14corefonts'] = True

import numpy as np
import glob
import os

import glob
from matplotlib.collections import LineCollection

from f1tenth_controllers.map_utils.MapData import MapData
from f1tenth_controllers.map_utils.Track import Track 
from f1tenth_controllers.analysis.plotting_utils import *
from matplotlib.ticker import MultipleLocator
from scipy.interpolate import splev, splprep
import pandas as pd
from f1tenth_sim.utils.track_utils import RaceTrack, CentreLine

SAVE_PDF = False
# SAVE_PDF = True


def ensure_path_exists(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


class TrajectoryPlotter:
    def __init__(self):
        self.vehicle_name = None
        self.map_name = None
        self.states = None
        self.actions = None
        self.map_data = None
        self.lap_n = 0
        
        self.track_progresses = None
        self.tracking_accuracy = None

        self.load_folder = None
        self.save_folder = None
        self.pdf_save_folder = None

    def process_folder(self, folder, test_id):
        self.load_folder = folder + f"RawData_{test_id}/"
        self.save_folder = folder + f"Images_{test_id}/"
        ensure_path_exists(self.save_folder)
        if SAVE_PDF:
            self.pdf_save_folder = folder + f"Images_pdf_{test_id}/"
            ensure_path_exists(self.pdf_save_folder)

        self.vehicle_name = folder.split("/")[-2]
        print(f"Vehicle name: {self.vehicle_name}")
        
        testing_logs = glob.glob(f"{self.load_folder}/SimLog*.npy")
        for test_log in testing_logs:
            test_folder_name = test_log.split("/")[-1]
            self.test_log_key = test_folder_name.split(".")[0].split("_")[1:]
            self.test_log_key = "_".join(self.test_log_key)
            self.map_name = test_folder_name.split("_")[1]
        
            self.map_data = MapData(self.map_name)

            if not self.load_lap_data(): break # no more laps
            
            self.plot_analysis()
            self.plot_trajectory()

    def load_lap_data(self):
        data = np.load(f"{self.load_folder}SimLog_{self.test_log_key}.npy")
        self.states = data[:, :7]
        self.actions = data[:, 7:9]
        self.track_progresses = data[:, 9]

        return 1 # to say success
    

    def plot_analysis(self):
        fig = plt.figure(figsize=(8, 6))
        a1 = plt.subplot(311)
        a2 = plt.subplot(312, sharex=a1)
        a3 = plt.subplot(313, sharex=a1)
        plt.rcParams['lines.linewidth'] = 2

        a1.plot(self.track_progresses[:-1], self.actions[:-1, 1], label="Actions", alpha=0.6, color=sunset_orange)
        a1.plot(self.track_progresses[:-1], self.states[:-1, 3], label="State", color=periwinkle)
        a1.set_ylabel("Speed (m/s)")
        a1.grid(True)

        a2.plot(self.track_progresses[:-1], self.states[:-1, 2], label="State", color=periwinkle)
        a2.plot(self.track_progresses[:-1], self.actions[:-1, 0], label="Actions", color=sunset_orange)
        max_value = np.max(np.abs(self.actions[:-1, 0])) * 1.1
        a2.set_ylim(-max_value, max_value)
        a2.grid(True)
        a2.legend()
        a2.set_ylabel("Steering (rad)")
    
        a3.plot(self.track_progresses[:-1], self.states[:-1, 6], label="State", color=periwinkle)
        max_value = np.max(np.abs(self.states[:-1, 6])) * 1.1
        a3.set_ylim(-max_value, max_value)
        a3.set_ylabel("Slip (rad)")
        a3.grid(True)
        a3.set_xlabel("Track progress (m)")

        plt.tight_layout()
        plt.savefig(f"{self.save_folder}Analysis_{self.test_log_key}.svg", bbox_inches='tight', pad_inches=0)  
    
    def plot_trajectory(self): 
        plt.figure(1)
        plt.clf()
        points = self.states[:, 0:2]
        vs = self.states[:, 3]
        
        self.map_data.plot_map_img()

        xs, ys = self.map_data.pts2rc(points)
        points = np.concatenate([xs[:, None], ys[:, None]], axis=1)
        points = points.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        norm = plt.Normalize(0, 8)
        lc = LineCollection(segments, cmap='jet', norm=norm)
        lc.set_array(vs)
        lc.set_linewidth(5)
        line = plt.gca().add_collection(lc)
        cbar = plt.colorbar(line,fraction=0.046, pad=0.04, shrink=0.99)
        cbar.ax.tick_params(labelsize=25)
        plt.gca().set_aspect('equal', adjustable='box')

        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.axis('off')
        
        name = self.save_folder + f"Trajectory_{self.test_log_key}"
        # std_img_saving(name)
        plt.savefig(name + ".svg", bbox_inches='tight', pad_inches=0)
        
        if SAVE_PDF:
            name = self.pdf_save_folder + f"Trajectory_{self.test_log_key}"
            plt.savefig(name + ".pdf", bbox_inches='tight', pad_inches=0)



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


def plot_trajectory_analysis(vehicle_name, test_id):
    calculate_tracking_accuracy(vehicle_name, test_id, centerline=False)

    TestData = TrajectoryPlotter()
    TestData.process_folder(f"Logs/{vehicle_name}/", test_id)



if __name__ == '__main__':
    # analyse_folder()
    plot_trajectory_analysis("GlobalPlanPP", "mu70")
    # plot_trajectory_analysis("GlobalPlanMPCC", "mu70")
    # plot_analysis("follow_the_gap")
    # plot_analysis("TD3_endToEnd_5")
    # plot_analysis("SAC_endToEnd_5")
    # plot_analysis("TD3_endToEnd_1")
