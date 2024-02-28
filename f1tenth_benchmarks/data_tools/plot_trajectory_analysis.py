from matplotlib import pyplot as plt
plt.rcParams['pdf.use14corefonts'] = True

import numpy as np
import glob
import os

import glob
from matplotlib.collections import LineCollection

from f1tenth_benchmarks.utils.MapData import MapData
from f1tenth_benchmarks.data_tools.plotting_utils import *
from matplotlib.ticker import MultipleLocator
from f1tenth_benchmarks.utils.track_utils import CentreLine, TrackLine


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
        
        testing_Logs = glob.glob(f"{self.load_folder}/SimLog*.npy")
        for test_log in testing_Logs:
            test_folder_name = test_log.split("/")[-1]
            self.test_log_key = test_folder_name.split(".")[0].split("_")[1:]
            self.test_log_key = "_".join(self.test_log_key)
            self.map_name = test_folder_name.split("_")[1]
        
            self.map_data = MapData(self.map_name)

            if not self.load_lap_data(): break # no more laps
            
            self.plot_analysis()
            self.plot_trajectory()
            self.plot_path()

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

    def plot_path(self):
        plt.figure(1)
        plt.clf()
        
        self.map_data.plot_map_img()

        centre_line = CentreLine(self.map_name)
        xs, ys = self.map_data.pts2rc(centre_line.path)
        plt.plot(xs, ys, 'k--', linewidth=1.5)

        xs, ys = self.map_data.pts2rc(self.states[:, 0:2])
        plt.plot(xs, ys, 'r-', linewidth=1.5)

        plt.gca().set_aspect('equal', adjustable='box')

        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.axis('off')
        
        name = self.save_folder + f"Path_{self.test_log_key}"
        # std_img_saving(name)
        plt.savefig(name + ".svg", bbox_inches='tight', pad_inches=0)
        
        if SAVE_PDF:
            name = self.pdf_save_folder + f"Path_{self.test_log_key}"
            plt.savefig(name + ".pdf", bbox_inches='tight', pad_inches=0)


        plt.close()


def plot_trajectory_analysis(vehicle_name, test_id):

    TestData = TrajectoryPlotter()
    TestData.process_folder(f"Logs/{vehicle_name}/", test_id)



if __name__ == '__main__':
    # analyse_folder()
    # plot_trajectory_analysis("LocalMPCC", "mu60")
    # plot_trajectory_analysis("FullStackMPCC", "mu60")
    # plot_trajectory_analysis("EndToEnd", "TD3_CTH_13_mco")
    # plot_trajectory_analysis("EndToEnd", "TD3_Progress_12_aut")
    # plot_trajectory_analysis("EndToEnd", "TD3_CTH_99999_mco")
    # plot_trajectory_analysis("EndToEnd", "TD3_TAL_12_aut")
    # plot_trajectory_analysis("ConstantMPCC", "mu70")
    # plot_trajectory_analysis("GlobalPlanPP", "mu95_steps10")
    # plot_trajectory_analysis("FullStackPP", "mu60")
    plot_trajectory_analysis("FullStackMPCC", "mpcc_t3")
    # plot_trajectory_analysis("GlobalPlanMPCC", "mu70")
