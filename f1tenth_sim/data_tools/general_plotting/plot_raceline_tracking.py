from matplotlib import pyplot as plt
plt.rcParams['pdf.use14corefonts'] = True

import numpy as np
import glob
import os
import math, cmath

import glob
from matplotlib.ticker import PercentFormatter
from matplotlib.collections import LineCollection

from f1tenth_controllers.map_utils.MapData import MapData
from f1tenth_controllers.map_utils.Track import Track 
from f1tenth_controllers.analysis.plotting_utils import *
from matplotlib.ticker import MultipleLocator
import trajectory_planning_helpers as tph

SAVE_PDF = False
# SAVE_PDF = True


def ensure_path_exists(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


MAX_TRACKING_ERROR = 40 #cm

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
        self.tracking_points = None

    def process_folder(self, folder, test_id):
        self.load_folder = folder + f"RawData_{test_id}/"
        self.save_folder = folder + f"Images_{test_id}/"
        ensure_path_exists(self.save_folder)
        if SAVE_PDF:
            self.pdf_save_folder = self.save_folder + f"Images_pdf_{test_id}/"
            ensure_path_exists(self.pdf_save_folder)

        self.vehicle_name = folder.split("/")[-2]
        print(f"Vehicle name: {self.vehicle_name}")
        
        testing_logs = glob.glob(f"{self.load_folder}*.npy")
        for test_log in testing_logs:
            test_folder_name = test_log.split("/")[-1]
            self.test_log_key = test_folder_name.split(".")[0].split("_")[1:]
            self.test_log_key = "_".join(self.test_log_key)
            self.map_name = test_folder_name.split("_")[1]
        
            self.map_data = MapData(self.map_name)

            if not self.load_lap_data(): break # no more laps
            
            self.plot_tracking_accuracy()
            self.plot_tracking_heatmap()
            self.plot_tracking_path()

    def load_lap_data(self):
        data = np.load(f"{self.load_folder}SimLog_{self.test_log_key}.npy")
        self.states = data[:, :7]
        self.actions = data[:, 7:9]
        # self.track_progresses = data[:, 9]

        accuracy_data = np.load(f"{self.load_folder}TrackingAccuracy_{self.test_log_key}.npy")
        self.track_progresses = accuracy_data[:, 0] * 100
        # rather use racing line progresses here
        self.tracking_accuracy = accuracy_data[:, 1] * 100
        self.tracking_points = accuracy_data[:, 2:4]
        
        return 1 # to say success

    def plot_tracking_heatmap(self): 
        plt.figure(1)
        plt.clf()
        self.map_data.plot_map_img()

        xs, ys = self.map_data.pts2rc(self.states[:, 0:2])
        points = np.concatenate([xs[:, None], ys[:, None]], axis=1)
        points = points.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        norm = plt.Normalize(0, MAX_TRACKING_ERROR)
        lc = LineCollection(segments, cmap='bwr', norm=norm)
        lc.set_array(self.tracking_accuracy)
        lc.set_linewidth(5)
        line = plt.gca().add_collection(lc)
        cbar = plt.colorbar(line,fraction=0.046, pad=0.04, shrink=0.99)
        cbar.ax.tick_params(labelsize=18)
        plt.gca().set_aspect('equal', adjustable='box')

        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.axis('off')
        
        name = self.save_folder + f"RacelineTrackingHeatMap_{self.test_log_key}"
        plt.savefig(name + ".svg", bbox_inches='tight', pad_inches=0)
        if SAVE_PDF:
            name = self.pdf_save_folder + f"RacelineTrackingHeatMap_{self.test_log_key}"
            plt.savefig(name + ".pdf", bbox_inches='tight', pad_inches=0)
    
    def plot_tracking_path(self): 
        plt.figure(1)
        plt.clf()
        self.map_data.plot_map_img()

        xs, ys = self.map_data.pts2rc(self.states[:, 0:2])
        plt.plot(xs, ys, color=sunset_orange, alpha=0.6, linewidth=1, label="Vehicle")

        xs, ys = self.map_data.pts2rc(self.tracking_points)
        plt.plot(xs, ys, color=periwinkle, alpha=0.6, linewidth=1, label="Raceline")

        plt.legend(fontsize=16)
        plt.gca().set_aspect('equal', adjustable='box')

        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.axis('off')
        
        name = self.save_folder + f"TrackingAccuracy_{self.test_log_key}"
        plt.savefig(name + ".svg", bbox_inches='tight', pad_inches=0)

    def plot_tracking_accuracy(self):
        plt.figure(1, figsize=(10, 5))
        plt.clf()
        plt.plot(self.track_progresses, self.tracking_accuracy)
        
        plt.ylim(0, MAX_TRACKING_ERROR)
        plt.title("Tracking Accuracy (cm)")
        plt.xlabel("Track Progress (%)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{self.save_folder}Tracking_{self.test_log_key}.svg", bbox_inches='tight', pad_inches=0)

    def plot_trackign_error_distribution(self):
        plt.figure(1, figsize=(5, 4))
        plt.clf()
        bins = np.linspace(0, MAX_TRACKING_ERROR, 21) - 1
        bars = np.digitize(np.clip(self.tracking_accuracy, 0, MAX_TRACKING_ERROR), bins)
        bars = np.bincount(bars, minlength=len(bins))
        bars = bars / np.sum(bars) * 100
        plt.bar(bins, bars[:21], width=2)

        plt.xlabel("Tracking Accuracy (cm)")
        plt.ylabel("Frequency (%)")
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator())
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{self.save_folder}TrackingHist_{self.test_log_key}.svg", bbox_inches='tight', pad_inches=0)

        

def plot_raceline_tracking(vehicle_name, test_id):
    TestData = TrajectoryPlotter()

    TestData.process_folder(f"Logs/{vehicle_name}/", test_id)



if __name__ == '__main__':
    plot_raceline_tracking("GlobalPlanPP", "mu70")
