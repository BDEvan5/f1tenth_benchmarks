import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib.collections import LineCollection
from f1tenth_benchmarks.utils.track_utils import RaceTrack, CentreLine
from f1tenth_benchmarks.data_tools.plotting_utils import *


class RaceTrackPlotter(RaceTrack):
    def __init__(self, map_name, raceline_id) -> None:
        super().__init__(map_name, raceline_id)
        self.raceline_id = raceline_id
        self.centre_line = CentreLine(map_name)
        self.raceline_data_path = f"Data/raceline_data/{raceline_id}/"

        self.plot_minimum_curvature_path()
        self.plot_raceline_trajectory()

    def plot_minimum_curvature_path(self):
        plt.figure(3)
        plt.clf()
        plt.plot(self.centre_line.path[:, 0], self.centre_line.path[:, 1], '-', linewidth=2, color=periwinkle, label="Track")
        plt.plot(self.path[:, 0], self.path[:, 1], '-', linewidth=2, color=sizzling_red, label="Raceline")

        l_line = self.centre_line.path + self.centre_line.nvecs * self.centre_line.widths[:, 0][:, None]
        r_line = self.centre_line.path - self.centre_line.nvecs * self.centre_line.widths[:, 1][:, None]
        plt.plot(l_line[:, 0], l_line[:, 1], linewidth=1, color=sweedish_green, label="Boundaries")
        plt.plot(r_line[:, 0], r_line[:, 1], linewidth=1, color=sweedish_green)
        
        plt.legend()
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(self.raceline_data_path + f"raceline_path_{self.map_name}.svg", pad_inches=0)

    def plot_raceline_trajectory(self):
        plt.figure(1)
        plt.clf()

        plt.plot(self.centre_line.path[:, 0], self.centre_line.path[:, 1], '-', linewidth=2, color=periwinkle, label="Track")

        l_line = self.centre_line.path + self.centre_line.nvecs * self.centre_line.widths[:, 0][:, None]
        r_line = self.centre_line.path - self.centre_line.nvecs * self.centre_line.widths[:, 1][:, None]
        plt.plot(l_line[:, 0], l_line[:, 1], linewidth=1, color=sweedish_green, label="Boundaries")
        plt.plot(r_line[:, 0], r_line[:, 1], linewidth=1, color=sweedish_green)
        

        points = self.path.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        norm = plt.Normalize(0, 8)
        lc = LineCollection(segments, cmap='jet', norm=norm)
        lc.set_array(self.speeds)
        lc.set_linewidth(5)
        line = plt.gca().add_collection(lc)
        plt.colorbar(line)
        plt.xlim(self.path[:, 0].min()-1, self.path[:, 0].max()+1)
        plt.ylim(self.path[:, 1].min()-1, self.path[:, 1].max()+1)
        plt.gca().set_aspect('equal', adjustable='box')

        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(self.raceline_data_path + f"raceline_speeds_{self.map_name}.svg", pad_inches=0)



if __name__ == '__main__':
    map_list = ['aut', 'esp', 'gbr', 'mco']
    raceline_id = "mu60"
    for map_name in map_list:
        RaceTrackPlotter(map_name, raceline_id)



