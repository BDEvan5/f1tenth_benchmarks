import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib.collections import LineCollection
from f1tenth_benchmarks.utils.track_utils import RaceTrack, CentreLine
from f1tenth_benchmarks.data_tools.plotting_utils import *
from f1tenth_benchmarks.utils.MapData import MapData


class RaceTrackPlotter(RaceTrack):
    def __init__(self, map_name, raceline_id) -> None:
        super().__init__(map_name, raceline_id)
        self.raceline_id = raceline_id
        self.centre_line = CentreLine(map_name)
        self.map_data = MapData(map_name)
        self.raceline_data_path = f"Data/raceline_data/{raceline_id}/"

        self.plot_neat_img()

    def plot_neat_img(self):
        fig = plt.figure(1)
        plt.clf()

        self.map_data.plot_map_img()

        xs, ys = self.map_data.pts2rc(self.centre_line.path)
        plt.plot(xs, ys, 'k--', linewidth=2, label="Track")

        xs, ys = self.map_data.pts2rc(self.path)
        path = np.stack([xs, ys], axis=1)
        points = path.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        norm = plt.Normalize(2, 8)
        lc = LineCollection(segments, cmap='jet', norm=norm)
        lc.set_array(self.speeds)
        lc.set_linewidth(5)
        ax1 = plt.gca()
        line = plt.gca().add_collection(lc)

        ax = plt.gca()
        cax = fig.add_axes([ax.get_position().x1-0.05, ax.get_position().y0+0.35, 0.025, ax.get_position().height*0.6])
        cbar = plt.colorbar(line, cax=cax, label="Speed [m/s]", ticks=[2, 4, 6, 8], shrink=0.5)
        # cbar = plt.colorbar(line, label="Speed [m/s]", ticks=[2, 4, 6, 8])
        cbar.ax.tick_params(labelsize=20)
        cbar.set_label(label='Speed [m/s]', size=18)

        ax1.text(50, 50, "Global raceline", fontsize=20)

        # plt.xlim(self.path[:, 0].min()-1, self.path[:, 0].max()+1)
        # plt.ylim(self.path[:, 1].min()-1, self.path[:, 1].max()+1)
        ax1.set_aspect('equal', adjustable='box')
        ax1.set_axis_off()

        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(self.raceline_data_path + f"neat_raceline_{self.map_name}.svg", pad_inches=0, bbox_inches="tight")



if __name__ == '__main__':
    map_list = ['esp']
    # map_list = ['aut']
    # map_list = ['aut', 'esp', 'gbr', 'mco']
    raceline_id = "mu70"
    for map_name in map_list:
        RaceTrackPlotter(map_name, raceline_id)



