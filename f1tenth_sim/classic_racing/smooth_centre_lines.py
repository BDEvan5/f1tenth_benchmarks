import yaml 
import numpy as np 
import matplotlib.pyplot as plt
from PIL import Image
import csv
import io
import trajectory_planning_helpers as tph

from copy import copy
from f1tenth_sim.classic_racing.planner_utils import *
from f1tenth_sim.data_tools.plotting_utils import *

save_path = f"Logs/map_generation/"
    
class Track:
    def __init__(self, track) -> None:
        self.path = track[:, :2]
        self.widths = track[:, 2:]

        self.el_lengths = np.linalg.norm(np.diff(self.path, axis=0), axis=1)
        self.s_path = np.insert(np.cumsum(self.el_lengths), 0, 0)
        self.psi, self.kappa = tph.calc_head_curv_num.calc_head_curv_num(self.path, self.el_lengths, False)
        self.nvecs = tph.calc_normal_vectors.calc_normal_vectors(self.psi)

    def check_normals_crossing(self):
        track = np.concatenate([self.path, self.widths], axis=1)
        crossing = tph.check_normals_crossing.check_normals_crossing(track, self.nvecs)

        return crossing 

def smooth_centre_line(map_name, smoothing):
    centre_line = CentreLineTrack(map_name)
    centre_track = np.concatenate([centre_line.path, centre_line.widths], axis=1)
    old_track = copy(centre_track)
    centre_track = Track(centre_track)

    crossing = centre_track.check_normals_crossing()
    if not crossing: print(f"No smoothing needed!!!!!!!!!!!!!!")

    track = np.concatenate([centre_track.path, centre_track.widths], axis=1)
    new_track = tph.spline_approximation.spline_approximation(track, 5, smoothing, 0.01, 0.3, True)   
    new_track = Track(new_track)

    if not new_track.check_normals_crossing():
        txt = f"Smoothing ({smoothing}) successful --> Minimum widths, L: {np.min(new_track.widths[:, 0]):.2f}, R: {np.min(new_track.widths[:, 1]):.2f}"
    else: 
        txt = f"Smoothing ({smoothing}) FAILED --> Minimum widths, L: {np.min(new_track.widths[:, 0]):.2f}, R: {np.min(new_track.widths[:, 1]):.2f}"

    smooth_track = np.concatenate([new_track.path, new_track.widths], axis=1)
    map_c_name = f"racelines/{map_name}_centerline.csv"
    with open(map_c_name, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(smooth_track)

    plt.figure(6, figsize=(12, 12))
    plt.clf()
    plt.plot(old_track[:, 0], old_track[:, 1], '-', linewidth=2, color=periwinkle, label="Centre line")
    plt.plot(new_track.path[:, 0], new_track.path[:, 1], '-', linewidth=2, color=red_orange, label="Smoothed track")

    l1 = centre_line.path + centre_line.nvecs * centre_line.widths[:, 0][:, None] # inner
    l2 = centre_line.path - centre_line.nvecs * centre_line.widths[:, 1][:, None] # outer
    plt.plot(l1[:, 0], l1[:, 1], linewidth=1, color=fresh_t)
    plt.plot(l2[:, 0], l2[:, 1], linewidth=1, color=fresh_t)

    l1 = new_track.path + new_track.nvecs * new_track.widths[:, 0][:, None] # inner
    l2 = new_track.path - new_track.nvecs * new_track.widths[:, 1][:, None] # outer

    for i in range(len(new_track.path)):
        plt.plot([l1[i, 0], l2[i, 0]], [l1[i, 1], l2[i, 1]], linewidth=1, color=nartjie)

    plt.plot(l1[:, 0], l1[:, 1], linewidth=1, color=sweedish_green)
    plt.plot(l2[:, 0], l2[:, 1], linewidth=1, color=sweedish_green)

    print(txt)
    plt.title(txt)

    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    plt.savefig(save_path + f"Smoothing_{map_name}.svg")

    print("")

    # plt.show()

if __name__ == '__main__':
    smooth_centre_line("aut", 250)
    smooth_centre_line("esp", 300)
    smooth_centre_line("gbr", 650)
    smooth_centre_line("mco", 300)