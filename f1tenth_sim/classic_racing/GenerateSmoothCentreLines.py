import numpy as np 
import trajectory_planning_helpers as tph
from f1tenth_sim.utils.BasePlanner import *
from f1tenth_sim.utils.track_utils import CentreLine
from f1tenth_sim.data_tools.specific_plotting.plot_racelines import RaceTrackPlotter
import matplotlib.pyplot as plt

from copy import copy
from f1tenth_sim.data_tools.plotting_utils import *

np.printoptions(precision=3, suppress=True)


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

def smooth_centre_line2(centre_track, track):
    coeffs_x, coeffs_y, A, normvec_normalized = tph.calc_splines.calc_splines(centre_track.path, centre_track.el_lengths, psi_s=centre_track.psi[0], psi_e=centre_track.psi[-1])
    
    alpha, error = tph.opt_min_curv.opt_min_curv(track, centre_track.nvecs, A, 1, 0, print_debug=True, closed=False, fix_s=True, psi_s=centre_track.psi[0], psi_e=centre_track.psi[-1], fix_e=True)

    # new_path, A_raceline, coeffs_x_raceline, coeffs_y_raceline, spline_inds_raceline_interp, t_values_raceline_interp, s_raceline, spline_lengths_raceline, el_lengths_raceline_interp_cl = tph.create_raceline.create_raceline(centre_track.path, centre_track.nvecs, alpha, 0.2) 


    new_path = centre_track.path + centre_track.nvecs * alpha[:, None]

    new_widths = centre_track.widths.copy()
    new_widths[:, 0] -= alpha
    new_widths[:, 1] += alpha
    new_track = np.concatenate([new_path, new_widths], axis=1)

    new_track = tph.interp_track.interp_track(new_track, 0.2)

    new_track = Track(new_track)

    return new_track

    # plt.show()


def run_smoothing_process(map_name):
    save_path = "Data/smooth_centre_lines/"
    ensure_path_exists(save_path)
    centre_line = CentreLine(map_name)
    centre_track = np.concatenate([centre_line.path, centre_line.widths], axis=1)
    centre_track = Track(centre_track)

    # crossing = centre_track.check_normals_crossing()
    # if not crossing: print(f"No smoothing needed!!!!!!!!!!!!!!")

    run_n = 0
    while centre_track.check_normals_crossing():

        for width_reduce in np.arange(0.1, np.max(centre_track.widths), 0.1):
            widths = centre_track.widths.copy() - width_reduce
            track = np.concatenate([centre_track.path, widths], axis=1)
            opti_track = Track(track)
            crossing = opti_track.check_normals_crossing()
            if not crossing:
                break

        print(f"Width reduce selected: {width_reduce}")
        new_track = smooth_centre_line2(centre_track, track)

        if not new_track.check_normals_crossing():
            success = True
            txt = f"Width reduce ({width_reduce}) successful --> Minimum widths, L: {np.min(new_track.widths[:, 0]):.2f}, R: {np.min(new_track.widths[:, 1]):.2f}"
        else: 
            success = False
            txt = f"Width reduce ({width_reduce}) FAILED --> Minimum widths, L: {np.min(new_track.widths[:, 0]):.2f}, R: {np.min(new_track.widths[:, 1]):.2f}"

        # smooth_track = np.concatenate([new_track.path, new_track.widths], axis=1)
        # map_c_name = save_path + f"{map_name}_centerline.csv"
        # with open(map_c_name, 'w') as csvfile:
        #     csvwriter = csv.writer(csvfile)
        #     csvwriter.writerows(smooth_track)

        plt.figure(6, figsize=(12, 12))
        plt.clf()
        plt.plot(centre_line.path[:, 0], centre_line.path[:, 1], '-', linewidth=2, color=periwinkle, label="Centre line")
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
        save_path = f"Data/MapSmoothing/"
        ensure_path_exists(save_path)
        plt.savefig(save_path + f"Smoothing_{map_name}_{run_n}.svg")
        run_n += 1

        print("")

        centre_track = new_track


if __name__ == "__main__":
    # generate_racelines()
    # generate_smooth_centre_lines()

    run_smoothing_process("mco")
