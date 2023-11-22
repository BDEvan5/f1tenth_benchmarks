import numpy as np 
import cProfile, pstats
import pandas as pd
import trajectory_planning_helpers as tph
from f1tenth_sim.utils.BasePlanner import *
from f1tenth_sim.utils.track_utils import CentreLine, RaceTrack
from f1tenth_sim.data_tools.specific_plotting.plot_racelines import RaceTrackPlotter
import matplotlib.pyplot as plt
from f1tenth_sim.utils.smooth_centre_lines import smooth_centre_lines
import csv

from copy import copy
from f1tenth_sim.data_tools.plotting_utils import *


np.printoptions(precision=3, suppress=True)


class RaceTrackGenerator(RaceTrack):
    def __init__(self, map_name, raceline_id, params) -> None:
        super().__init__(map_name, load=False)
        self.raceline_id = raceline_id
        try:
            self.centre_line = CentreLine(map_name, "Data/smooth_centre_lines/")
        except:
            smooth_centre_lines()
            self.centre_line = CentreLine(map_name, "Data/smooth_centre_lines/")
        ensure_path_exists(f"Data/racelines/")
        ensure_path_exists(f"Data/raceline_data/")
        self.raceline_path = f"Data/racelines/{raceline_id}/"
        ensure_path_exists(self.raceline_path)
        self.raceline_data_path = f"Data/raceline_data/{raceline_id}/"
        ensure_path_exists(self.raceline_data_path)

        self.params = params
        save_params(params, self.raceline_data_path)
        self.vehicle = load_parameter_file("vehicle_params")
        self.prepare_centre_line()

        self.pr = cProfile.Profile()
        self.pr.enable()

        self.generate_minimum_curvature_path()
        self.generate_velocity_profile()
        self.save_raceline()

    def prepare_centre_line(self):
        track = np.concatenate([self.centre_line.path, self.centre_line.widths - self.params.vehicle_width / 2], axis=1)
        crossing = tph.check_normals_crossing.check_normals_crossing(track, self.centre_line.nvecs)
        if crossing: print(f"Major problem: nvecs are crossing. Result will be incorrect. Fix the center line file.")

    def generate_minimum_curvature_path(self):
        path_cl = np.row_stack([self.centre_line.path, self.centre_line.path[0]])
        el_lengths_cl = np.append(self.centre_line.el_lengths, self.centre_line.el_lengths[0])
        coeffs_x, coeffs_y, A, normvec_normalized = tph.calc_splines.calc_splines(path_cl, el_lengths_cl)
    
        widths = self.centre_line.widths.copy() - self.params.vehicle_width / 2
        track = np.concatenate([self.centre_line.path, widths], axis=1)
        alpha, error = tph.opt_min_curv.opt_min_curv(track, self.centre_line.nvecs, A, 1, 0, print_debug=True, closed=True)

        self.path, A_raceline, coeffs_x_raceline, coeffs_y_raceline, spline_inds_raceline_interp, t_values_raceline_interp, self.s_raceline, spline_lengths_raceline, el_lengths_raceline_interp_cl = tph.create_raceline.create_raceline(self.centre_line.path, self.centre_line.nvecs, alpha, self.params.raceline_step) 
        self.psi, self.kappa = tph.calc_head_curv_num.calc_head_curv_num(self.path, el_lengths_raceline_interp_cl, True)

    def generate_velocity_profile(self):
        mu = self.params.mu * np.ones(len(self.path))
        self.el_lengths = np.linalg.norm(np.diff(self.path, axis=0), axis=1)

        ggv = np.array([[0, self.params.max_longitudinal_acc, self.params.max_lateral_acc], 
                        [self.vehicle.max_speed, self.params.max_longitudinal_acc, self.params.max_lateral_acc]])
        ax_max_machine = np.array([[0, self.params.max_longitudinal_acc],
                                   [self.vehicle.max_speed, self.params.max_longitudinal_acc]])

        self.speeds = tph.calc_vel_profile.calc_vel_profile(ax_max_machine, self.kappa, self.el_lengths, False, 0, self.vehicle.vehicle_mass, ggv=ggv, mu=mu, v_max=self.vehicle.max_speed, v_start=self.vehicle.max_speed)

        ts = tph.calc_t_profile.calc_t_profile(self.speeds, self.el_lengths, 0)
        print(f"Planned Lap Time: {ts[-1]}")

    def save_raceline(self):
        acc = tph.calc_ax_profile.calc_ax_profile(self.speeds, self.el_lengths, True)

        self.s_track = np.insert(np.cumsum(self.el_lengths), 0, 0)
        raceline = np.concatenate([self.s_track[:, None], self.path, self.psi[:, None], self.kappa[:, None], self.speeds[:, None], acc[:, None]], axis=1)
        np.savetxt(self.raceline_path + self.map_name+ '_raceline.csv', raceline, delimiter=',')

    def __del__(self):
        try:
            self.pr.disable()
            ps = pstats.Stats(self.pr).sort_stats('cumulative')
            stats_profile_functions = ps.get_stats_profile().func_profiles
            df_entries = []
            for k in stats_profile_functions.keys():
                v = stats_profile_functions[k]
                entry = {"func": k, "ncalls": v.ncalls, "tottime": v.tottime, "percall_tottime": v.percall_tottime, "cumtime": v.cumtime, "percall_cumtime": v.percall_cumtime, "file_name": v.file_name, "line_number": v.line_number}
                df_entries.append(entry)
            df = pd.DataFrame(df_entries)
            df = df[df.cumtime > 0]
            df = df[df.file_name != "~"] # this removes internatl file calls.
            df = df[~df['file_name'].str.startswith('<')]
            df = df.sort_values(by=['cumtime'], ascending=False)
            df.to_csv(self.raceline_data_path + f"Profile_{self.map_name}.csv")
        except Exception as e:
            pass
    
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



def generate_racelines():
    params = load_parameter_file("RaceTrackGenerator")
    # params.mu = 0.5
    params.mu = 0.6
    # raceline_id = f"_drl_training"
    raceline_id = f"mu{int(params.mu*100)}"
    map_list = ['aut', 'esp', 'gbr', 'mco']
    map_list = ['mco']
    # map_list = ['aut']
    for map_name in map_list: 
        RaceTrackGenerator(map_name, raceline_id, params)
        RaceTrackPlotter(map_name, raceline_id)



if __name__ == "__main__":
    generate_racelines()
    # generate_smooth_centre_lines()


