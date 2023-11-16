import numpy as np 
import cProfile, pstats
import pandas as pd
import trajectory_planning_helpers as tph
from f1tenth_sim.general_utils import *
from f1tenth_sim.classic_racing.planner_utils import CentreLineTrack, RaceTrack
from f1tenth_sim.data_tools.specific_plotting.plot_racelines import RaceTrackPlotter

np.printoptions(precision=3, suppress=True)


class RaceTrackGenerator(RaceTrack):
    def __init__(self, map_name, raceline_id, params) -> None:
        super().__init__(map_name, load=False)
        self.raceline_id = raceline_id
        self.centre_line = CentreLineTrack(map_name, "racelines/")
        ensure_path_exists(f"racelines/{raceline_id}/")
        ensure_path_exists(f"racelines/{raceline_id}_data/")

        self.params = params
        save_params(params, f"racelines/{raceline_id}_data/")
        self.vehicle = load_parameter_file("vehicle_params")
        self.prepare_centre_line()

        self.pr = cProfile.Profile()
        self.pr.enable()

        self.generate_minimum_curvature_path()
        self.generate_velocity_profile()

    def prepare_centre_line(self):
        track = np.concatenate([self.centre_line.path, self.centre_line.widths - self.params.vehicle_width / 2], axis=1)
        crossing = tph.check_normals_crossing.check_normals_crossing(track, self.centre_line.nvecs)
        if crossing: print(f"Major problem: nvecs are crossing. Result will be incorrect. Fix the center line file.")

    def generate_minimum_curvature_path(self):
        coeffs_x, coeffs_y, A, normvec_normalized = tph.calc_splines.calc_splines(self.centre_line.path, self.centre_line.el_lengths, psi_s=self.centre_line.psi[0], psi_e=self.centre_line.psi[-1])

        widths = self.centre_line.widths.copy() - self.params.vehicle_width / 2
        track = np.concatenate([self.centre_line.path, widths], axis=1)
        alpha, error = tph.opt_min_curv.opt_min_curv(track, self.centre_line.nvecs, A, self.params.max_kappa, 0, print_debug=True, closed=False, fix_s=True, psi_s=self.centre_line.psi[0], psi_e=self.centre_line.psi[-1], fix_e=True)

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

        raceline = np.concatenate([self.s_track[:, None], self.path, self.psi[:, None], self.kappa[:, None], self.speeds[:, None], acc[:, None]], axis=1)
        np.savetxt(f"racelines/{self.raceline_id}/"+ self.map_name+ '_raceline.csv', raceline, delimiter=',')

    def __del__(self):
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
        df.to_csv(f"racelines/{self.raceline_id}_data/Profile_{self.map_name}.csv")



if __name__ == "__main__":
    params = load_parameter_file("RaceTrackGenerator")
    params.mu = 0.7
    raceline_id = f"mu{int(params.mu*100)}"
    map_list = ['aut', 'esp', 'gbr', 'mco']
    # map_list = ['mco']
    # map_list = ['aut']
    for map_name in map_list: 
        RaceTrackGenerator(map_name, raceline_id, params)
        RaceTrackPlotter(map_name, raceline_id)
