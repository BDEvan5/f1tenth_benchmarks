import numpy as np
import matplotlib.pyplot as plt
import trajectory_planning_helpers as tph
np.set_printoptions(precision=4)
from scipy import interpolate, optimize

from f1tenth_sim.localmap_racing.local_map_utils import *

KAPPA_BOUND = 0.8
VEHICLE_WIDTH = 1
ax_max_machine = np.array([[0, 8.5],[8, 8.5]])
ggv = np.array([[0, 8.5, 8.5], [8, 8.5, 8.5]])
MU = 0.7
V_MAX = 8
VEHICLE_MASS = 3.4

# when making imgs
# KAPPA_BOUND = 0.3
# VEHICLE_WIDTH = 0.1
# a_max = 38
# ax_max_machine = np.array([[0, a_max],[8, a_max]])
# ggv = np.array([[0, a_max, a_max], [8, a_max, a_max]])
# MU = 0.4

class LocalRaceline:
    def __init__(self, path, test_id):
        self.lm = None

        self.raceline = None
        self.el_lengths = None
        self.s_track = None
        self.psi = None
        self.kappa = None
        self.vs = None

        self.counter = 0
        self.raceline_data_path = path + f"RacingLineData_{test_id}/"
        ensure_path_exists(self.raceline_data_path)

    def generate_raceline(self, local_map):
        # track = local_map.track
        # track = track[track[:, 0] > 0.4, :] # remove negative values
        # track = np.insert(track, 0, [0, 0, 0.9, 0.9], axis=0) # add start point
        # local_map = LocalMap(track)

        self.lm = local_map

        raceline = self.generate_minimum_curvature_path()
        self.normalise_raceline(raceline)
        self.generate_max_speed_profile()

        raceline = np.concatenate([self.raceline, self.vs[:, None]], axis=-1)
        
        self.tck = interpolate.splprep([self.raceline[:, 0], self.raceline[:, 1]], k=3, s=0)[0]
        # self.tck = interpolate.splprep([self.raceline[:, 0], self.raceline[:, 1], self.vs], k=3, s=0)[0]
        
        # xs = np.linspace(0, 1, 100)
        # new_raceline = np.array(interpolate.splev(xs, self.tck)).T
        # self.raceline = new_raceline[:, :2]
        # self.vs = new_raceline[:, 2]

        np.save(self.raceline_data_path  + f'LocalRaceline_{self.counter}.npy', raceline)
        self.counter += 1

        # return new_raceline
        return raceline

    def generate_minimum_curvature_path(self):
        coeffs_x, coeffs_y, M, normvec_normalized = tph.calc_splines.calc_splines(self.lm.track[:, :2], self.lm.el_lengths, self.lm.psi[0], self.lm.psi[-1])
        psi = self.lm.psi #- np.pi/2 # Why?????

        start_psi = psi[0]
        # print(start_psi)
        # start_psi = -np.pi/2 # vehicle always faces forwards
        # start_psi = (psi[0] - np.pi/2)/2 # average current heading and vehicle direction
        try:
            # alpha, error = tph.opt_min_curv.opt_min_curv(self.lm.track, self.lm.nvecs, M, KAPPA_BOUND, VEHICLE_WIDTH, print_debug=False, closed=False, psi_s=psi[0], psi_e=psi[-1], fix_s=True, fix_e=True)
            alpha, error = tph.opt_min_curv.opt_min_curv(self.lm.track, self.lm.nvecs, M, KAPPA_BOUND, VEHICLE_WIDTH, print_debug=False, closed=False, psi_s=start_psi, psi_e=psi[-1], fix_s=True, fix_e=False)

            raceline = self.lm.track[:, :2] + np.expand_dims(alpha, 1) * self.lm.nvecs
        except Exception as e:
            print("Error in optimising min curvature path")
            print(f"Exception: {e}")
            raceline = self.lm.track[:, :2]

        return raceline

    def normalise_raceline(self, raceline):
        psi = self.lm.psi #- np.pi/2
        self.raceline = raceline 
        self.el_lengths_r = np.linalg.norm(np.diff(self.raceline, axis=0), axis=1)
        self.s_track = np.insert(np.cumsum(self.el_lengths_r), 0, 0)

        self.psi_r, self.kappa_r = tph.calc_head_curv_num.calc_head_curv_num(self.raceline, self.el_lengths_r, False)


    def generate_max_speed_profile(self, starting_speed=V_MAX):
        mu = MU * np.ones_like(self.kappa_r) 

        self.vs = tph.calc_vel_profile.calc_vel_profile(ax_max_machine, self.kappa_r, self.el_lengths_r, False, 0, VEHICLE_MASS, ggv=ggv, mu=mu, v_max=V_MAX, v_start=V_MAX, v_end=V_MAX)
        # self.vs = tph.calc_vel_profile.calc_vel_profile(ax_max_machine, self.kappa_r, self.el_lengths_r, False, 0, VEHICLE_MASS, ggv=ggv, mu=mu, v_max=V_MAX, v_start=starting_speed, v_end=V_MAX)


    def calculate_s(self, point):
        dists = np.linalg.norm(point - self.raceline[:, :2], axis=1)
        t_guess = self.s_track[np.argmin(dists)] / self.s_track[-1]

        t_point = optimize.fmin(dist_to_p, x0=t_guess, args=(self.tck, point), disp=False)
        interp_return = interpolate.splev(t_point, self.tck, ext=3)
        closest_pt = np.array(interp_return).T
        if len(closest_pt.shape) > 1: closest_pt = closest_pt[0]

        return closest_pt, t_point

    def calculate_lookahead_point(self, lookahead_distance):
        track_pt, current_s = self.calculate_s([0, 0])
        lookahead_s = current_s + lookahead_distance / self.s_track[-1]

        lookahead_pt = np.array(interpolate.splev(lookahead_s, self.tck, ext=3)).T
        if len(lookahead_pt.shape) > 1: lookahead_pt = lookahead_pt[0]

        speed = np.interp(current_s, self.s_track/self.s_track[-1], self.vs)[0]

        return lookahead_pt, speed


def normalise_raceline(raceline, step_size, start_psi, end_psi):
    r_el_lengths = np.linalg.norm(np.diff(raceline, axis=0), axis=1)
    
    coeffs_x, coeffs_y, M, normvec_normalized = tph.calc_splines.calc_splines(raceline, r_el_lengths, start_psi, end_psi)
    
    spline_lengths_raceline = tph.calc_spline_lengths.            calc_spline_lengths(coeffs_x=coeffs_x, coeffs_y=coeffs_y)
    
    raceline_interp, spline_inds_raceline_interp, t_values_raceline_interp, s_raceline_interp = tph.            interp_splines.interp_splines(spline_lengths=spline_lengths_raceline,
                                    coeffs_x=coeffs_x,
                                    coeffs_y=coeffs_y,
                                    incl_last_point=False,
                                    stepsize_approx=step_size)
    
    return raceline_interp, s_raceline_interp


