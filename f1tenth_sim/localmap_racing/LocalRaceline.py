import numpy as np
import matplotlib.pyplot as plt
import trajectory_planning_helpers as tph
np.set_printoptions(precision=4)
from scipy import interpolate, optimize

from f1tenth_sim.localmap_racing.local_map_utils import *
from f1tenth_sim.localmap_racing.local_opt_min_curv import local_opt_min_curv

KAPPA_BOUND = 0.8
VEHICLE_WIDTH = 1
ax_max_machine = np.array([[0, 8.5],[8, 8.5]])
ggv = np.array([[0, 8.5, 8.5], [8, 8.5, 8.5]])
MU = 0.6
V_MAX = 8
VEHICLE_MASS = 3.4

# when making imgs
# KAPPA_BOUND = 0.3
# VEHICLE_WIDTH = 0.1
# a_max = 38
# ax_max_machine = np.array([[0, a_max],[8, a_max]])
# ggv = np.array([[0, a_max, a_max], [8, a_max, a_max]])
# MU = 0.4

#!TODO: this must be moved to a helper file that sets all these things up.
#It must auto setup if a new device is used
class MatrixAs:
    def __init__(self):
        self.start = 5
        self.matrxs = []
        for i in range(self.start, 80):
            A = np.load(f"Logs/Data_A/A_{i}.npy")
            self.matrxs.append(A)

    def get_A(self, i):
        return self.matrxs[i - self.start]

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
        self.mtrxs = MatrixAs()

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
        # plt.figure()
        # plt.plot(self.lm.track[:, 0], self.lm.track[:, 1], '--', linewidth=2, color='black')
        # # calculate the sides using the nvecs
        # l1 = self.lm.track[:, :2] + self.lm.nvecs * self.lm.track[:, 2][:, None]
        # l2 = self.lm.track[:, :2] - self.lm.nvecs * self.lm.track[:, 3][:, None]
        # plt.plot(l1[:, 0], l1[:, 1], color='green')
        # plt.plot(l2[:, 0], l2[:, 1], color='green')
        # for i in range(len(l1)):
        #     xs = [l1[i, 0], l2[i, 0]]
        #     ys = [l1[i, 1], l2[i, 1]]
        #     plt.plot(xs, ys, color='green')
        # plt.plot(0, 0, '*', markersize=10, color='red')
        # plt.axis('equal')

        # plt.show()

        adjusted_track = self.lm.track
        adjusted_track[:, 2] -= 0.5
        adjusted_track[:, 3] -= 0.5

        M = self.mtrxs.get_A(len(self.lm.track))
        # M = tph.calc_spline_M.calc_spline_M(self.lm.track[:, :2], self.lm.el_lengths, self.lm.psi[0] + np.pi/2, self.lm.psi[-1] + np.pi/2, use_dist_scaling=True)

        # M = tph.calc_spline_M.calc_spline_M(self.lm.track[:, :2], self.lm.el_lengths, self.lm.psi[0] + np.pi/2, self.lm.psi[-1] + np.pi/2, use_dist_scaling=False)
        # coeffs_x, coeffs_y, M, normvec_normalized = tph.calc_splines.calc_splines(self.lm.track[:, :2], self.lm.el_lengths, self.lm.psi[0] + np.pi/2, self.lm.psi[-1] + np.pi/2)
        psi = self.lm.psi #- np.pi/2 # Why?????

        start_psi = psi[0]
        # print(start_psi)
        # start_psi = -np.pi/2 # vehicle always faces forwards
        # start_psi = (psi[0] - np.pi/2)/2 # average current heading and vehicle direction
        try:
            # alpha, error = tph.opt_min_curv.opt_min_curv(self.lm.track, self.lm.nvecs, M, KAPPA_BOUND, VEHICLE_WIDTH, print_debug=False, closed=False, psi_s=psi[0], psi_e=psi[-1], fix_s=True, fix_e=True)
            alpha, error = local_opt_min_curv(adjusted_track, self.lm.nvecs, M, KAPPA_BOUND, 0, print_debug=False, psi_s=start_psi, psi_e=psi[-1], fix_s=True, fix_e=False)
            # alpha, error = tph.opt_min_curv.opt_min_curv(self.lm.track, self.lm.nvecs, M, KAPPA_BOUND, VEHICLE_WIDTH, print_debug=False, closed=False, psi_s=start_psi, psi_e=psi[-1], fix_s=True, fix_e=False)

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
        new_points = np.linspace(0, self.s_track[-1], int(self.s_track[-1]*100)) #cm accuracy
        xs, ys = interp_2d_points(new_points, self.s_track, self.raceline)
        raceline = np.concatenate([xs[:, None], ys[:, None]], axis=-1)
        dists = np.linalg.norm(point - raceline, axis=1)
        t_new = (new_points[np.argmin(dists)] / self.s_track[-1])

        interp_return = interpolate.splev(t_new, self.tck, ext=3)
        closest_pt = np.array(interp_return).T
        if len(closest_pt.shape) > 1: closest_pt = closest_pt[0]

        return closest_pt, [t_new]

    def calculate_zero_point_progress(self):
        n_pts = np.count_nonzero(self.s_track < 5) # search first 4 m
        s_track = self.s_track[:n_pts]
        raceline = self.raceline[:n_pts]
        new_points = np.linspace(0, s_track[-1], int(s_track[-1]*100)) #cm accuracy
        xs, ys = interp_2d_points(new_points, s_track, raceline)
        raceline = np.concatenate([xs[:, None], ys[:, None]], axis=-1)
        dists = np.linalg.norm(raceline, axis=1)
        t_new = (new_points[np.argmin(dists)] / self.s_track[-1])

        return [t_new]

    def calculate_lookahead_point(self, lookahead_distance):
        current_s = self.calculate_zero_point_progress()
        lookahead_s = current_s + lookahead_distance / self.s_track[-1]

        lookahead_pt = np.array(interpolate.splev(lookahead_s, self.tck, ext=3)).T
        if len(lookahead_pt.shape) > 1: lookahead_pt = lookahead_pt[0]

        speed = np.interp(current_s, self.s_track/self.s_track[-1], self.vs)[0]

        return lookahead_pt, speed


def normalise_raceline(raceline, step_size, start_psi, end_psi):
    r_el_lengths = np.linalg.norm(np.diff(raceline, axis=0), axis=1)
    
    coeffs_x, coeffs_y, M, normvec_normalized = tph.calc_splines.calc_splines(raceline, r_el_lengths, start_psi, end_psi)
    
    spline_lengths_raceline = tph.calc_spline_lengths.calc_spline_lengths(coeffs_x=coeffs_x, coeffs_y=coeffs_y)
    
    raceline_interp, spline_inds_raceline_interp, t_values_raceline_interp, s_raceline_interp = tph.interp_splines.interp_splines(spline_lengths=spline_lengths_raceline,
                                    coeffs_x=coeffs_x,
                                    coeffs_y=coeffs_y,
                                    incl_last_point=False,
                                    stepsize_approx=step_size)
    
    return raceline_interp, s_raceline_interp


