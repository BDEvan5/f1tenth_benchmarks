import numpy as np
from scipy import interpolate
from scipy import optimize
import matplotlib.pyplot as plt
from LocalMapRacing.local_planning.LocalMapUtils.local_map_utils import *
import trajectory_planning_helpers as tph
from LocalMapRacing.DataTools.plotting_utils import *


POINT_SEP_DISTANCE = 0.8




class TrackBoundary:
    def __init__(self, points, smoothing=False) -> None:        
        self.smoothing_s = 0.2
        # self.smoothing_s = 0.5
        if points[0, 0] > points[-1, 0]:
            self.points = np.flip(points, axis=0)
        else:
            self.points = points

        if smoothing:
            self.apply_smoothing()

        self.el = np.linalg.norm(np.diff(self.points, axis=0), axis=1)
        self.cs = np.insert(np.cumsum(self.el), 0, 0)
        self.total_length = self.cs[-1]

        order_k = min(3, len(self.points) - 1)
        self.tck = interpolate.splprep([self.points[:, 0], self.points[:, 1]], k=order_k, s=0.001)[0]

    def extract_boundary_points(self, sep_distance):
        tck = interpolate.splprep([self.points[:, 0], self.points[:, 1]], u=self.cs, k=3, s=0.001)[0]

        new_cs = np.arange(0, self.cs[-1], sep_distance)
        assert new_cs[-1] < self.cs[-1], "The separation distance is too large for the track length"
        new_points = np.array(interpolate.splev(new_cs, tck)).T

        # new_el = np.linalg.norm(np.diff(new_points, axis=0), axis=1)
        # assert np.abs(np.min(new_el) - sep_distance) < 0.01, "The separation distance is not correct"
        # assert np.abs(np.max(new_el) - sep_distance) < 0.01, "The separation distance is not correct"
        # assert np.allclose(new_el, sep_distance, rtol=0.02), "The separation distance is not correct"

        return new_points


    def find_closest_point_old(self, pt, previous_maximum, string=""):
        dists = np.linalg.norm(self.points - pt, axis=1)
        closest_ind = np.argmin(dists)
        t_guess = self.cs[closest_ind] / self.cs[-1]

        closest_t = optimize.fmin(dist_to_p, x0=t_guess, args=(self.tck, pt), disp=False, xtol=1e-4, ftol=1e-4)
        if closest_t < 0:
            return self.points[0], closest_t
        t_pt = max(closest_t, previous_maximum)

        interp_return = interpolate.splev(t_pt, self.tck)
        closest_pt = np.array(interp_return).T
        if len(closest_pt.shape) > 1:
            closest_pt = closest_pt[0]

        return closest_pt, t_pt
    
    def find_closest_point_true(self, pt, previous_maximum, _s=None):
        dists = np.linalg.norm(self.points - pt, axis=1)
        closest_ind = np.argmin(dists)
        t_guess = self.cs[closest_ind] / self.cs[-1]

        closest_t = optimize.fmin(dist_to_p, x0=t_guess, args=(self.tck, pt), disp=False, xtol=1e-4, ftol=1e-4)
        # closest_t = optimize.fmin_powell(dist_to_p, x0=t_guess, args=(self.tck, pt), disp=False, xtol=1e-4, ftol=1e-4)
        if closest_t < 0:
            return self.points[0], closest_t
        t_pt = max(closest_t, previous_maximum)

        interp_return = interpolate.splev(t_pt, self.tck)
        closest_pt = np.array(interp_return).T
        if len(closest_pt.shape) > 1:
            closest_pt = closest_pt[0]

        return closest_pt, t_pt
    
    def find_closest_point(self, pt, previous_maximum, string=""):
        dists = np.linalg.norm(self.points - pt, axis=1)
        closest_ind = np.argmin(dists)
        if closest_ind == 0:
            start = 0
            end = 1
        elif closest_ind == len(self.points) - 1:
            start = len(self.points) - 2
            end = len(self.points) - 1
        elif dists[closest_ind-1] < dists[closest_ind + 1]:
            start = closest_ind - 1
            end = closest_ind
        else:
            start = closest_ind
            end = closest_ind + 1

        d_ss = self.cs[end] - self.cs[start]
        d1, d2 = dists[start], dists[end]

        if d1 < 0.01: # at the first point
            x = 0   
        elif d2 < 0.01: # at the second point
            x = dists[start] # the distance to the previous point
        else: 
            # if the point is somewhere along the line
            s = (d_ss + d1 + d2)/2
            Area_square = (s*(s-d1)*(s-d2)*(s-d_ss))
            if Area_square < 0:
                print(f"Error with square")
            Area = Area_square**0.5
            h = Area * 2/d_ss
            if np.isnan(h):
                print(f"Error with h")
                h = 0

            x = (d1**2 - h**2)**0.5
            x = min(x, d_ss)

        if np.isnan(x):
            print(f"Error with x")
            x = 0

        t_pt = (self.cs[start] + x) / self.total_length

        t_pt = min(t_pt, 1)
        t_pt = max(t_pt, previous_maximum)
        # t_pt = np.array([t_pt])
        closest_pt = np.array(interpolate.splev(t_pt, self.tck)).T

        # t_guess = self.cs[closest_ind] / self.cs[-1]

        # closest_t = optimize.fmin(dist_to_p, x0=t_pt, args=(self.tck, pt), disp=False, xtol=1e-4, ftol=1e-4)
        # diff = closest_t - t_pt
        # if abs(diff) > 0.01:
        #     print(f"Point: {pt} - prev max: {previous_maximum}]")
        #     print(f"Old: {closest_t}, New: {t_pt}, diff: {closest_t - t_pt}")

        # t_pt = max(closest_t, previous_maximum)

        # interp_return = interpolate.splev(t_pt, self.tck)
        # closest_pt = np.array(interp_return).T
        # if len(closest_pt.shape) > 1:
        #     closest_pt = closest_pt[0]

        return closest_pt, t_pt
    
    def extract_line_portion(self, s_array):
        assert np.min(s_array) >= 0, "S must be positive"
        assert  np.max(s_array) <= 1, "S must be < 1"
        point_set = np.array(interpolate.splev(s_array, self.tck)).T

        return point_set
        
    def plot_line(self):
        plt.plot(self.points[:, 0], self.points[:, 1], '.', markersize=10, color="#20bf6b")
        plt.plot(self.points[0, 0], self.points[0, 1], "o", color='#20bf6b', markersize=15)
        pts_interp = np.array(interpolate.splev(np.linspace(0, 1, 500), self.tck)).T
        plt.plot(pts_interp[:, 0], pts_interp[:, 1], color="#20bf6b")  

    def plot_line2(self):
        # plt.plot(self.points[:, 0], self.points[:, 1], '.', markersize=10, color="#20bf6b")
        # plt.plot(self.points[0, 0], self.points[0, 1], "o", color='#20bf6b', markersize=15)
        pts_interp = np.array(interpolate.splev(np.linspace(0, 1, 500), self.tck)).T
        plt.plot(pts_interp[:, 0], pts_interp[:, 1], color="#20bf6b")

    def plot_line4(self):
        # plt.plot(self.points[:, 0], self.points[:, 1], '.', markersize=10, color="#20bf6b")
        # plt.plot(self.points[0, 0], self.points[0, 1], "o", color='#20bf6b', markersize=15)
        pts_interp = np.array(interpolate.splev(np.linspace(0, 1, 500), self.tck)).T
        plt.plot(pts_interp[:, 0], pts_interp[:, 1], color="#20bf6b", linewidth=4)

    def plot_line3(self):
        # plt.plot(self.points[:, 0], self.points[:, 1], '.', markersize=10, color="#20bf6b")
        # plt.plot(self.points[0, 0], self.points[0, 1], "o", color='#20bf6b', markersize=15)
        pts_interp = np.array(interpolate.splev(np.linspace(0, 1, 500), self.tck)).T
        plt.plot(pts_interp[:, 0], pts_interp[:, 1], color=minty_green, linewidth=2)

    def apply_smoothing(self):
        line_length = np.sum(np.linalg.norm(np.diff(self.points, axis=0), axis=1))
        n_pts = max(int(line_length / POINT_SEP_DISTANCE), 2)
        smooth_line = interpolate_track_new(self.points, None, self.smoothing_s)
        self.points = interpolate_track_new(smooth_line, n_pts*2, 0)
        #NOTE: the double interpolation ensures that the lengths are correct.
        # the first step smooths the points and the second step ensures the correct spacing.
        # TODO: this could be achieved using the same tck and just recalculating the s values based on the new lengths. Do this for computational speedup.


class LocalLine:
    def __init__(self, track):
        self.track = track
        self.el_lengths = None
        self.psi = None
        self.kappa = None
        self.nvecs = None
        self.s_track = None

        self.calculate_length_heading_nvecs()

    def calculate_length_heading_nvecs(self):
        self.el_lengths = np.linalg.norm(np.diff(self.track[:, :2], axis=0), axis=1)
        self.s_track = np.insert(np.cumsum(self.el_lengths), 0, 0)
        self.psi, self.kappa = tph.calc_head_curv_num.calc_head_curv_num(self.track, self.el_lengths, False)
        self.nvecs = tph.calc_normal_vectors_ahead.calc_normal_vectors_ahead(self.psi)
        # self.nvecs = tph.calc_normal_vectors_ahead.calc_normal_vectors_ahead(self.psi+np.pi/2)



