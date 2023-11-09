import numpy as np
import trajectory_planning_helpers as tph
import matplotlib.pyplot as plt
from f1tenth_sim.localmap_racing.local_map_utils import *

from scipy import interpolate
from scipy import optimize
from scipy import spatial


class LocalMap:
    def __init__(self, track):
        self.track = track
        self.el_lengths = None
        self.psi = None
        self.kappa = None
        self.nvecs = None
        self.s_track = None

        if len(self.track) > 3:
            self.tck = interpolate.splprep([self.track[:, 0], self.track[:, 1]], k=3, s=0)[0]
        else:
            self.tck = None

        self.calculate_length_heading_nvecs()

    def calculate_length_heading_nvecs(self):
        self.el_lengths = np.linalg.norm(np.diff(self.track[:, :2], axis=0), axis=1)
        self.s_track = np.insert(np.cumsum(self.el_lengths), 0, 0)
        self.psi, self.kappa = tph.calc_head_curv_num.calc_head_curv_num(self.track, self.el_lengths, False)
        self.nvecs = tph.calc_normal_vectors_ahead.calc_normal_vectors_ahead(self.psi)
        # self.nvecs = tph.calc_normal_vectors_ahead.calc_normal_vectors_ahead(self.psi-np.pi/2)

    def calculate_s(self, point):
        if self.tck == None:
            return point, 0
        dists = np.linalg.norm(point - self.track[:, :2], axis=1)
        t_guess = self.s_track[np.argmin(dists)] / self.s_track[-1]

        t_point = optimize.fmin(dist_to_p, x0=t_guess, args=(self.tck, point), disp=False)
        interp_return = interpolate.splev(t_point, self.tck, ext=3)
        closest_pt = np.array(interp_return).T
        if len(closest_pt.shape) > 1: closest_pt = closest_pt[0]

        return closest_pt, t_point
    
    def interpolate_track_scipy(self, n_pts=None, s=0):
        ws = np.ones_like(self.track[:, 0])
        ws[0:2] = 100
        ws[-2:] = 100
        tck = interpolate.splprep([self.track[:, 0], self.track[:, 1]], w=ws, k=3, s=s)[0]
        if n_pts is None: n_pts = len(self.track)
        self.track[:, :2] = np.array(interpolate.splev(np.linspace(0, 1, n_pts), tck)).T

        self.calculate_length_heading_nvecs()

    def check_nvec_crossing(self):
        crossing_horizon = min(5, len(self.track)//2 -1)
        if crossing_horizon < 2: return False
        crossing = tph.check_normals_crossing.check_normals_crossing(self.track, self.nvecs, crossing_horizon)

        return crossing



class PlotLocalMap(LocalMap):
    def __init__(self, track):
        super().__init__(track)

        # self.local_map_img_path = self.path + "LocalMapImgs/"
        # ensure_path_exists(self.local_map_img_path)
    
    def plot_local_map(self, save_path=None, counter=0, xs=None, ys=None):
        l1 = self.track[:, :2] + self.nvecs * self.track[:, 2][:, None]
        l2 = self.track[:, :2] - self.nvecs * self.track[:, 3][:, None]

        plt.figure(1)
        plt.clf()
        if xs is not None and ys is not None:
            plt.plot(xs, ys, '.', color='#0057e7', alpha=0.1)
        plt.plot(self.track[:, 0], self.track[:, 1], '-', color='#E74C3C', label="Center", linewidth=3)
        plt.plot(0, 0, 'x', color='black', markersize=10)

        plt.plot(l1[:, 0], l1[:, 1], color='#ffa700')
        plt.plot(l2[:, 0], l2[:, 1], color='#ffa700')

        for i in range(len(self.track)):
            xs = [l1[i, 0], l2[i, 0]]
            ys = [l1[i, 1], l2[i, 1]]
            plt.plot(xs, ys, '#ffa700')

        plt.title("Local Map")
        plt.gca().set_aspect('equal', adjustable='box')

        if save_path is not None:
            plt.savefig(save_path + f"Local_map_std_{counter}.svg")

    def plot_local_map_offset(self, offset_pos, offset_theta, origin, resolution, save_path=None, counter=0):
        l1 = self.track[:, :2] + self.nvecs * self.track[:, 2][:, None]
        l2 = self.track[:, :2] - self.nvecs * self.track[:, 3][:, None]

        rotation = np.array([[np.cos(offset_theta), -np.sin(offset_theta)],
                                [np.sin(offset_theta), np.cos(offset_theta)]])
        
        l1 = np.matmul(rotation, l1.T).T
        l2 = np.matmul(rotation, l2.T).T

        l1 = l1 + offset_pos
        l2 = l2 + offset_pos

        l1 = (l1 - origin) / resolution
        l2 = (l2 - origin) / resolution

        track = np.matmul(rotation, self.track[:, :2].T).T
        track = track + offset_pos
        track = (track - origin) / resolution

        position = (offset_pos - origin) / resolution

        # plt.figure(1)
        # plt.clf()
        plt.plot(track[:, 0], track[:, 1], '--', color='#E74C3C', label="Center", linewidth=2)
        plt.plot(track[0, 0], track[0, 1], '*', color='#E74C3C', markersize=10)

        plt.plot(l1[:, 0], l1[:, 1], color='#ffa700')
        plt.plot(l2[:, 0], l2[:, 1], color='#ffa700')

        plt.gca().set_aspect('equal', adjustable='box')

        buffer = 50
        xlim = [np.min(track[:, 0]) - buffer, np.max(track[:, 0]) + buffer]
        ylim = [np.min(track[:, 1]) - buffer, np.max(track[:, 1]) + buffer]
        plt.xlim(xlim)
        plt.ylim(ylim)

        if save_path is not None:
            plt.savefig(save_path + f"Local_map_{counter}.svg")

        return xlim, ylim

    def plot_smoothing(self, old_track, old_nvecs, counter, path):
        plt.figure(2)
        plt.clf()
        plt.plot(old_track[:, 0], old_track[:, 1], 'r', linewidth=2)
        l1 = old_track[:, :2] + old_track[:, 2][:, None] * old_nvecs
        l2 = old_track[:, :2] - old_track[:, 3][:, None] * old_nvecs
        plt.plot(l1[:, 0], l1[:, 1], 'r', linestyle='--', linewidth=1)
        plt.plot(l2[:, 0], l2[:, 1], 'r', linestyle='--', linewidth=1)
        for z in range(len(old_track)):
            xs = [l1[z, 0], l2[z, 0]]
            ys = [l1[z, 1], l2[z, 1]]
            plt.plot(xs, ys, color='orange', linewidth=1)

        plt.plot(self.track[:, 0], self.track[:, 1], 'b', linewidth=2)
        l1 = self.track[:, :2] + self.track[:, 2][:, None] * self.nvecs
        l2 = self.track[:, :2] - self.track[:, 3][:, None] * self.nvecs
        plt.plot(l1[:, 0], l1[:, 1], 'b', linestyle='--', linewidth=1)
        plt.plot(l2[:, 0], l2[:, 1], 'b', linestyle='--', linewidth=1)
        for z in range(len(self.track)):
            xs = [l1[z, 0], l2[z, 0]]
            ys = [l1[z, 1], l2[z, 1]]
            plt.plot(xs, ys, color='green', linewidth=1)

        plt.axis('equal')
        # plt.show()
        # plt.pause(0.0001)
        plt.savefig(path + f"Smoothing_{counter}.svg")


