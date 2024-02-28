import csv
import numpy as np
from numba import njit
import os
import trajectory_planning_helpers as tph
from scipy.interpolate import splev, splprep


class TrackLine:
    def __init__(self, path) -> None:
        self.path = path
        self.cm_path = None

    def init_path(self):
        self.diffs = self.path[1:, :] - self.path[:-1, :]
        self.l2s = self.diffs[:, 0] ** 2 + self.diffs[:, 1] ** 2

        self.el_lengths = np.linalg.norm(np.diff(self.path, axis=0), axis=1)
        self.s_path = np.insert(np.cumsum(self.el_lengths), 0, 0)
        self.tck = splprep([self.path[:, 0], self.path[:, 1]], k=3, s=0, per=True)[0]

        self.cm_ss = np.linspace(0, self.s_path[-1], int(self.s_path[-1] * 100))
        self.cm_path = np.array(splev(self.cm_ss, self.tck, ext=3)).T

    def init_track(self):
        if self.el_lengths is None:
            self.init_path()
        self.psi, self.kappa = tph.calc_head_curv_num.calc_head_curv_num(
            self.path, self.el_lengths, False
        )
        self.nvecs = tph.calc_normal_vectors.calc_normal_vectors(self.psi)

    def calculate_progress_m(self, position):
        if len(position) > 2:
            position = position[:2]
        dists = np.linalg.norm(position - self.path, axis=1)
        minimum_dist_idx = np.argmin(dists)
        if minimum_dist_idx == 0:
            if dists[-1] < dists[1]:
                return self.s_path[-1]  # current shortcut
            else:
                start_ind = 0
                end_ind = int(self.s_path[1] * 100)
        elif minimum_dist_idx == len(dists) - 1:
            if dists[-2] < dists[0]:
                start_ind = int(self.s_path[-2] * 100)
                end_ind = int(self.s_path[-1] * 100)
            else:
                return self.s_path[-1]
        else:
            if dists[minimum_dist_idx + 1] > dists[minimum_dist_idx - 1]:
                minimum_dist_idx -= 1
            start_ind = int(self.s_path[minimum_dist_idx] * 100)
            end_ind = int(self.s_path[minimum_dist_idx + 1] * 100)

        cm_path = self.cm_path[start_ind:end_ind]
        dists = np.linalg.norm(position - cm_path, axis=1)
        s_point_m = self.cm_ss[np.argmin(dists) + start_ind]

        return s_point_m

    def calculate_progress_percent(self, position):
        progress_m = self.calculate_progress_m(position)
        progress_percent = progress_m / self.s_path[-1]

        return progress_percent

    def find_nearest_point(self, s):
        point = np.array(splev(s, self.tck, ext=3)).T
        if len(point.shape) > 1:
            point = point[0]

        return point

    def calculate_pose(self, s):
        point = np.array(splev(s, self.tck, ext=3)).T
        dx, dy = splev(s, self.tck, der=1, ext=3)
        theta = np.arctan2(dy, dx)
        pose = np.array([point[0], point[1], theta])
        return pose


class CentreLine(TrackLine):
    def __init__(self, map_name, directory=f"maps/") -> None:
        self.map_name = map_name

        self.load_track(map_name, directory)
        self.init_path()
        self.init_track()

    def load_track(self, map_name, directory):
        filename = directory + map_name + "_centerline.csv"
        track = np.loadtxt(filename, delimiter=",", skiprows=1)

        self.path = track[:, 0:2]
        self.widths = track[:, 2:]


class RaceTrack(TrackLine):
    def __init__(self, map_name, raceline_id=None, load=True) -> None:
        self.map_name = map_name

        if load:
            self.load_track(map_name, raceline_id)
            self.init_path()

    def load_track(self, map_name, raceline_set):
        try:
            filename = f"Data/racelines/{raceline_set}/" + map_name + "_raceline.csv"
            track = np.loadtxt(filename, delimiter=",", skiprows=1)
        except:
            filename = f"../Data/racelines/{raceline_set}/" + map_name + "_raceline.csv"
            track = np.loadtxt(filename, delimiter=",", skiprows=1)

        self.path = track[:, 1:3]
        self.speeds = track[:, 5]
        self.psi = track[:, 3]
        self.kappa = track[:, 4]
        self.s_track = track[:, 0]
