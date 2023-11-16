import csv
import numpy as np
from numba import njit
import os
import trajectory_planning_helpers as tph
from scipy.interpolate import splev, splprep


class RaceTrack:
    def __init__(self, map_name, raceline_id=None, load=True) -> None:
        self.map_name = map_name
        self.path = None
        self.speeds = None
        self.psi = None
        self.kappa = None 
        self.s_track = None 

        self.el_lengths = None
        self.nvecs = None

        if load:
            self.load_track(map_name, raceline_id)

    def load_track(self, map_name, raceline_set):
        if raceline_set is None:
            filename = "racelines/" + map_name + "_raceline.csv"
        else:
            filename = f"racelines/{raceline_set}/" + map_name + "_raceline.csv"
        track = np.loadtxt(filename, delimiter=',', skiprows=1)

        self.path = track[:, 1:3]
        self.speeds = track[:, 5] 
        self.psi = track[:, 3]
        self.kappa = track[:, 4]
        self.s_track = track[:, 0]
        self.el_lengths = None

        self.diffs = self.path[1:,:] - self.path[:-1,:]
        self.l2s   = self.diffs[:,0]**2 + self.diffs[:,1]**2

        self.el_lengths = np.linalg.norm(np.diff(self.path, axis=0), axis=1)
        self.s_path = np.insert(np.cumsum(self.el_lengths), 0, 0)
        self.tck = splprep([self.path[:, 0], self.path[:, 1]], k=3, s=0)[0]

    def calculate_s(self, position):
        dists = np.linalg.norm(position - self.path, axis=1)
        t_guess = np.argmin(dists)
        if t_guess != 0 and t_guess <= len(dists) -2 :
            if dists[t_guess+1] < dists[t_guess -1]:
                t_guess += 1
        if t_guess == len(dists) -1:
            t_guess -= 1

        seg_distance = np.linalg.norm(self.path[t_guess+1] - self.path[t_guess])
        interp_ss = np.linspace(0, seg_distance, int(seg_distance*100))
        interp_xs = np.interp(interp_ss, [0, seg_distance], self.path[t_guess:t_guess+2, 0])
        interp_ys = np.interp(interp_ss, [0, seg_distance], self.path[t_guess:t_guess+2, 1])
        interp_points = np.array([interp_xs, interp_ys]).T

        dists = np.linalg.norm(position - interp_points, axis=1)
        s_point = interp_ss[np.argmin(dists)] + self.s_path[t_guess]

        s_normal = s_point / self.s_path[-1]

        return s_normal
    
    def find_nearest_point(self, s):
        point = np.array(splev(s, self.tck, ext=3)).T
        if len(point.shape) > 1: point = point[0]

        return point
    
    
class CentreLine:
    def __init__(self, map_name, directory=f"maps/") -> None:
        self.map_name = map_name
        self.path = None
        self.widths = None

        self.load_track(map_name, directory)

    def load_track(self, map_name, directory):
        filename = directory + map_name + "_centerline.csv"
        track = np.loadtxt(filename, delimiter=',', skiprows=1)

        self.path = track[:, 0:2]
        self.widths = track[:, 2:]

        self.diffs = self.path[1:,:] - self.path[:-1,:]
        self.l2s   = self.diffs[:,0]**2 + self.diffs[:,1]**2

        self.el_lengths = np.linalg.norm(np.diff(self.path, axis=0), axis=1)
        self.s_path = np.insert(np.cumsum(self.el_lengths), 0, 0)
        self.tck = tph.calc_head_curv_num.calc_head_curv_num(self.path, self.el_lengths, False)

    def calculate_s(self, position):
        dists = np.linalg.norm(position - self.path, axis=1)
        t_guess = self.s_path[np.argmin(dists)] / self.s_path[-1]
        if t_guess != 0 and t_guess != len(dists) -1 :
            if dists[t_guess+1] < dists[t_guess -1]:
                t_guess += 1

        seg_distance = np.linalg.norm(self.path[t_guess+1] - self.path[t_guess])
        interp_ss = np.linspace(0, seg_distance, int(seg_distance*100))
        interp_xs = np.interp(interp_ss, self.path[t_guess:t_guess+2, 0], interp_ss)
        interp_ys = np.interp(interp_ss, self.path[t_guess:t_guess+2, 1], interp_ss)
        interp_points = np.array([interp_xs, interp_ys]).T

        dists = np.linalg.norm(position - interp_points, axis=1)
        s_point = interp_ss[np.argmin(dists)] + self.s_path[t_guess]

        return s_point
    
    def find_nearest_point(self, s):
        point = np.array(splev(s, self.tck, ext=3)).T

        return point
    
class CentreLineTrack(CentreLine):
    def __init__(self, map_name, directory="maps/") -> None:
        super().__init__(map_name, directory)
        self.calculate_track_quantities()

    def calculate_track_quantities(self):
        self.el_lengths = np.linalg.norm(np.diff(self.path, axis=0), axis=1)
        self.s_path = np.insert(np.cumsum(self.el_lengths), 0, 0)
        self.psi, self.kappa = tph.calc_head_curv_num.calc_head_curv_num(self.path, self.el_lengths, False)
        self.nvecs = tph.calc_normal_vectors.calc_normal_vectors(self.psi)
        self.tck = tph.calc_head_curv_num.calc_head_curv_num(self.path, self.el_lengths, False)

    def calculate_s(self, position):
        dists = np.linalg.norm(position - self.path, axis=1)
        t_guess = self.s_path[np.argmin(dists)] / self.s_path[-1]
        if t_guess != 0 and t_guess != len(dists) -1 :
            if dists[t_guess+1] < dists[t_guess -1]:
                t_guess += 1

        seg_distance = np.linalg.norm(self.path[t_guess+1] - self.path[t_guess])
        interp_ss = np.linspace(0, seg_distance, int(seg_distance*100))
        interp_xs = np.interp(interp_ss, self.path[t_guess:t_guess+2, 0], interp_ss)
        interp_ys = np.interp(interp_ss, self.path[t_guess:t_guess+2, 1], interp_ss)
        interp_points = np.array([interp_xs, interp_ys]).T

        dists = np.linalg.norm(position - interp_points, axis=1)
        s_point = interp_ss[np.argmin(dists)] + self.s_path[t_guess]

        return s_point
    
    def find_nearest_point(self, s):
        point = np.array(splev(s, self.tck, ext=3)).T

        return point
