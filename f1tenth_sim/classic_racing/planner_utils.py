import csv
import numpy as np
from numba import njit
import os
import trajectory_planning_helpers as tph


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

    
class CentreLineTrack(CentreLine):
    def __init__(self, map_name, directory="maps/") -> None:
        super().__init__(map_name, directory)

        self.el_lengths = np.linalg.norm(np.diff(self.path, axis=0), axis=1)
        self.s_path = np.insert(np.cumsum(self.el_lengths), 0, 0)
        self.psi, self.kappa = tph.calc_head_curv_num.calc_head_curv_num(self.path, self.el_lengths, False)
        self.nvecs = tph.calc_normal_vectors.calc_normal_vectors(self.psi)

