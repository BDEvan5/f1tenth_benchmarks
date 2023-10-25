import numpy as np
from scipy import interpolate
import datetime 
import os

class CenterLine:
    def __init__(self, map_name) -> None:
        center_line = np.loadtxt("maps/" + map_name + "_centerline.csv", delimiter=',')[:, :2]
        el_lengths = np.linalg.norm(np.diff(center_line, axis=0), axis=1)
        old_s_track = np.insert(np.cumsum(el_lengths), 0, 0)
        self.s_track = np.arange(0, old_s_track[-1], 0.01) # cm level resolution
        tck = interpolate.splprep([center_line[:, 0], center_line[:, 1]], u=old_s_track, k=3, s=0)[0]
        self.center_line = np.array(interpolate.splev(self.s_track, tck, ext=3)).T

    def calculate_pose_progress(self, pose):
        dists = np.linalg.norm(pose[:2] - self.center_line[:, :2], axis=1) # last 20 points.
        progress = self.s_track[np.argmin(dists)] / self.s_track[-1]

        return progress

#TODO: change this to use arrays not lists.
class SimulatorHistory:
    def __init__(self, run_name):
        self.path = f"Logs/{run_name}/"
        if os.path.exists(self.path) == False:
            os.mkdir(self.path)

        self.map_name = ""
        self.lap_n = 0
        self.states = []
        self.actions = []

    def set_map_name(self, map_name):
        self.map_name = map_name
    
    def add_memory_entry(self, state, action):
        self.states.append(state)
        self.actions.append(action)
    
    def save_history(self):
        states = np.array(self.states)
        actions = np.array(self.actions)
        lap_history = np.concatenate((states, actions), axis=1)
        
        np.save(self.path + f"SimLog_{self.map_name}_{self.lap_n}.npy", lap_history)

        self.states = []
        self.actions = []
        self.lap_n += 1
        