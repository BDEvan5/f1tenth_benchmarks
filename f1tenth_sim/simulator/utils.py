import numpy as np
from scipy import interpolate

class CenterLine:
    def __init__(self, map_name) -> None:
        center_line = np.loadtxt("maps/" + map_name + "_centerline.csv", delimiter=',')[:, :2]
        el_lengths = np.linalg.norm(np.diff(center_line, axis=0), axis=1)
        old_s_track = np.insert(np.cumsum(el_lengths), 0, 0)
        self.s_track = np.arange(0, old_s_track[-1], 0.01) # cm level resolution
        self.tck = interpolate.splprep([center_line[:, 0], center_line[:, 1]], u=old_s_track, k=3, s=0)[0]
        self.center_line = np.array(interpolate.splev(self.s_track, self.tck, ext=3)).T

    def calculate_pose_progress(self, pose):
        dists = np.linalg.norm(pose[:2] - self.center_line[:, :2], axis=1) # last 20 points.
        progress = self.s_track[np.argmin(dists)] / self.s_track[-1]

        return progress
    
    def get_pose_from_progress(self, progress):
        s_progress = progress * self.s_track[-1]
        point = np.array(interpolate.splev(s_progress, self.tck, ext=3)).T
        dx, dy = interpolate.splev(s_progress, self.tck, der=1, ext=3)
        theta = np.arctan2(dy, dx)
        pose = np.array([point[0], point[1], theta])
        return pose


class SimulatorHistory:
    def __init__(self, path, test_id, save_scan=False):
        self.path = path + f"RawData_{test_id}/"
        self.save_scan = save_scan

        self.map_name = ""
        self.lap_n = 0
        self.states = []
        self.actions = []
        self.scans = []
        self.progresses = []

    def set_map_name(self, map_name):
        self.map_name = map_name
    
    def add_memory_entry(self, state, action, scan, progress):
        self.states.append(state)
        self.actions.append(action)
        self.scans.append(scan)
        self.progresses.append(progress)
    
    def save_history(self):
        states = np.array(self.states)
        actions = np.array(self.actions)
        progresses = np.array(self.progresses)[:, None]
        lap_history = np.concatenate((states, actions, progresses), axis=1)
        
        np.save(self.path + f"SimLog_{self.map_name}_{self.lap_n}.npy", lap_history)

        if self.save_scan:
            scans = np.array(self.scans)
            np.save(self.path + f"ScanLog_{self.map_name}_{self.lap_n}.npy", scans)

        self.progresses = []
        self.states = []
        self.actions = []
        self.lap_n += 1
        