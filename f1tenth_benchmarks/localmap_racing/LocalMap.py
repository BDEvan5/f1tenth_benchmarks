import numpy as np 
from scipy import interpolate, spatial, optimize
import trajectory_planning_helpers as tph


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

    def check_nvec_crossing(self):
        crossing_horizon = min(5, len(self.track)//2 -1)
        if crossing_horizon < 2: return False
        crossing = tph.check_normals_crossing.check_normals_crossing(self.track, self.nvecs, crossing_horizon)

        return crossing

# @jit(cache=True)
def dist_to_p(t_glob: np.ndarray, path: list, p: np.ndarray):
    s = interpolate.splev(t_glob, path, ext=3)
    s = np.concatenate(s)
    return spatial.distance.euclidean(p, s)



