import numpy as np
import yaml
from f1tenth_benchmarks.utils.track_utils import TrackLine
import casadi as ca


def normalise_psi(psi):
    while psi > np.pi:
        psi -= 2*np.pi
    while psi < -np.pi:
        psi += 2*np.pi
    return psi


def init_track_interpolants(centre_line, exclusion_width):
    widths = np.row_stack((centre_line.widths, centre_line.widths[1:int(centre_line.widths.shape[0] / 2), :]))
    path = np.row_stack((centre_line.path, centre_line.path[1:int(centre_line.path.shape[0] / 2), :]))
    extended_track = TrackLine(path)
    extended_track.init_path()
    extended_track.init_track()

    centre_interpolant = LineInterpolant(extended_track.path, extended_track.s_path, extended_track.psi + np.pi / 2) # add pi/2 to account for coord frame change

    left_path = extended_track.path - extended_track.nvecs * np.clip((widths[:, 0][:, None]  - exclusion_width), 0, np.inf)
    left_interpolant = LineInterpolant(left_path, extended_track.s_path)
    right_path = extended_track.path + extended_track.nvecs * np.clip((widths[:, 1][:, None] - exclusion_width), 0, np.inf)
    right_interpolant = LineInterpolant(right_path, extended_track.s_path)

    return centre_interpolant, left_interpolant, right_interpolant


class LineInterpolant:
    def __init__(self, path, s_path, angles=None):
        self.lut_x = ca.interpolant('lut_x', 'bspline', [s_path], path[:, 0])
        self.lut_y = ca.interpolant('lut_y', 'bspline', [s_path], path[:, 1])
        if angles is not None:
            self.lut_angle = ca.interpolant('lut_angle', 'bspline', [s_path], angles)

    def get_point(self, s):
        return np.array([self.lut_x(s).full()[0, 0], self.lut_y(s).full()[0, 0]])


