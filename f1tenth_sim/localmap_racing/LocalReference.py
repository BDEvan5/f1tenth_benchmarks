import numpy as np
import matplotlib.pyplot as plt
import casadi as ca
import trajectory_planning_helpers as tph
import csv



class LocalReference:
    def __init__(self, local_map):
        self.path = local_map.track[:, :2]
        self.track = local_map.track
        self.el_lengths = local_map.el_lengths 
        self.s_track = local_map.s_track
        self.psi = local_map.psi
        self.nvecs = local_map.nvecs
        self.track_length = local_map.s_track[-1]

        self.center_lut_x, self.center_lut_y = None, None
        self.left_lut_x, self.left_lut_y = None, None
        self.right_lut_x, self.right_lut_y = None, None

        self.center_lut_x, self.center_lut_y = self.get_interpolated_path_casadi('lut_center_x', 'lut_center_y', self.path, self.s_track)
        self.angle_lut_t = self.get_interpolated_heading_casadi('lut_angle_t', self.psi, self.s_track)

        w = 0.35
        left_path = self.path - self.nvecs * (self.track[:, 2][:, None]  - w)
        right_path = self.path + self.nvecs * (self.track[:, 3][:, None] - w)
        self.left_lut_x, self.left_lut_y = self.get_interpolated_path_casadi('lut_left_x', 'lut_left_y', left_path, self.s_track)
        self.right_lut_x, self.right_lut_y = self.get_interpolated_path_casadi('lut_right_x', 'lut_right_y', right_path, self.s_track)

    def get_interpolated_path_casadi(self, label_x, label_y, pts, arc_lengths_arr):
        u = arc_lengths_arr
        V_X = pts[:, 0]
        V_Y = pts[:, 1]
        lut_x = ca.interpolant(label_x, 'bspline', [u], V_X)
        lut_y = ca.interpolant(label_y, 'bspline', [u], V_Y)
        return lut_x, lut_y
    
    def get_interpolated_heading_casadi(self, label, pts, arc_lengths_arr):
        u = arc_lengths_arr
        V = pts
        lut = ca.interpolant(label, 'bspline', [u], V)
        return lut
    
    def calculate_s(self, point):
        distances = np.linalg.norm(self.path - point, axis=1)
        idx = np.argmin(distances)
        x, h = self.interp_pts(idx, distances)
        s = (self.s_track[idx] + x) 

        return s

    def interp_pts(self, idx, dists):
        if idx == len(dists) -1: 
            return dists[idx], 0
        d_ss = self.s_track[idx+1] - self.s_track[idx]
        d1, d2 = dists[idx], dists[idx+1]

        if d1 < 0.01: # at the first point
            x = 0   
            h = 0
        elif d2 < 0.01: # at the second point
            x = dists[idx] # the distance to the previous point
            h = 0 # there is no distance
        else:     # if the point is somewhere along the line
            s = (d_ss + d1 + d2)/2
            Area_square = (s*(s-d1)*(s-d2)*(s-d_ss))
            if Area_square < 0:  # negative due to floating point precision
                h = 0
                x = d_ss + d1
            else:
                Area = Area_square**0.5
                h = Area * 2/d_ss
                x = (d1**2 - h**2)**0.5

        return x, h


    def plot_path(self):
        plt.figure(2)
        plt.clf()

        plt.plot(self.center_lut_x(self.s_track), self.center_lut_y(self.s_track), label="center", color='blue', alpha=0.7)
        plt.plot(self.left_lut_x(self.s_track), self.left_lut_y(self.s_track), label="left", color='green', alpha=0.7)
        plt.plot(self.right_lut_x(self.s_track), self.right_lut_y(self.s_track), label="right", color='green', alpha=0.7)

        # plt.show()


    def plot_angles(self):
        plt.figure(3)
        plt.clf()

        plt.plot(self.s_track, self.angle_lut_t(self.s_track), label="Fixed angles", color='blue')



if __name__ == "__main__":
    path = LocalReference("aut")
    path.plot_path()
    path.plot_angles()

    print(path.center_lut_x(120))
    plt.show()
