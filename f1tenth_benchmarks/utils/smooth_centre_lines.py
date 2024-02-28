import numpy as np 
import trajectory_planning_helpers as tph
from f1tenth_benchmarks.utils.BasePlanner import *
from f1tenth_benchmarks.utils.track_utils import CentreLine
from f1tenth_benchmarks.data_tools.specific_plotting.plot_racelines import RaceTrackPlotter
import matplotlib.pyplot as plt

from copy import copy
from f1tenth_benchmarks.data_tools.plotting_utils import *

np.printoptions(precision=3, suppress=True)


track_save_path = "Data/smooth_centre_lines/"
ensure_path_exists(track_save_path)

img_save_path = f"Data/MapSmoothing/"
ensure_path_exists(img_save_path)

class Track:
    def __init__(self, path, widths) -> None:
        self.path = path
        self.widths = widths

        self.calculate_track_vecs()

    def calculate_track_vecs(self):
        self.track = np.concatenate([self.path, self.widths], axis=1)
        self.el_lengths = np.linalg.norm(np.diff(self.path, axis=0), axis=1)
        self.s_path = np.insert(np.cumsum(self.el_lengths), 0, 0)
        self.psi, self.kappa = tph.calc_head_curv_num.calc_head_curv_num(self.path, self.el_lengths, False)
        self.nvecs = tph.calc_normal_vectors.calc_normal_vectors(self.psi)

    def check_normals_crossing(self, widths=None):
        if widths is None:
            track = self.track
        else:
            track = np.concatenate([self.path, widths], axis=1)
        crossing = tph.check_normals_crossing.check_normals_crossing(track, self.nvecs)

        return crossing 

    def smooth_centre_line(self):
        path_cl = np.row_stack([self.path, self.path[0]])
        el_lengths_cl = np.append(self.el_lengths, self.el_lengths[0])
        coeffs_x, coeffs_y, A, normvec_normalized = tph.calc_splines.calc_splines(path_cl, el_lengths_cl)
    
        alpha, error = tph.opt_min_curv.opt_min_curv(self.track, self.nvecs, A, 1, 0, print_debug=True, closed=True)

        self.path, A_raceline, coeffs_x_raceline, coeffs_y_raceline, spline_inds_raceline_interp, t_values_raceline_interp, s_raceline, spline_lengths_raceline, el_lengths_raceline_interp_cl = tph.create_raceline.create_raceline(self.path, self.nvecs, alpha, 0.2) 

        self.widths[:, 0] -= alpha
        self.widths[:, 1] += alpha
        self.widths = tph.interp_track_widths.interp_track_widths(self.widths, spline_inds_raceline_interp, t_values_raceline_interp)

        self.track = np.concatenate([self.path, self.widths], axis=1)
        self.calculate_track_vecs()


WIDTH_STEP_SIZE = 0.1
NUMBER_OF_WIDTH_STEPS = 8

def run_smoothing_process(map_name):
    """
    This assumes that the track width is 0.9 m on each side of the centre line
    """
    centre_line = CentreLine(map_name)
    widths = np.ones_like(centre_line.path) * WIDTH_STEP_SIZE
    track = Track(centre_line.path.copy(), widths)

    for i in range(NUMBER_OF_WIDTH_STEPS):
        track.widths += np.ones_like(track.path) * WIDTH_STEP_SIZE
        crossing = track.check_normals_crossing()
        if crossing:
            raise ValueError("Track is crossing before optimisation.: use smaller step size")

        track.smooth_centre_line()
        plot_map_line(map_name, centre_line, i, track)

        test_widths = track.widths + np.ones_like(track.path) * (NUMBER_OF_WIDTH_STEPS-i) * WIDTH_STEP_SIZE
        crossing = track.check_normals_crossing(test_widths)
        if not crossing:
            print(f"No longer crossing: {i}")
            track.widths = test_widths
            track.calculate_track_vecs()
            plot_map_line(map_name, centre_line, "final", track)
            break

        print("")

    map_c_name = track_save_path + f"{map_name}_centerline.csv"
    with open(map_c_name, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(track.track)
    

def plot_map_line(map_name, centre_line, run_n, new_track):
    plt.figure(6, figsize=(12, 12))
    plt.plot(centre_line.path[:, 0], centre_line.path[:, 1], '-', linewidth=2, color=periwinkle, label="Centre line")
    plt.plot(new_track.path[:, 0], new_track.path[:, 1], '-', linewidth=2, color=sunset_orange, label="Smoothed track")

    plot_line_and_boundaries(centre_line, fresh_t, False)
    plot_line_and_boundaries(new_track, sweedish_green, True)

    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend(["Centre line", "Smoothed track"])
    plt.savefig(img_save_path + f"Smoothing_{map_name}_{run_n}.svg")
    plt.close()

def plot_line_and_boundaries(new_track, color, normals=False):
    l1 = new_track.path + new_track.nvecs * new_track.widths[:, 0][:, None] # inner
    l2 = new_track.path - new_track.nvecs * new_track.widths[:, 1][:, None] # outer

    if normals:
        for i in range(len(new_track.path)):
            plt.plot([l1[i, 0], l2[i, 0]], [l1[i, 1], l2[i, 1]], linewidth=1, color=nartjie)

    plt.plot(l1[:, 0], l1[:, 1], linewidth=1, color=color)
    plt.plot(l2[:, 0], l2[:, 1], linewidth=1, color=color)


import cv2 as cs 
from PIL import Image
from scipy import ndimage

def clip_widths_with_dt(map_name):
    smooth_centre_line = CentreLine(map_name, directory=track_save_path)

    file_name = 'maps/' + map_name + '.yaml'
    with open(file_name) as file:
        documents = yaml.full_load(file)

    yaml_file = dict(documents.items())
    resolution = yaml_file['resolution']
    origin = yaml_file['origin']


    flipped_map_img = np.array(Image.open('maps/' + yaml_file['image']).transpose(Image.FLIP_TOP_BOTTOM))
    dt = ndimage.distance_transform_edt(flipped_map_img)
    np.array(dt *resolution)

    l1 = smooth_centre_line.path + smooth_centre_line.nvecs * smooth_centre_line.widths[:, 0][:, None] 
    for i in range(l1.shape[0]):
        new_width = smooth_centre_line.widths[i, 0]
        new_pt = l1[i]
        while get_dt_value(new_pt, origin, resolution, dt) < 0.01:
            new_width -= 0.01
            new_pt = smooth_centre_line.path[i] + smooth_centre_line.nvecs[i] * new_width
        smooth_centre_line.widths[i, 0] = new_width

    l2 = smooth_centre_line.path - smooth_centre_line.nvecs * smooth_centre_line.widths[:, 1][:, None] 
    for i in range(l2.shape[0]):
        new_width = smooth_centre_line.widths[i, 1]
        new_pt = l2[i]
        while get_dt_value(new_pt, origin, resolution, dt) < 0.05:
            new_width -= 0.01
            new_pt = smooth_centre_line.path[i] - smooth_centre_line.nvecs[i] * new_width
        smooth_centre_line.widths[i, 1] = new_width


    centre_line = CentreLine(map_name)
    plot_map_line(map_name, centre_line, "final2", smooth_centre_line)

    map_c_name = track_save_path + f"{map_name}_centerline.csv"
    track = np.concatenate([smooth_centre_line.path, smooth_centre_line.widths], axis=1)
    with open(map_c_name, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(track)
    

    print(f"Min widths: {np.min(smooth_centre_line.widths, axis=0)}")

def get_dt_value(pt_xy, origin, resolution, dt):
    c = int((pt_xy[0] - origin[0]) / resolution)
    r = int((pt_xy[1] - origin[1]) / resolution)

    if c >= dt.shape[1]:
        c = dt.shape[1] - 1
    if r >= dt.shape[0]:
        r = dt.shape[0] - 1

    distance = dt[r, c] * resolution

    return distance


def smooth_centre_lines():
    run_smoothing_process("aut")
    run_smoothing_process("esp")
    run_smoothing_process("gbr")
    run_smoothing_process("mco")


if __name__ == "__main__":
    # smooth_centre_lines()

    # run_smoothing_process("mco")
    clip_widths_with_dt("mco")
    # clip_widths_with_dt("aut")
    # clip_widths_with_dt("aut")
    # clip_widths_with_dt("aut")


