import numpy as np
import os
import numpy as np 
from scipy import interpolate, spatial, optimize
import trajectory_planning_helpers as tph

np.set_printoptions(precision=4)

DISTNACE_THRESHOLD = 1.4 # distance in m for an exception
TRACK_WIDTH = 1.8 # use fixed width
FOV = 4.7
BOUNDARY_SMOOTHING = 0.2
MAX_TRACK_WIDTH = 2.5
TRACK_SEPEARTION_DISTANCE = 0.4
BOUNDARY_STEP_SIZE = 0.4
# FILTER_THRESHOLD = 2.4

class LocalMapGenerator:
    def __init__(self, path, test_id, save_data) -> None:
        self.angles = np.linspace(-FOV/2, FOV/2, 1080)
        self.z_transform = np.stack([np.cos(self.angles), np.sin(self.angles)], axis=1)

        self.save_data = save_data
        if save_data:
            self.local_map_data_path = path + f"LocalMapData_{test_id}/"
            ensure_path_exists(self.local_map_data_path)
        self.counter = 0
        self.left_longer = None

    def generate_line_local_map(self, scan):
        z = scan[:, None] * self.z_transform
        self.extract_track_boundaries(z)
        left_line, right_line = self.extract_track_boundaries(z)
        left_boundary, right_boundary = self.calculate_visible_segments(left_line, right_line)
        left_extension, right_extension = self.estimate_semi_visible_segments(left_line, right_line, left_boundary, right_boundary)
        local_track = self.regularise_track(left_boundary, right_boundary, left_extension, right_extension)

        if self.save_data:
            np.save(self.local_map_data_path + f"local_map_{self.counter}", local_track)
            np.save(self.local_map_data_path + f"line1_{self.counter}", left_line)
            np.save(self.local_map_data_path + f"line2_{self.counter}", right_line)
            boundaries = np.concatenate((left_boundary, right_boundary), axis=1)
            np.save(self.local_map_data_path + f"boundaries_{self.counter}", boundaries)
            if left_extension is not None:
                boundaries = np.concatenate((left_extension, right_extension), axis=1)
                np.save(self.local_map_data_path + f"boundExtension_{self.counter}", boundaries)
            else:
                np.save(self.local_map_data_path + f"boundExtension_{self.counter}", np.array([]))

        self.counter += 1

        return local_track

    def extract_track_boundaries(self, z):
        z = z[z[:, 0] > -2] # remove points behind the car 
        z = z[np.logical_or(z[:, 0] > 0, np.abs(z[:, 1]) < 2)] # remove points behind the car or too far away
        pt_distances = np.linalg.norm(z[1:] - z[:-1], axis=1)
        inds = np.array(np.where(pt_distances > DISTNACE_THRESHOLD))

        arr_inds = np.arange(len(pt_distances))[inds]
        if np.min(arr_inds) > 2:
            arr_inds = np.insert(arr_inds, 0, -2)
        if np.max(arr_inds) < len(z)-3:
            arr_inds = np.append(arr_inds, len(z)-1)

        # build a list of all the possible boundaries to use. 
        candidate_lines = [z[arr_inds[i]+2:arr_inds[i+1]+1] for i in range(len(arr_inds)-1)]
        # Remove any boundaries that are not realistic
        candidate_lines = [line for line in candidate_lines if not np.all(line[:, 0] < -0.8) or np.all(np.abs(line[:, 1]) > 2.5)]
        candidate_lines = [line for line in candidate_lines if len(line) > 1]

        try:
            left_line = resample_track_points(candidate_lines[0], BOUNDARY_STEP_SIZE, BOUNDARY_SMOOTHING)
            right_line = resample_track_points(candidate_lines[-1], BOUNDARY_STEP_SIZE, BOUNDARY_SMOOTHING)
        except Exception as e:
            print("Exception in track boundary extraction")
            print(e)
            print(len(candidate_lines))
            print(arr_inds)
        if left_line.shape[0] > right_line.shape[0]:
            self.left_longer = True
        else:
            self.left_longer = False

        return left_line, right_line # in time, remove the self.

    def calculate_visible_segments(self, left_line, right_line):
        if self.left_longer:
            left_boundary, right_boundary = calculate_boundary_segments(left_line, right_line)
        else:
            right_boundary, left_boundary = calculate_boundary_segments(right_line, left_line)

        # if len(left_boundary) == 0 or len(right_boundary) == 0:
        #     return left_boundary, right_boundary
        # distances = np.linalg.norm(left_boundary - right_boundary, axis=1)
        # i = -1
        # while distances[i] > FILTER_THRESHOLD and i > -len(distances)+3:
        #     i -= 1
        # left_boundary = left_boundary[:i]
        # right_boundary = right_boundary[:i]

        return left_boundary, right_boundary

    def estimate_semi_visible_segments(self, left_line, right_line, left_boundary, right_boundary):
        if self.left_longer:
            if len(left_line) - len(left_boundary) < 3:
                return None, None
            right_extension, left_extension = extend_boundary_lines(left_line, left_boundary, right_boundary, -1)
        else:
            if len(right_line) - len(right_boundary) < 3:
                return None, None
            left_extension, right_extension = extend_boundary_lines(right_line, right_boundary, left_boundary, 1)

        return left_extension, right_extension
        
    def regularise_track(self, left_boundary, right_boundary, left_extension, right_extension):
        if left_extension is not None:
            left_boundary = np.append(left_boundary, left_extension, axis=0)
            right_boundary = np.append(right_boundary, right_extension, axis=0)
        track_centre_line = (left_boundary + right_boundary) / 2
        widths = np.ones_like(track_centre_line) * TRACK_WIDTH / 2
        local_track = np.concatenate((track_centre_line, widths), axis=1)

        local_track = interpolate_4d_track(local_track, TRACK_SEPEARTION_DISTANCE, 0.01)

        return local_track


def interpolate_4d_track(track, point_seperation_distance=0.8, s=0):
    el_lengths = np.linalg.norm(np.diff(track[:, :2], axis=0), axis=1)
    ss = np.insert(np.cumsum(el_lengths), 0, 0)
    n_points = int(ss[-1] / point_seperation_distance  + 1)
    order_k = min(3, len(track) - 1)
    tck = interpolate.splprep([track[:, 0], track[:, 1], track[:, 2], track[:, 3]], u=ss, k=order_k, s=s)[0]
    track = np.array(interpolate.splev(np.linspace(0, ss[-1], n_points), tck)).T

    return track

def resample_track_points(points, seperation_distance=0.2, smoothing=0.2):
    if points[0, 0] > points[-1, 0]:
        points = np.flip(points, axis=0)

    line_length = np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1))
    n_pts = max(int(line_length / seperation_distance), 2)
    smooth_line = interpolate_track_new(points, None, smoothing)
    resampled_points = interpolate_track_new(smooth_line, n_pts, 0)

    return resampled_points

def calculate_boundary_segments(long_line, short_line):
    found_normal = False
    long_boundary, short_boundary = np.zeros_like(long_line), np.zeros_like(long_line)
    for i in range(long_line.shape[0]):
        distances = np.linalg.norm(short_line - long_line[i], axis=1)

        idx = np.argmin(distances)
        if distances[idx] > MAX_TRACK_WIDTH: 
            if found_normal: 
                break
        else:
            found_normal = True

        long_boundary[i] = long_line[i]
        short_boundary[i] = short_line[idx]

    return long_boundary[:i], short_boundary[:i]

def extend_boundary_lines(long_line, long_boundary, short_boundary, direction=1):
    long_extension = long_line[len(long_boundary):]
    nvecs = calculate_nvecs(long_extension)
    short_extension = long_extension - nvecs * TRACK_WIDTH * direction

    if len(short_boundary) > 0 and len(long_boundary) > 0:
        centre_line = (long_boundary + short_boundary) / 2
        threshold = np.linalg.norm(short_boundary[-1] - centre_line[-1])
        for z in range(len(short_extension)):
            if np.linalg.norm(short_extension[z] - centre_line[-1]) < threshold:
                short_extension[z] = short_boundary[-1]

    return short_extension, long_extension


def interpolate_track_new(points, n_points=None, s=0):
    if len(points) <= 1:
        return points
    order_k = min(3, len(points) - 1)
    tck = interpolate.splprep([points[:, 0], points[:, 1]], k=order_k, s=s)[0]
    if n_points is None: n_points = len(points)
    track = np.array(interpolate.splev(np.linspace(0, 1, n_points), tck)).T

    return track

def calculate_nvecs(line):
    el_lengths = np.linalg.norm(np.diff(line, axis=0), axis=1)
    psi, kappa = tph.calc_head_curv_num.calc_head_curv_num(line, el_lengths, False)
    nvecs = tph.calc_normal_vectors_ahead.calc_normal_vectors_ahead(psi)

    return nvecs

def ensure_path_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
    


if __name__ == "__main__":
    pass