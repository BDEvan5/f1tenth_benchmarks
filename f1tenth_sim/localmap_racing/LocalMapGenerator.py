import numpy as np

from f1tenth_sim.localmap_racing.local_map_utils import *
from f1tenth_sim.localmap_racing.generator_utils import *
from f1tenth_sim.localmap_racing.LocalMap import LocalMap

np.set_printoptions(precision=4)

DISTNACE_THRESHOLD = 1.4 # distance in m for an exception
TRACK_WIDTH = 1.8 # use fixed width
POINT_SEP_DISTANCE = 0.8
FOV = 4.7


class LocalMapGenerator:
    def __init__(self, path, test_id, save_data) -> None:
        self.angles = np.linspace(-FOV/2, FOV/2, 1080)
        self.z_transform = np.stack([np.cos(self.angles), np.sin(self.angles)], axis=1)

        self.save_data = save_data
        if save_data:
            self.local_map_data_path = path + f"RawData_{test_id}/LocalMapData_{test_id}/"
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

        local_map = LocalMap(local_track)

        if self.save_data:
            np.save(self.local_map_data_path + f"local_map_{self.counter}", local_map.track)
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
        print(f"Counter: {self.counter} Track length: {local_track.shape[0]}")

        return local_map

    def extract_track_boundaries(self, z):
        z = z[z[:, 0] > -2] # remove points behind the car 
        z = z[np.logical_or(z[:, 0] > 0, np.abs(z[:, 1]) < 2)] # remove points behind the car or too far away
        pt_distances = np.linalg.norm(z[1:] - z[:-1], axis=1)
        inds = np.array(np.where(pt_distances > DISTNACE_THRESHOLD))

        arr_inds = np.arange(len(pt_distances))[inds]

        candidate_lines = []
        arr_inds = np.insert(arr_inds, 0, -2)
        arr_inds = np.append(arr_inds, len(z)-1)
        # build a list of all the possible boundaries to use. 
        candidate_lines = [z[arr_inds[i]+2:arr_inds[i+1]+1] for i in range(len(arr_inds)-1)]
        # Remove any boundaries that are not realistic
        candidate_lines = [line for line in candidate_lines if not np.all(line[:, 0] < -0.8) or np.all(np.abs(line[:, 1]) > 2.5)]

        step_size = 0.6
        left_line = resample_track_points(candidate_lines[0], step_size, 0.2)
        right_line = resample_track_points(candidate_lines[-1], step_size, 0.2)
        if left_line.shape[0] > right_line.shape[0]:
            self.left_longer = True
        else:
            self.left_longer = False

        return left_line, right_line # in time, remove the self.

    def calculate_visible_segments(self, left_line, right_line):
        max_pts = max(left_line.shape[0], right_line.shape[0])
        # consdier refactoring this to a function that can be called with reverse order
        left_boundary = np.zeros((max_pts, 2))
        right_boundary = np.zeros((max_pts, 2))
        for i in range(max_pts):
            if self.left_longer:
                distances = np.linalg.norm(right_line - left_line[i], axis=1)
            else:
                distances = np.linalg.norm(left_line - right_line[i], axis=1)
            idx = np.argmin(distances)
            if distances[idx] > 2.5: #!Magic number
                break # no more points are visible
            
            if self.left_longer:
                left_boundary[i] = left_line[i]
                right_boundary[i] = right_line[idx]
            else:
                left_boundary[i] = left_line[idx]
                right_boundary[i] = right_line[i]

        left_boundary = left_boundary[:i]
        right_boundary = right_boundary[:i]

        return left_boundary, right_boundary

    def estimate_semi_visible_segments(self, left_line, right_line, left_boundary, right_boundary):
        if self.left_longer:
            unmatched_points = len(left_line) - len(left_boundary)
        else:
            unmatched_points = len(right_line) - len(right_boundary)

        if unmatched_points < 3: 
            return None, None

        if self.left_longer:
            left_extension = left_line[len(left_boundary):]
            side_lm = LocalMap(left_extension)
            right_extension = left_extension + side_lm.nvecs * TRACK_WIDTH 
        else:
            right_extension = right_line[len(right_boundary):]
            side_lm = LocalMap(right_extension)
            left_extension = right_extension - side_lm.nvecs * TRACK_WIDTH

        # remove corner points closer to the centre line than the last true point
        if len(left_boundary) > 0 and len(right_boundary) > 0:
            centre_line = (left_boundary + right_boundary) / 2
            if self.left_longer:
                threshold = np.linalg.norm(right_boundary[-1] - centre_line[-1])
            else:
                threshold = np.linalg.norm(left_boundary[-1] - centre_line[-1])

            if self.left_longer:
                for z in range(len(right_extension)):
                    if np.linalg.norm(right_extension[z] - centre_line[-1]) < threshold:
                        right_extension[z] = right_boundary[-1]
            else:
                for z in range(len(left_extension)):
                    if np.linalg.norm(left_extension[z] - centre_line[-1]) < threshold:
                        left_extension[z] = left_boundary[-1]

        return left_extension, right_extension
        
    def regularise_track(self, left_boundary, right_boundary, left_extension, right_extension):
        if left_extension is not None:
            left_boundary = np.append(left_boundary, left_extension, axis=0)
            right_boundary = np.append(right_boundary, right_extension, axis=0)
        new_track = (left_boundary + right_boundary) / 2
        w1 = np.linalg.norm(left_boundary - new_track, axis=1)[:, None]
        w2 = np.linalg.norm(right_boundary - new_track, axis=1)[:, None]
        local_track = np.concatenate((new_track, w1, w2), axis=1)

        track_regularisation = 0.4
        local_track = interpolate_4d_track(local_track, track_regularisation)

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



if __name__ == "__main__":
    pass