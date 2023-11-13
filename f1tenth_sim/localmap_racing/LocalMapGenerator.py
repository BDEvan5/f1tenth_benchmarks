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

        self.line_1 = None
        self.line_2 = None
        self.max_s_1 = 0
        self.max_s_2 = 0
        self.boundary_1 = None
        self.boundary_2 = None
        self.boundary_extension_1 = None
        self.boundary_extension_2 = None
        self.smooth_track = None

        self.search_pts_a = []
        self.search_pts_b = []
        self.true_search_pts = []

    def generate_line_local_map(self, scan):
        z = scan[:, None] * self.z_transform
        self.extract_track_boundaries(z)
        left_line, right_line = self.extract_track_boundaries(z)
        left_boundary, right_boundary = self.calculate_visible_segments(left_line, right_line)
        left_boundary, right_boundary = self.trim_points(left_boundary, right_boundary)
        left_extension, right_extension = self.estimate_semi_visible_segments(left_line, right_line, left_boundary, right_boundary)
        # self.regularise_track()

        # self.search_pts_a = []
        # self.search_pts_b = []

        # self.scan_xs = self.coses * scan
        # self.scan_ys = self.sines * scan

        # pts, pt_distances, inds = self.extract_track_lines()
        # self.extract_boundaries(pts, pt_distances, inds)
        # self.estimate_center_line_dual_boundary()
        # self.extend_center_line_projection()

        # local_track = self.build_local_track()
        # smooth_track = self.smooth_track_spline(local_track)
        b1 = left_boundary 
        b2 = right_boundary 
        if left_extension is not None:
            b1 = np.append(left_boundary, left_extension, axis=0)
            b2 = np.append(right_boundary, right_extension, axis=0)
        new_track = (b1 + b2) / 2
        w1 = np.linalg.norm(b1 - new_track, axis=1)[:, None]
        w2 = np.linalg.norm(b2 - new_track, axis=1)[:, None]
        local_track = np.concatenate((new_track, w1, w2), axis=1)
        # if self.counter > 250:
        #     print(f"B1: {self.boundary_1}")
        #     print(f"B2: {self.boundary_2}")
        #     print(f"BE1: {self.boundary_extension_1}")
        #     print(f"BE2: {self.boundary_extension_2}")
        #     print(f"LT: {local_track}")
        #     plt.figure(1)
        #     plt.clf()
        #     plt.plot(self.boundary_1[:, 0], self.boundary_1[:, 1], '-o', color='black', markersize=8)
        #     plt.plot(self.boundary_2[:, 0], self.boundary_2[:, 1], '-o', color='black', markersize=8)
        #     if self.boundary_extension_1 is not None:
        #         plt.plot(self.boundary_extension_1[:, 0], self.boundary_extension_1[:, 1], '-o', color='pink', markersize=8)
        #         plt.plot(self.boundary_extension_2[:, 0], self.boundary_extension_2[:, 1], '-o', color='pink', markersize=8)
        #     plt.plot(local_track[:, 0], local_track[:, 1], '-X', color='orange', markersize=10)
        #     plt.axis('equal')
        #     plt.show()

        print(f"LocalTrack: {local_track.shape[0]}")
        local_map = LocalMap(local_track)

        self.smooth_track = local_map
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

    def get_extraction_stats(self):
        line_inner = self.line_1.cs[-1]
        line_outer = self.line_2.cs[-1]
        true_center = (self.boundary_1 + self.boundary_2) / 2
        calculated_center = np.sum(np.linalg.norm(np.diff(true_center, axis=0), axis=1))
        if self.boundary_extension_1 is not None:
            true_center_proj = (self.boundary_extension_1 + self.boundary_extension_2) / 2
            projected_center = np.sum(np.linalg.norm(np.diff(true_center_proj, axis=0), axis=1))
        else: projected_center = 0

        lm_stats = np.array([line_inner, line_outer, calculated_center, projected_center])
        return lm_stats

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
        
    def trim_points(self, left_boundary, right_boundary):
        true_center_line = (left_boundary + right_boundary) / 2
        dists = np.linalg.norm(np.diff(true_center_line, axis=0), axis=1)
        center_point_threshold = 0.1
        removal_n = np.sum(dists < center_point_threshold)
        if removal_n > 0:
            # print(f"Removing: {removal_n} points")
            left_boundary = left_boundary[:-removal_n]
            right_boundary = right_boundary[:-removal_n]

        return left_boundary, right_boundary
        

    def project_side_to_track(self, side):
        side_lm = LocalMap(side)
        center_line = side + side_lm.nvecs * TRACK_WIDTH / 2
        n_pts = int(side.shape[0] / 2)
        center_line = interpolate_track(center_line, n_pts, 0)

        center_line = center_line[center_line[:, 0] > -1] # remove points behind the car

        pt_init = np.linalg.norm(center_line[0, :])
        pt_final = np.linalg.norm(center_line[-1, :])
        if pt_final < pt_init: center_line = np.flip(center_line, axis=0)

        ws = np.ones_like(center_line) * TRACK_WIDTH / 2
        track = np.concatenate((center_line, ws), axis=1)

        return track

    def estimate_center_line_dual_boundary(self):
        search_pt = [-1, 0]
        max_pts = 50
        end_threshold = 0.05
        self.boundary_1 = np.zeros((max_pts, 2))
        self.boundary_2 = np.zeros((max_pts, 2))
        center_pts = np.zeros((max_pts, 2))
        self.max_s_1, self.max_s_2 = 0, 0 
        for i in range(max_pts):
            if i == 0:
                pt_1, self.max_s_1 = self.line_1.find_closest_point(search_pt, self.max_s_1, "Line1")
                pt_2, self.max_s_2 = self.line_2.find_closest_point(search_pt, self.max_s_2, "Line2")
            else:
                pt_1, pt_2, search_pt = self.calculate_next_boundaries(pt_1, pt_2)

            line_center = (pt_1 + pt_2) / 2
            center_pts[i] = line_center

            if np.all(np.isclose(pt_1, self.boundary_1[i-1])) and np.all(np.isclose(pt_2, self.boundary_2[i-1])): 
                i -= 1
                break

            self.boundary_1[i] = pt_1
            self.boundary_2[i] = pt_2

            long_distance = np.linalg.norm(pt_1 - self.line_1.points[-1])
            short_distance = np.linalg.norm(pt_2 - self.line_2.points[-1])
            if long_distance < end_threshold and short_distance < end_threshold:
                # print(f"{i}-> Breaking because of long ({long_distance}) and short ({short_distance}) distances >>> Pt1: {pt_1} :: Pt2: {pt_2}")
                break

        self.boundary_1 = self.boundary_1[:i+1]
        self.boundary_2 = self.boundary_2[:i+1]

        # if len(self.boundary_1) < 2:
        #     raise RuntimeError(f"Not enough points found -> {len(self.boundary_1)}")

    def extend_center_line_projection(self):
        true_center_line = (self.boundary_1 + self.boundary_2) / 2
        dists = np.linalg.norm(np.diff(true_center_line, axis=0), axis=1)
        center_point_threshold = 0.15
        removal_n = np.sum(dists < center_point_threshold)
        # print(f"last 5 dists: {dists[-5:]}")
        if removal_n > 0:
            # print(f"Removing: {removal_n} points")
            self.boundary_1 = self.boundary_1[:-removal_n]
            self.boundary_2 = self.boundary_2[:-removal_n]
        true_center_line = (self.boundary_1 + self.boundary_2) / 2

        if self.max_s_1 > 0.99 and self.max_s_2 > 0.99:
            self.boundary_extension_1 = None
            self.boundary_extension_2 = None
            return # no extension required
        
        if self.max_s_1 > self.max_s_2:
            projection_line = self.line_2
            boundary = self.boundary_2
            direction = -1
        else:
            projection_line = self.line_1
            boundary = self.boundary_1
            direction = 1

        _pt, current_s = projection_line.find_closest_point_true(boundary[-1], 0)
        # current_s = [current_s]
        length_remaining = (1-current_s[0]) * projection_line.cs[-1]
        step_size = 0.5
        if length_remaining < step_size:
            self.boundary_extension_1 = None
            self.boundary_extension_2 = None
            return # no extension required

        n_pts = int((1-current_s[0]) * projection_line.cs[-1] / step_size + 1)
        new_boundary_points = projection_line.extract_line_portion(np.linspace(current_s[0], 1, n_pts))
        new_projection_line = LocalLine(new_boundary_points)

        # project to center line
        extra_center_line = new_projection_line.track + new_projection_line.nvecs * TRACK_WIDTH/2 * direction

        extra_projected_boundary = new_projection_line.track + new_projection_line.nvecs * TRACK_WIDTH * direction

        if self.max_s_1 > self.max_s_2:
            self.boundary_extension_1 = extra_projected_boundary
            self.boundary_extension_2 = new_projection_line.track
        else:
            self.boundary_extension_1 = new_projection_line.track
            self.boundary_extension_2 = extra_projected_boundary

        # remove last point since it has now been replaced.
        self.boundary_1 = self.boundary_1[:-1]
        self.boundary_2 = self.boundary_2[:-1]

    def calculate_next_boundaries(self, pt_1, pt_2):
        step_size = 0.6
        line_center = (pt_1 + pt_2) / 2
        theta = calculate_track_direction(pt_1, pt_2)

        weighting = 0.7
        search_pt_a = (pt_2 * (weighting) + line_center * (1- weighting)) 
        search_pt_b = (pt_1 * (weighting) + line_center * (1- weighting)) 
        search_pt_a = search_pt_a + step_size * np.array([np.cos(theta), np.sin(theta)])
        search_pt_b = search_pt_b + step_size * np.array([np.cos(theta), np.sin(theta)])

        self.search_pts_a.append(search_pt_a)
        self.search_pts_b.append(search_pt_b)

        pt_1_a, max_s_1_a = self.line_1.find_closest_point(search_pt_a, self.max_s_1, "Line1_a")
        pt_2_a, max_s_2_a = self.line_2.find_closest_point(search_pt_a, self.max_s_2, "Line2_a")

        pt_1_b, max_s_1_b = self.line_1.find_closest_point(search_pt_b, self.max_s_1, "Line1_b")
        pt_2_b, max_s_2_b = self.line_2.find_closest_point(search_pt_b, self.max_s_2, "Line2_b")

        # test to find the best candidate
        sum_s_a = max_s_1_a + max_s_2_a
        sum_s_b = max_s_1_b + max_s_2_b

        if sum_s_a < sum_s_b:
            pt_1, self.max_s_1 = pt_1_a, max_s_1_a
            pt_2, self.max_s_2 = pt_2_a, max_s_2_a
            search_pt = search_pt_a
        else:
            pt_1, self.max_s_1 = pt_1_b, max_s_1_b
            pt_2, self.max_s_2 = pt_2_b, max_s_2_b
            search_pt = search_pt_b
        self.true_search_pts.append(search_pt)

        return pt_1, pt_2, search_pt

    def build_local_track(self):
        """
        This generates the list of center line points that keep the nvecs the same as they were before
        - if the track is a dual-build track, then an extra point is added to preserve the last nvec
        - if the track is extended, the last nvec is adjusted to keep the correct directions.
        """
        if self.boundary_extension_1 is not None:
            boundary_1 = np.append(self.boundary_1, self.boundary_extension_1, axis=0)
            boundary_2 = np.append(self.boundary_2, self.boundary_extension_2, axis=0)

            nvecs = boundary_1 - boundary_2
            psi_nvecs = np.arctan2(nvecs[:, 1], nvecs[:, 0])
            psi_tanvecs = psi_nvecs + np.pi/2
            c_line = np.zeros((boundary_1.shape[0], 2))
        else:
            boundary_1 = self.boundary_1
            boundary_2 = self.boundary_2

            nvecs = boundary_1 - boundary_2
            psi_nvecs = np.arctan2(nvecs[:, 1], nvecs[:, 0])
            psi_tanvecs = psi_nvecs + np.pi/2
            psi_tanvecs = normalise_psi(psi_tanvecs)

            extension = 0.6 * np.array([np.cos(psi_tanvecs[-1]), np.sin(psi_tanvecs[-1])])
            extension = np.reshape(extension, (1, 2))
            pt_1 = (boundary_1[-1, :] + extension)
            pt_2 = (boundary_2[-1, :] + extension)
            boundary_1 = np.append(boundary_1, pt_1, axis=0)
            boundary_2 = np.append(boundary_2, pt_2, axis=0)
            psi_tanvecs = np.append(psi_tanvecs, [psi_tanvecs[-1]], axis=0)
            c_line = np.zeros((boundary_1.shape[0], 2))

        c_line[0] = (boundary_1[0] + boundary_2[0]) / 2
        c_line[0, 1] = 0
        # c_line[0] = np.zeros(2)
        search_size = 2
        for i in range(1, len(c_line)):
            theta = psi_tanvecs[i-1]
            line1 = [boundary_1[i], boundary_2[i]]
            if i == 1:
                line2 = [c_line[i-1], c_line[i-1] + np.array([np.cos(theta), np.sin(theta)]) * search_size]
            else:
                line2 = [c_line[i-2], c_line[i-2] + np.array([np.cos(theta), np.sin(theta)]) * search_size]

            intersection = calculate_intersection(line1, line2)
            if intersection is None: # or intersection[0] == 1e9:
                raise ValueError(f"No intersection found between {line1} and {line2}")
            c_line[i] = intersection

        ws_1 = np.linalg.norm(c_line - boundary_1[:len(c_line)], axis=1)[:, None]
        ws_2 = np.linalg.norm(c_line - boundary_2[:len(c_line)], axis=1)[:, None]

        track = np.concatenate((c_line, ws_2, ws_1), axis=1)

        return track

    def build_true_center_track(self):
        true_center_line = (self.boundary_1 + self.boundary_2) / 2
        ws = np.linalg.norm(self.boundary_1 - true_center_line, axis=1)
        ws = ws[:, None] * np.ones_like(true_center_line)
        track = np.concatenate([true_center_line, ws], axis=1)

        return track



    def smooth_track_spline(self, local_track):
        old_track = np.copy(local_track)
        old_line = LocalLine(old_track)

        o_l1 = old_line.track[:, :2] + old_line.nvecs * old_line.track[:, 2][:, None]
        o_l2 = old_line.track[:, :2] - old_line.nvecs * old_line.track[:, 3][:, None]

        smoothing_s = 0.5

        ws = np.ones(len(local_track))
        order_k = min(3, len(local_track))
        if len(local_track) < 5: 
            print(f"Problem: n points too small {len(local_track)} points")
            return old_track
        tck = interpolate.splprep([local_track[:, 0], local_track[:, 1]], w=ws, k=order_k, s=smoothing_s)[0]
        n_points = len(local_track)
        new_track_pts = np.array(interpolate.splev(np.linspace(0, 1, n_points), tck)).T
        # repeat for correct spacing
        tck = interpolate.splprep([new_track_pts[:, 0], new_track_pts[:, 1]], w=ws, k=order_k, s=0)[0]
        new_track_pts = np.array(interpolate.splev(np.linspace(0, 1, n_points), tck)).T

        dists1 = np.linalg.norm(np.diff(o_l1, axis=0), axis=1)
        dists1 = np.append(dists1, 1)
        dists2 = np.linalg.norm(np.diff(o_l2, axis=0), axis=1)
        dists2 = np.append(dists2, 1)

        o_l1_t = o_l1[dists1 > 0.2]
        o_l2_t = o_l2[dists2 > 0.2]

        boundary_1 = TrackBoundary(o_l1_t, False)
        boundary_2 = TrackBoundary(o_l2_t, False)
        ws = np.zeros_like(new_track_pts)
        for i in range(len(local_track)):
            closest_pt_1, t_1 = boundary_1.find_closest_point(new_track_pts[i], 0)
            closest_pt_2, t_2 = boundary_2.find_closest_point(new_track_pts[i], 0)
            ws[i, 0] = np.linalg.norm(closest_pt_1 - new_track_pts[i])
            ws[i, 1] = np.linalg.norm(closest_pt_2 - new_track_pts[i])

        new_track = np.concatenate((new_track_pts, ws), axis=1)

        return new_track




if __name__ == "__main__":
    pass