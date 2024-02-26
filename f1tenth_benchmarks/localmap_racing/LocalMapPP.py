import numpy as np 
from numba import njit  
import trajectory_planning_helpers as tph
from scipy import interpolate
import os

from f1tenth_benchmarks.localmap_racing.LocalMapGenerator import LocalMapGenerator
from f1tenth_benchmarks.localmap_racing.local_opt_min_curv import local_opt_min_curv
from f1tenth_benchmarks.utils.BasePlanner import BasePlanner


class LocalMapPP(BasePlanner): 
    def __init__(self, test_id, save_data=False, raceline=True):
        super().__init__("LocalMapPP", test_id)
        self.local_map_generator = LocalMapGenerator(self.data_root_path, test_id, save_data)
        self.local_track = None

        self.use_raceline = raceline
        if self.use_raceline:
            self.raceline = None
            self.s_raceline = None
            self.vs = None
            self.raceline_data_path = self.data_root_path + f"RacingLineData_{test_id}/"
            ensure_path_exists(self.raceline_data_path)

            p = self.planner_params
            self.ggv = np.array([[0, p.max_longitudinal_acc, p.max_lateral_acc],
                        [self.vehicle_params.max_speed, p.max_longitudinal_acc, p.max_lateral_acc]])
            self.ax_max_machine = np.array([[0, p.max_longitudinal_acc],
                                            [self.vehicle_params.max_speed, p.max_longitudinal_acc]])

    def plan(self, obs):
        self.local_track = self.local_map_generator.generate_line_local_map(np.copy(obs['scan']))
        self.step_counter += 1 
        if len(self.local_track) < 4:
            self.step_counter += 1
            return np.zeros(2)

        if self.use_raceline:
            self.generate_minimum_curvature_path()
            self.generate_max_speed_profile()

            raceline = np.concatenate([self.raceline, self.vs[:, None]], axis=-1)
            np.save(self.raceline_data_path  + f'LocalRaceline_{self.step_counter}.npy', raceline) 
            #! URGENT: Must fix
            #! Bug: the racelines are saved from 0...1500000. They should rather be saved with map name and only for the first lap..... 
            action = self.pure_pursuit_racing_line(obs)
        else:
            action = self.pure_pursuit_center_line()

        return action
        
    def pure_pursuit_center_line(self):
        current_progress = np.linalg.norm(self.local_track[0, 0:2])
        lookahead = self.planner_params.centre_lookahead_distance + current_progress

        local_el = np.linalg.norm(np.diff(self.local_track[:, 0:2], axis=0), axis=1)
        s_track = np.insert(np.cumsum(local_el), 0, 0)
        lookahead = min(lookahead, s_track[-1]) 
        lookahead_point = interp_2d_points(lookahead, s_track, self.local_track[:, 0:2])

        true_lookahead_distance = np.linalg.norm(lookahead_point)
        steering_angle = get_local_steering_actuation(lookahead_point, true_lookahead_distance, self.vehicle_params.wheelbase)
        action = np.array([steering_angle, self.planner_params.constant_speed])
        
        return action

    def generate_minimum_curvature_path(self):
        track = self.local_track.copy()

        track[:, 2:] -= self.planner_params.path_exclusion_width / 2

        try:
            alpha, nvecs = local_opt_min_curv(track, self.planner_params.kappa_bound, 0, fix_s=True, fix_e=False)
            # alpha, nvecs = local_opt_min_curv(track, local_map.nvecs, self.planner_params.kappa_bound, 0, print_debug=False, psi_s=local_map.psi[0], psi_e=local_map.psi[-1], fix_s=True, fix_e=False)
            self.raceline = track[:, :2] + np.expand_dims(alpha, 1) * nvecs
        except Exception as e:
            self.raceline = track[:, :2]

        self.tck = interpolate.splprep([self.raceline[:, 0], self.raceline[:, 1]], k=3, s=0)[0]
        
    def generate_max_speed_profile(self):
        max_speed = self.planner_params.max_speed
        mu = self.planner_params.mu * np.ones_like(self.raceline[:, 0])

        raceline_el_lengths = np.linalg.norm(np.diff(self.raceline, axis=0), axis=1)
        self.s_raceline = np.insert(np.cumsum(raceline_el_lengths), 0, 0)
        _, raceline_curvature = tph.calc_head_curv_num.calc_head_curv_num(self.raceline, raceline_el_lengths, False)

        self.vs = tph.calc_vel_profile.calc_vel_profile(self.ax_max_machine, raceline_curvature, raceline_el_lengths, False, 0, self.vehicle_params.vehicle_mass, ggv=self.ggv, mu=mu, v_max=max_speed, v_start=max_speed, v_end=max_speed)

    def calculate_zero_point_progress(self):
        n_pts = np.count_nonzero(self.s_raceline < 5) # search first 4 m
        s_raceline = self.s_raceline[:n_pts]
        raceline = self.raceline[:n_pts]
        new_points = np.linspace(0, s_raceline[-1], int(s_raceline[-1]*100)) #cm accuracy
        xs, ys = interp_2d_points(new_points, s_raceline, raceline)
        raceline = np.concatenate([xs[:, None], ys[:, None]], axis=-1)
        dists = np.linalg.norm(raceline, axis=1)
        t_new = (new_points[np.argmin(dists)] / self.s_raceline[-1])

        return [t_new]

    def pure_pursuit_racing_line(self, obs):
        lookahead_distance = self.planner_params.constant_lookahead + (obs['vehicle_speed']/self.vehicle_params.max_speed) * (self.planner_params.variable_lookahead)
        current_s = self.calculate_zero_point_progress()
        lookahead_s = current_s + lookahead_distance / self.s_raceline[-1]
        lookahead_point = np.array(interpolate.splev(lookahead_s, self.tck, ext=3)).T
        if len(lookahead_point.shape) > 1: lookahead_point = lookahead_point[0]

        exact_lookahead = np.linalg.norm(lookahead_point)
        steering_angle = get_local_steering_actuation(lookahead_point, exact_lookahead, self.vehicle_params.wheelbase) 
        speed = np.interp(current_s, self.s_raceline/self.s_raceline[-1], self.vs)[0]

        return np.array([steering_angle, speed])

def ensure_path_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
    

def interp_2d_points(ss, xp, points):
    xs = np.interp(ss, xp, points[:, 0])
    ys = np.interp(ss, xp, points[:, 1])
    
    return xs, ys


@njit(fastmath=False, cache=True)
def get_local_steering_actuation(lookahead_point, lookahead_distance, wheelbase):
    waypoint_y = lookahead_point[1]
    if np.abs(waypoint_y) < 1e-6:
        return 0.0
    radius = 1/(2.0*waypoint_y/lookahead_distance**2)
    steering_angle = np.arctan(wheelbase/radius)
    return steering_angle
   
