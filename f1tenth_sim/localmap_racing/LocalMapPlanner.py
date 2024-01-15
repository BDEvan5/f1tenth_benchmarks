import numpy as np 
from numba import njit  
import trajectory_planning_helpers as tph
from scipy import interpolate
import os

from f1tenth_sim.localmap_racing.LocalMap import *
from f1tenth_sim.localmap_racing.LocalMapGenerator import LocalMapGenerator
from f1tenth_sim.localmap_racing.local_opt_min_curv import local_opt_min_curv

np.set_printoptions(precision=4)
KAPPA_BOUND = 0.8
VEHICLE_WIDTH = 1
ax_max_machine = np.array([[0, 8.5],[8, 8.5]])
ggv = np.array([[0, 8.5, 8.5], [8, 8.5, 8.5]])
MU = 0.6
MAX_SPEED = 8
VEHICLE_MASS = 3.4

LOOKAHEAD_DISTANCE = 1.4
WHEELBASE = 0.33
MAX_STEER = 0.4

class LocalMapPlanner:
    def __init__(self, test_id, save_data=False, raceline=True):
        self.name = "LocalMapPlanner"
        self.path = f"logs/{self.name}/"
        ensure_path_exists(self.path)
        ensure_path_exists(self.path + f"RawData_{test_id}/")
        self.counter = 0
                
        self.local_map_generator = LocalMapGenerator(self.path, test_id, save_data)
        self.local_map = None

        self.use_raceline = raceline
        if self.use_raceline:
            self.raceline = None
            self.s_raceline = None
            self.vs = None
            self.raceline_data_path = self.path + f"RacingLineData_{test_id}/"
            ensure_path_exists(self.raceline_data_path)

    def set_map(self, map_name):
        pass

    def plan(self, obs):
        self.local_map = self.local_map_generator.generate_line_local_map(np.copy(obs['scan']))
        if len(self.local_map.track) < 4:
            self.counter += 1
            return np.zeros(2)

        if self.use_raceline:
            self.generate_minimum_curvature_path(self.local_map)
            self.generate_max_speed_profile()

            raceline = np.concatenate([self.raceline, self.vs[:, None]], axis=-1)
            np.save(self.raceline_data_path  + f'LocalRaceline_{self.counter}.npy', raceline)
            action, lhd_pt = self.pure_pursuit_racing_line(obs)
        else:
            action, lhd_pt = self.pure_pursuit_center_line()

        self.counter += 1
        return action
        
    def pure_pursuit_center_line(self):
        current_progress = np.linalg.norm(self.local_map.track[0, 0:2])
        lookahead = LOOKAHEAD_DISTANCE + current_progress

        lookahead = min(lookahead, self.local_map.s_track[-1]) 
        lookahead_point = interp_2d_points(lookahead, self.local_map.s_track, self.local_map.track[:, 0:2])

        steering_angle = get_local_steering_actuation(lookahead_point, LOOKAHEAD_DISTANCE, WHEELBASE)
        speed = 3
        
        return np.array([steering_angle, speed]), lookahead_point

    def generate_minimum_curvature_path(self, local_map):
        track = local_map.track.copy()
        track[:, 2:] -= VEHICLE_WIDTH / 2

        try:
            alpha = local_opt_min_curv(track, local_map.nvecs, KAPPA_BOUND, 0, print_debug=False, psi_s=local_map.psi[0], psi_e=local_map.psi[-1], fix_s=True, fix_e=False)
            self.raceline = track[:, :2] + np.expand_dims(alpha, 1) * local_map.nvecs
        except Exception as e:
            print(f"Error in optimising min curvature path: {e}")
            self.raceline = track[:, :2]

        self.tck = interpolate.splprep([self.raceline[:, 0], self.raceline[:, 1]], k=3, s=0)[0]
        
    def generate_max_speed_profile(self, starting_speed=MAX_SPEED):
        mu = MU * np.ones_like(self.raceline[:, 0]) 
        raceline_el_lengths = np.linalg.norm(np.diff(self.raceline, axis=0), axis=1)
        self.s_raceline = np.insert(np.cumsum(raceline_el_lengths), 0, 0)
        _, raceline_curvature = tph.calc_head_curv_num.calc_head_curv_num(self.raceline, raceline_el_lengths, False)

        self.vs = tph.calc_vel_profile.calc_vel_profile(ax_max_machine, raceline_curvature, raceline_el_lengths, False, 0, VEHICLE_MASS, ggv=ggv, mu=mu, v_max=MAX_SPEED, v_start=MAX_SPEED, v_end=MAX_SPEED)

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
        lookahead_distance = 0.3 + obs['vehicle_speed'] * 0.2
        current_s = self.calculate_zero_point_progress()
        lookahead_s = current_s + lookahead_distance / self.s_raceline[-1]
        lookahead_point = np.array(interpolate.splev(lookahead_s, self.tck, ext=3)).T
        if len(lookahead_point.shape) > 1: lookahead_point = lookahead_point[0]

        exact_lookahead = np.linalg.norm(lookahead_point)
        steering_angle = get_local_steering_actuation(lookahead_point, exact_lookahead, WHEELBASE) 
        speed = np.interp(current_s, self.s_raceline/self.s_raceline[-1], self.vs)[0]

        return np.array([steering_angle, speed]), lookahead_point

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
    # steering_angle = np.clip(steering_angle, -MAX_STEER, MAX_STEER)
    return steering_angle
   
