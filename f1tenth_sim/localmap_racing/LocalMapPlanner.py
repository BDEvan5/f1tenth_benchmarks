import numpy as np 
from numba import njit  

from f1tenth_sim.localmap_racing.local_map_utils import *
from f1tenth_sim.localmap_racing.LocalMapGenerator import LocalMapGenerator
from f1tenth_sim.localmap_racing.LocalRaceline import LocalRaceline

np.set_printoptions(precision=4)


LOOKAHEAD_DISTANCE = 1.4
WHEELBASE = 0.33
MAX_STEER = 0.4
MAX_SPEED = 8

VERBOSE = False
# VERBOSE = True

class LocalMapPlanner:
    def __init__(self, test_id, save_data=False, raceline=True):
        self.name = "LocalMapPlanner"
        self.path = f"Logs/{self.name}/"
        ensure_path_exists(self.path)
        self.use_raceline = raceline
        self.save_data = save_data
        if self.save_data:
            self.scan_data_path = self.path + f"ScanData_{test_id}/"
            ensure_path_exists(self.scan_data_path)

        self.counter = 0
                
        self.local_map_generator = LocalMapGenerator(self.path, test_id, save_data)
        self.local_map = None
        if raceline:
            self.local_raceline = LocalRaceline(self.path, test_id)

    def set_map(self, map_name):
        pass
        # self.local_map_generator.set_map(map_name)
        # if self.use_raceline:
        #     self.local_raceline.set_map(map_name)

    def plan(self, obs):
        if self.save_data:
            np.save(self.scan_data_path + f'Scan_{self.counter}.npy', obs['scan'])
        self.local_map = self.local_map_generator.generate_line_local_map(np.copy(obs['scan']), save=True, counter=None)
        if len(self.local_map.track) < 4:
            self.counter += 1
            return np.zeros(2)

        if self.use_raceline:
            self.local_raceline.generate_raceline(self.local_map)
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
        steering_angle = np.clip(steering_angle, -MAX_STEER, MAX_STEER)
        speed = 3
        
        return np.array([steering_angle, speed]), lookahead_point

    def pure_pursuit_racing_line(self, obs):
        lhd_distance = 0.3 + obs['vehicle_speed'] * 0.2
        lookahead_point, speed = self.local_raceline.calculate_lookahead_point(lhd_distance)

        exact_lookahead = np.linalg.norm(lookahead_point)
        steering_angle = get_local_steering_actuation(lookahead_point, exact_lookahead*0.8, WHEELBASE)
        steering_angle = np.clip(steering_angle, -MAX_STEER, MAX_STEER)

        return np.array([steering_angle, speed]), lookahead_point

    
# @njit(cache=True)
def calculate_offset_coords(pts, position, heading):
    rotation = np.array([[np.cos(heading), -np.sin(heading)],
                        [np.sin(heading), np.cos(heading)]])
        
    new_pts = np.matmul(rotation, pts.T).T + position

    return new_pts


     
def get_steering_actuation(pose_theta, lookahead_point, position, lookahead_distance, wheelbase):
    waypoint_y = np.dot(np.array([np.sin(-pose_theta), np.cos(-pose_theta)]), lookahead_point[0:2]-position)
    if np.abs(waypoint_y) < 1e-6:
        return 0.0
    radius = 1/(2.0*waypoint_y/lookahead_distance**2)
    steering_angle = np.arctan(wheelbase/radius)
    return steering_angle


@njit(fastmath=False, cache=True)
def get_local_steering_actuation(lookahead_point, lookahead_distance, wheelbase):
    waypoint_y = lookahead_point[1]
    if np.abs(waypoint_y) < 1e-6:
        return 0.0
    radius = 1/(2.0*waypoint_y/lookahead_distance**2)
    steering_angle = np.arctan(wheelbase/radius)
    return steering_angle
   
