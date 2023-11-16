
import numpy as np
from numba import njit
from f1tenth_sim.classic_racing.planner_utils import RaceTrack, CentreLineTrack
from f1tenth_sim.general_utils import BasePlanner


class PurePursuit(BasePlanner):
    def __init__(self, test_id):
        super().__init__("GlobalPurePursuit", test_id)
        self.racetrack = None
        self.constant_lookahead = self.planner_params.constant_lookahead
        self.variable_lookahead = self.planner_params.variable_lookahead

    def set_map(self, map_name):
        self.racetrack = RaceTrack(map_name, self.test_id)

    def set_map_centerline(self, map_name):
        self.racetrack = CentreLineTrack(map_name, 3)

    def plan(self, obs):
        pose = obs["pose"]
        vehicle_speed = obs["vehicle_speed"]

        lookahead_distance = self.constant_lookahead + vehicle_speed * self.variable_lookahead
        lookahead_point = self.racetrack.get_lookahead_point(pose[:2], lookahead_distance)

        if vehicle_speed < 1:
            return np.array([0.0, 4])

        speed_raceline, steering_angle = get_actuation(pose[2], lookahead_point, pose[:2], lookahead_distance, self.vehicle_params.wheelbase)
        steering_angle = np.clip(steering_angle, -self.vehicle_params.max_steer, self.vehicle_params.max_steer)
            
        speed = min(speed_raceline, self.vehicle_params.max_speed)
        action = np.array([steering_angle, speed])

        return action



@njit(fastmath=False, cache=True)
def get_actuation(pose_theta, lookahead_point, position, lookahead_distance, wheelbase):
    waypoint_y = np.dot(np.array([np.sin(-pose_theta), np.cos(-pose_theta)]), lookahead_point[0:2]-position)
    speed = lookahead_point[2]
    if np.abs(waypoint_y) < 1e-6:
        return speed, 0.
    radius = 1/(2.0*waypoint_y/lookahead_distance**2)
    steering_angle = np.arctan(wheelbase/radius)
    return speed, steering_angle

