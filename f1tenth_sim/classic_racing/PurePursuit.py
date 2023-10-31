
import numpy as np
import os
from numba import njit
from f1tenth_sim.classic_racing.planner_utils import RaceTrack


WHEELBASE = 0.33
MAX_STEER = 0.4
MAX_SPEED = 8
# MAX_SPEED = 4
GRAVITY = 9.81
LOOKAHEAD_DISTANCE = 0.8


class PurePursuit:
    def __init__(self):
        self.name = "pp_traj_following"
        self.racetrack = None
        self.counter = 0
        self.constant_lookahead = 0.8
        self.variable_lookahead = 0

    def set_map(self, map_name):
        self.racetrack = RaceTrack(map_name)

    def plan(self, obs):
        state = obs["vehicle_state"]

        # lookahead_distance = 0.5 + state[3] * 0.18
        # lookahead_distance = 0.2 + state[3] * 0.14
        lookahead_distance = self.constant_lookahead + state[3] * self.variable_lookahead
        lookahead_point = self.racetrack.get_lookahead_point(state[:2], lookahead_distance)
        # lookahead_point = self.racetrack.get_lookahead_point(state[:2], 0.8)

        if state[3] < 1:
            return np.array([0.0, 4])

        speed_raceline, steering_angle = get_actuation(state[4], lookahead_point, state[:2], lookahead_distance, WHEELBASE)
        steering_angle = np.clip(steering_angle, -MAX_STEER, MAX_STEER)
            
        speed = min(speed_raceline, MAX_SPEED) *0.75 # cap the speed
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

