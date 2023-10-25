
import numpy as np
import os
from f1tenth_sim.racing_methods.planning.pp_traj_following.planner_utils import RaceTrack, get_actuation


WHEELBASE = 0.33
MAX_STEER = 0.4
MAX_SPEED = 8
GRAVITY = 9.81
LOOKAHEAD_DISTANCE = 0.8


class PpTrajectoryFollower:
    def __init__(self):
        self.name = "pp_traj_following"
        self.racetrack = None
        self.counter = 0

    def set_map(self, map_name):
        self.racetrack = RaceTrack(map_name)

    def plan(self, obs):
        state = obs["vehicle_state"]

        lookahead_distance = 0.5 + state[3] * 0.18
        lookahead_point = self.racetrack.get_lookahead_point(state[:2], lookahead_distance)

        if state[3] < 1:
            return np.array([0.0, 4])

        speed_raceline, steering_angle = get_actuation(state[4], lookahead_point, state[:2], lookahead_distance, WHEELBASE)
        steering_angle = np.clip(steering_angle, -MAX_STEER, MAX_STEER)
            
        speed = min(speed_raceline, MAX_SPEED) # cap the speed
        action = np.array([steering_angle, speed])

        return action




