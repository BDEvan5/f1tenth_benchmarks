import numpy as np

from f1tenth_sim.classic_racing.GlobalPurePursuit import GlobalPurePursuit
from f1tenth_sim.classic_racing.planner_utils import RaceTrack
from numba import njit


# WHEELBASE = 0.33
# MAX_STEER = 0.4
# MAX_SPEED = 8
# GRAVITY = 9.81


# class PurePursuit:
#     def __init__(self, map_name):
#         self.racetrack = RaceTrack(map_name, "mu50")
#         self.counter = 0
#         self.constant_lookahead = 0.8
#         self.variable_lookahead = 0.1

#     def plan(self, obs):
#         state = obs["vehicle_state"]

#         lookahead_distance = self.constant_lookahead + state[3] * self.variable_lookahead
#         lookahead_point = self.racetrack.get_lookahead_point(state[:2], lookahead_distance)

#         if state[3] < 1:
#             return np.array([0.0, 4])

#         speed_raceline, steering_angle = get_actuation(state[4], lookahead_point, state[:2], lookahead_distance, WHEELBASE)
#         steering_angle = np.clip(steering_angle, -MAX_STEER, MAX_STEER)
            
#         speed = min(speed_raceline, MAX_SPEED) 
#         action = np.array([steering_angle, speed])

#         return action



# @njit(fastmath=False, cache=True)
# def get_actuation(pose_theta, lookahead_point, position, lookahead_distance, wheelbase):
#     waypoint_y = np.dot(np.array([np.sin(-pose_theta), np.cos(-pose_theta)]), lookahead_point[0:2]-position)
#     speed = lookahead_point[2]
#     if np.abs(waypoint_y) < 1e-6:
#         return speed, 0.
#     radius = 1/(2.0*waypoint_y/lookahead_distance**2)
#     steering_angle = np.arctan(wheelbase/radius)
#     return speed, steering_angle





class TrajectoryAidedLearningReward:
    def __init__(self, map_name, params):
        self.pp = GlobalPurePursuit("_drl_training")
        self.pp.set_map(map_name) 
        # self.pp = PurePursuit(map_name) 

        self.beta_c = params.reward_tal_constant
        self.weights = params.reward_tal_inv_scales
        
    def __call__(self, observation, prev_obs, action):
        if prev_obs is None: return 0

        if observation['lap_complete']:
            return 1  # complete
        if observation['collision']:
            return -1 # crash
        
        pp_act = self.pp.plan(prev_obs)
        rewards = np.abs(pp_act - action) / self.weights 
        reward = self.beta_c * max(1 - np.sum(rewards), 0)

        return reward



