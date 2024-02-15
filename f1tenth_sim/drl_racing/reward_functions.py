import numpy as np
np.printoptions(precision=2, suppress=True)

from f1tenth_sim.classic_racing.GlobalPurePursuit import GlobalPurePursuit
from f1tenth_sim.utils.track_utils import CentreLine



def create_reward_function(params, map_name):
    if params.reward == "Progress":
        return ProgressReward(params)
    elif params.reward == "CTH":
        return CrossTrackHeadingReward(params, map_name)
    elif params.reward == "TAL":
        return TrajectoryAidedLearningReward(params, map_name)
    else:
        raise ValueError(f"Unknown reward function: {params.reward}")




class TrajectoryAidedLearningReward:
    def __init__(self, params, map_name):
        self.pp = GlobalPurePursuit("_drl_training", init_folder=False, extra_params={"training": True})
        self.pp.set_map(map_name) 

        self.beta_c = params.reward_tal_constant
        self.weights = params.reward_tal_inv_scales
        
    def __call__(self, observation, prev_obs, action):
        if prev_obs is None: return 0

        if observation['lap_complete']:
            return 1  # complete
        if observation['collision']:
            return -1 # crash
        
        pp_act = self.pp.plan(prev_obs)
        weighted_difference = np.abs(pp_act - action) / self.weights 
        reward = self.beta_c * max(1 - np.sum(weighted_difference), 0)

        return reward


class ProgressReward:
    def __init__(self, params):
        pass
        self.previous_progress = 0
        self.progress_weight = params.progress_weight
        
    def __call__(self, observation, prev_obs, action):
        if self.previous_progress == 0: return 0

        if observation['lap_complete']:
            self.previous_progress = 0
            return 1  # complete
        if observation['collision']:
            self.previous_progress = 0
            return -1 # crash
        
        progress_diff = observation['progress'] - self.previous_progress
        reward = self.progress_weight * progress_diff

        return reward




class CrossTrackHeadingReward:
    def __init__(self, params, map_name):
        self.centre_line = CentreLine(map_name)
        self.cth_speed_weight = params.cth_speed_weight
        self.cth_distance_weight = params.cth_distance_weight
        self.cth_max_speed = params.cth_max_speed 

    def __call__(self, observation, prev_obs, action):        
        if observation['lap_complete']:
            self.previous_progress = 0
            return 1  # complete
        if observation['collision']:
            self.previous_progress = 0
            return -1 # crash
        
        state = observation['vehicle_state']
        pose = self.centre_line.calculate_pose(observation['progress'])

        distance = np.linalg.norm(pose[:2] - state[:2])
        d_heading = abs(robust_angle_difference_rad(pose[2], state[4]))

        r_heading  = (state[3] / self.cth_max_speed) * np.cos(d_heading) * self.cth_speed_weight 
        r_distance = distance * self.cth_distance_weight 

        reward = max(r_heading - r_distance, 0)
        return reward
    

def robust_angle_difference_rad(x, y):
    """Returns the difference between two angles in RADIANS
    r = x - y"""
    return np.arctan2(np.sin(x-y), np.cos(x-y))


