import numpy as np
np.printoptions(precision=2, suppress=True)

from f1tenth_sim.classic_racing.GlobalPurePursuit import GlobalPurePursuit


class TrajectoryAidedLearningReward:
    def __init__(self, map_name, params):
        self.pp = GlobalPurePursuit("_drl_training", init_folder=False)
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
        # print(f"PP: {pp_act}, Action: {action}, Diff: {weighted_difference}")
        reward = self.beta_c * max(1 - np.sum(weighted_difference), 0)

        return reward


class ProgressReward:
    def __init__(self, params):
        pass
        self.previous_progress = 0

        self.beta_c = params.progress_constant
        
    def __call__(self, observation, prev_obs, action):
        if self.previous_progress == 0: return 0

        if observation['lap_complete']:
            self.previous_progress = 0
            return 1  # complete
        if observation['collision']:
            self.previous_progress = 0
            return -1 # crash
        
        progress_diff = observation['progress'] - self.previous_progress
        reward = self.beta_c * progress_diff

        return reward



