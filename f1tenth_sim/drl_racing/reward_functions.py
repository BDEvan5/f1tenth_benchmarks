import numpy as np

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
        rewards = np.abs(pp_act - action) / self.weights 
        reward = self.beta_c * max(1 - np.sum(rewards), 0)

        return reward



