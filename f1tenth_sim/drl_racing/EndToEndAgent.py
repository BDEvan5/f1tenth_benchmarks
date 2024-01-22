import numpy as np
from f1tenth_sim.drl_racing.sac import TrainSAC, TestSAC
from f1tenth_sim.drl_racing.td3 import TrainTD3, TestTD3
from f1tenth_sim.drl_racing.reward_functions import TrajectoryAidedLearningReward, ProgressReward
from f1tenth_sim.utils.BasePlanner import BasePlanner, save_params
import torch

def create_train_agent(state_dim, algorithm):
    action_dim = 2
    if algorithm == "TD3":
        agent = TrainTD3(state_dim, action_dim)
    elif algorithm == "SAC":
        agent = TrainSAC(state_dim, action_dim)
    else: raise ValueError(f"Algorithm {algorithm} not recognised")
    
    return agent
    
# def create_test_agent(filename, directory):
#     algorithm = filename[0:3]
#     if algorithm == "TD3":
#         agent = TestTD3(filename, directory)
#     elif algorithm == "SAC":
#         agent = TestSAC(filename, directory)
#     else: raise ValueError(f"Algorithm {algorithm} not recognised")
    
#     return agent
    




class EndToEndAgent(BasePlanner):
    def __init__(self, test_id):
        BasePlanner.__init__(self, "EndToEnd", test_id)
        self.range_finder_scale = 10

        self.skip_n = int(1080 / self.planner_params.number_of_beams)
        self.state_space = self.planner_params.number_of_beams *2 + 1 
        self.scan_buffer = np.zeros((self.planner_params.n_scans, self.planner_params.number_of_beams))
        self.actor = torch.load(self.data_root_path + f'{self.name}_actor.pth')
        
    def plan(self, obs):
        nn_state = self.transform_obs(obs)
        
        if obs['vehicle_speed'] < 1:
            return np.array([0, 2])

        self.nn_act = self.actor.test_action(nn_state)
        self.action = self.transform_action(self.nn_act)
        
        return self.action 
    
    def transform_obs(self, obs):
        """
        Transforms the observation received from the environment into a vector which can be used with a neural network.
    
        Args:
            obs: observation from sim

        Returns:
            nn_obs: observation vector for neural network
        """
        speed = obs['vehicle_speed'] / self.planner_params.max_speed
        scan = np.clip(obs['scan'][::self.skip_n] /self.planner_params.range_finder_scale, 0, 1)

        if self.scan_buffer.all() ==0: # first reading
            for i in range(self.scan_buffer.shape[0]):
                self.scan_buffer[i, :] = scan 
        else:
            self.scan_buffer = np.roll(self.scan_buffer, 1, axis=0)
            self.scan_buffer[0, :] = scan

        dual_scan = np.reshape(self.scan_buffer, (-1))
        nn_obs = np.concatenate((dual_scan, [speed]))

        return nn_obs

    def transform_action(self, nn_action):
        steering_angle = nn_action[0] * self.vehicle_params.max_steer
        speed = (nn_action[1] + 1) * (self.vehicle_params.max_speed  / 2 - 0.5) + 1
        speed = min(speed, self.planner_params.max_speed) # cap the speed for the planner

        action = np.array([steering_angle, speed])

        return action
    


class TrainEndToEndAgent(EndToEndAgent): 
    def __init__(self, map_name, test_id):
        BasePlanner.__init__(self, "EndToEnd", test_id) #NOTE: do not call the inherited __init__()

        self.reward_generator = ProgressReward(self.planner_params)
        # self.reward_generator = TrajectoryAidedLearningReward(map_name, self.planner_params) 
        # save_params(self.reward_generator.pp.planner_params, self.data_root_path, "pp_params")
        self.state = None
        self.nn_state = None
        self.nn_act = None
        self.action = None

        self.skip_n = int(1080 / self.planner_params.number_of_beams)
        self.state_space = self.planner_params.number_of_beams *2 + 1 
        self.scan_buffer = np.zeros((self.planner_params.n_scans, self.planner_params.number_of_beams))

        self.agent = create_train_agent(self.state_space, self.planner_params.algorithm)
        self.current_ep_reward = 0
        self.reward_history = []

    def plan(self, obs): # This overwrites the above plan method
        nn_state = self.transform_obs(obs)
        
        self.add_memory_entry(obs, nn_state)
        self.state = obs
            
        if obs["vehicle_speed"] < 1:
            self.action = np.array([0, 2])
            return self.action

        self.nn_state = nn_state 
        self.nn_act = self.agent.act(self.nn_state)
        self.action = self.transform_action(self.nn_act)
        
        self.agent.train()

        return self.action 

    def add_memory_entry(self, s_prime, nn_s_prime):
        if self.nn_state is not None:
            reward = self.reward_generator(s_prime, self.state, self.action)
            self.current_ep_reward += reward

            self.agent.replay_buffer.add(self.nn_state, self.nn_act, nn_s_prime, reward, False)

    def done_callback(self, s_prime):
        """
        To be called when ep is done.
        """
        nn_s_prime = self.transform_obs(s_prime)
        reward = self.reward_generator(s_prime, self.state, self.action)
        self.agent.replay_buffer.add(self.nn_state, self.nn_act, nn_s_prime, reward, True)
        
        self.reward_history.append(self.current_ep_reward+reward)
        self.current_ep_reward = 0
        self.nn_state = None
        self.state = None

        np.save(self.data_root_path + "RewardHistory.npy", self.reward_history)
        self.agent.save(self.name, self.data_root_path)





