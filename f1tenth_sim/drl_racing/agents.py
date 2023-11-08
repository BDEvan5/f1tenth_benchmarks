import numpy as np
from f1tenth_sim.mapless_racing.sac import TrainSAC, TestSAC
from f1tenth_sim.mapless_racing.td3 import TrainTD3, TestTD3
from f1tenth_sim.mapless_racing.reward_functions import TrajectoryAidedLearningReward


def create_train_agent(state_dim, algorithm):
    action_dim = 2
    if algorithm == "TD3":
        agent = TrainTD3(state_dim, action_dim)
    elif algorithm == "SAC":
        agent = TrainSAC(state_dim, action_dim)
    else: raise ValueError(f"Algorithm {algorithm} not recognised")
    
    return agent
    
def create_test_agent(filename, directory):
    algorithm = filename.split("_")[0]
    if algorithm == "TD3":
        agent = TestTD3(filename, directory)
    elif algorithm == "SAC":
        agent = TestSAC(filename, directory)
    else: raise ValueError(f"Algorithm {algorithm} not recognised")
    
    return agent
    

MAX_SPEED = 8
MAX_STEER = 0.4
NUMBER_OF_BEAMS = 20
RANGE_FINDER_SCALE = 10


class EndToEndAgent:
    def __init__(self):
        self.range_finder_scale = 10
        self.skip_n = int(1080 / NUMBER_OF_BEAMS)

        self.state_space = NUMBER_OF_BEAMS *2 + 1 
        self.n_scans = 2
        self.scan_buffer = np.zeros((self.n_scans, NUMBER_OF_BEAMS))

    def transform_obs(self, obs):
        """
        Transforms the observation received from the environment into a vector which can be used with a neural network.
    
        Args:
            obs: observation from sim

        Returns:
            nn_obs: observation vector for neural network
        """
        speed = obs['vehicle_speed'] / MAX_SPEED
        scan = np.clip(obs['scan'][::self.skip_n] /RANGE_FINDER_SCALE, 0, 1)

        if self.scan_buffer.all() ==0: # first reading
            for i in range(self.n_scans):
                self.scan_buffer[i, :] = scan 
        else:
            self.scan_buffer = np.roll(self.scan_buffer, 1, axis=0)
            self.scan_buffer[0, :] = scan

        dual_scan = np.reshape(self.scan_buffer, (NUMBER_OF_BEAMS * self.n_scans))
        nn_obs = np.concatenate((dual_scan, [speed]))

        return nn_obs

    def transform_action(self, nn_action):
        steering_angle = nn_action[0] * MAX_STEER
        speed = (nn_action[1] + 1) * (MAX_SPEED  / 2 - 0.5) + 1
        speed = min(speed, MAX_SPEED) # cap the speed

        action = np.array([steering_angle, speed])

        return action


class TrainingAgent(EndToEndAgent): 
    def __init__(self, map_name, test_id, algorithm="TD3"):
        super().__init__()
        self.name = f"{algorithm}_endToEnd"
        self.path = f"Logs/{self.name}/RawData_{test_id}/"

        self.reward_generator = TrajectoryAidedLearningReward(map_name)
        self.state = None
        self.nn_state = None
        self.nn_act = None
        self.action = None

        self.agent = create_train_agent(self.state_space, algorithm)
        self.current_ep_reward = 0
        self.reward_history = []

    def plan(self, obs):
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

        np.save(self.path + "RewardHistory.npy", self.reward_history)
        self.agent.save(self.name, self.path)




class TestingAgent(EndToEndAgent): 
    def __init__(self, agent_name):
        super().__init__()
        self.path = f"Logs/{agent_name}/"
        self.architecture = EndToEndAgent()
        self.agent = create_test_agent(agent_name, self.path)
        
    def plan(self, obs):
        nn_state = self.transform_obs(obs)
        
        if obs['vehicle_speed'] < 1:
            return np.array([0, 2])

        self.nn_act = self.agent.act(nn_state)
        self.action = self.transform_action(self.nn_act)
        
        return self.action 


