from testing_utils import *
import numpy as np 
import torch
from f1tenth_sim.simulator import F1TenthSim_TrueLocation
from f1tenth_sim.mapless_racing.agents import TrainingAgent, TestingAgent
    
def seed_randomness(random_seed):
    np.random.seed(random_seed)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(random_seed)
    


def train_agents():
    seed_randomness(10)
    map_name = "aut"
    # algorithm = "TD3"
    algorithm = "SAC"
    agent_name = f"{algorithm}_endToEnd_1"
    training_steps = 30000

    simulator = F1TenthSim_TrueLocation(map_name, agent_name, False)
    training_agent = TrainingAgent(map_name, agent_name, algorithm)

    run_training_loop_steps(simulator, training_agent, training_steps)


train_agents()


