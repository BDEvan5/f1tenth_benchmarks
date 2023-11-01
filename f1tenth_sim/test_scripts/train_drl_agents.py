from testing_utils import *
import numpy as np 
import torch
from f1tenth_sim.simulator import F1TenthSim_TrueLocation, F1TenthSim
from f1tenth_sim.mapless_racing.agents import TrainingAgent, TestingAgent
    
def seed_randomness(random_seed):
    np.random.seed(random_seed)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(random_seed)
    


def train_agents():
    seed_randomness(1)
    map_name = "mco"
    algorithm = "TD3"
    # algorithm = "SAC"
    agent_name = f"{algorithm}_endToEnd_2"
    training_steps = 40000

    simulator = F1TenthSim_TrueLocation(map_name, agent_name, False, True)
    training_agent = TrainingAgent(map_name, agent_name, algorithm)

    run_training_loop_steps(simulator, training_agent, training_steps)

    map_list = ["aut", "esp", "gbr", "mco"]
    testing_agent = TestingAgent(agent_name)
    for map_name in map_list:
        print(f"Testing on {map_name}...")
        simulator = F1TenthSim(map_name, agent_name)
        run_simulation_loop_laps(simulator, testing_agent, 2)

train_agents()


