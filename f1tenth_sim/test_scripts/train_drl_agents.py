from testing_utils import *
import numpy as np 
import torch
from f1tenth_sim.simulator import F1TenthSim_TrueLocation, F1TenthSim
from f1tenth_sim.drl_racing.agents import TrainingAgent, TestingAgent
from f1tenth_sim.data_tools.specific_plotting.plot_drl_training import plot_drl_training


def seed_randomness(random_seed):
    np.random.seed(random_seed)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(random_seed)
    


def train_and_test_agents():
    seed_randomness(10)
    # map_name = "mco"
    map_name = "gbr"
    algorithm = "TD3"
    # algorithm = "SAC"
    test_id = "v2"
    training_steps = 80000

    training_agent = TrainingAgent(map_name, test_id, algorithm)
    simulator = F1TenthSim_TrueLocation(map_name, training_agent.name, test_id, False, True)

    run_training_loop_steps(simulator, training_agent, training_steps)
    plot_drl_training(training_agent.name, test_id)

    map_list = ["aut", "esp", "gbr", "mco"]
    testing_agent = TestingAgent(test_id, algorithm)
    for map_name in map_list:
        print(f"Testing on {map_name}...")
        simulator = F1TenthSim(map_name, testing_agent.name, test_id)
        run_simulation_loop_laps(simulator, testing_agent, 5)



train_and_test_agents()


