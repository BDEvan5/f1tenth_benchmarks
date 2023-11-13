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
    train_map = "mco"
    # train_map = "gbr" # test_id is the training map.
    algorithm = "TD3"
    # algorithm = "SAC"
    training_steps = 80000

    training_agent = TrainingAgent(train_map, train_map, algorithm)
    simulator = F1TenthSim_TrueLocation(train_map, training_agent.name, train_map, False, True)

    run_training_loop_steps(simulator, training_agent, training_steps)
    plot_drl_training(training_agent.name, train_map)

    map_list = ["aut", "esp", "gbr", "mco"]
    testing_agent = TestingAgent(train_map, algorithm)
    for map_name in map_list:
        print(f"Testing on {map_name}...")
        simulator = F1TenthSim(map_name, testing_agent.name, train_map)
        run_simulation_loop_laps(simulator, testing_agent, 5)



train_and_test_agents()


