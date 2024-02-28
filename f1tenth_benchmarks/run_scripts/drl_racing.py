from f1tenth_benchmarks.run_scripts.run_functions import *
import numpy as np 
import torch
from f1tenth_benchmarks.simulator import F1TenthSim_TrueLocation, F1TenthSim
from f1tenth_benchmarks.drl_racing.EndToEndAgent import EndToEndAgent, TrainEndToEndAgent

from f1tenth_benchmarks.data_tools.specific_plotting.plot_drl_training import plot_drl_training


def seed_randomness(random_seed):
    np.random.seed(random_seed)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(random_seed)
    



def train_and_test_agents():
    seed_randomness(11)
    train_map = "mco"
    test_id = "SACv1"

    training_agent = TrainEndToEndAgent(train_map, test_id, extra_params={'reward': "progress"})
    simulate_training_steps(training_agent, train_map, test_id)
    plot_drl_training(training_agent.name, test_id)

    testing_agent = EndToEndAgent(test_id)
    test_mapless_all_maps(testing_agent, test_id)




train_and_test_agents()
