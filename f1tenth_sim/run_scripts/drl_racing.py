from f1tenth_sim.run_scripts.run_functions import *
import numpy as np 
import torch
from f1tenth_sim.simulator import F1TenthSim_TrueLocation, F1TenthSim
from f1tenth_sim.drl_racing.EndToEndAgent import EndToEndAgent, TrainEndToEndAgent

from f1tenth_sim.data_tools.specific_plotting.plot_drl_training import plot_drl_training


def seed_randomness(random_seed):
    np.random.seed(random_seed)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(random_seed)
    

def train_drl_agent():
    seed_randomness(10)
    train_map = "mco"
    test_id = "TestTD3"

    training_agent = TrainEndToEndAgent(train_map, test_id)
    simulate_training_steps(training_agent, train_map, test_id)
    plot_drl_training(training_agent.name, test_id)

def test_drl_agent():
    seed_randomness(10)
    test_id = "TestTD3"
    testing_agent = EndToEndAgent(test_id)
    test_mapless_all_maps(testing_agent, test_id)



def train_and_test_agents():
    seed_randomness(11)
    train_map = "mco"
    # test_id = "SACv2"
    # test_id = "TD3v1"
    test_id = "SACv1"
    # test_id = "TD3rewardP1"

    training_agent = TrainEndToEndAgent(train_map, test_id, extra_params={'reward': "progress"})
    simulate_training_steps(training_agent, train_map, test_id)
    plot_drl_training(training_agent.name, test_id)

    testing_agent = EndToEndAgent(test_id)
    test_mapless_all_maps(testing_agent, test_id)



def run_reward_tests():
    # rewards = ["CTH", "TAL"]
    seeds = [12, 13, 14]
    rewards = ["Progress", "CTH", "TAL"]
    train_maps = ["mco", "gbr", "esp", "aut"]
    for train_map in train_maps:
        for seed in seeds:
            for reward in rewards:
                seed_randomness(seed)
                test_id = f"TD3_{reward}_{seed}_{train_map}"
                # training_agent = TrainEndToEndAgent(train_map, test_id, extra_params={'reward': reward})
                # simulate_training_steps(training_agent, train_map, test_id)
                # plot_drl_training(training_agent.name, test_id)

                testing_agent = EndToEndAgent(test_id)
                test_mapless_all_maps(testing_agent, test_id, number_of_laps=5)


# train_and_test_agents()
# test_drl_agent()


run_reward_tests()