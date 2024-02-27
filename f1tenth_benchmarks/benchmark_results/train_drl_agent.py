import numpy as np 
import torch
from f1tenth_benchmarks.run_scripts.run_functions import *
from f1tenth_benchmarks.drl_racing.EndToEndAgent import EndToEndAgent, TrainEndToEndAgent
from f1tenth_benchmarks.data_tools.specific_plotting.plot_drl_training import plot_drl_training



def run_reward_tests():
    train_maps = ["mco", "gbr", "esp", "aut"]
    seeds = [12, 13, 14]
    rewards = ["TAL", "Progress", "CTH"]
    for train_map in train_maps:
        for seed in seeds:
            for reward in rewards:
                seed_randomness(seed)
                test_id = f"TD3_{reward}_{seed}_{train_map}"
                print(f"Training agent: {test_id}")
                training_agent = TrainEndToEndAgent(train_map, test_id, extra_params={'reward': reward})
                simulate_training_steps(training_agent, train_map, test_id, extra_params={'n_sim_steps': 10})
                plot_drl_training(training_agent.name, test_id)

                testing_agent = EndToEndAgent(test_id)
                test_mapless_all_maps(testing_agent, test_id, number_of_laps=10)


run_reward_tests()