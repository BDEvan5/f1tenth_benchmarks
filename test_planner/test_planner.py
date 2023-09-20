from f1tenth_sim import F1TenthSim 
from Planner import Planner
import numpy as np
import yaml 
from argparse import Namespace

def load_configuration(config_name):
    with open(f"configurations/{config_name}.yaml", 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    run_dict = Namespace(**config)
    return run_dict 

def run_simulation_loop_laps(env, planner, n_laps):
    for lap in range(n_laps):
        observation, done = env.reset(poses=np.array([0, 0, 0]))
        while not done:
            action = planner.plan(observation)
            observation, done = env.step(action)
    

def run_test():
    map_name = "aut"
    std_config = load_configuration("std_config")

    simulator = F1TenthSim(map_name, std_config)
    planner = Planner(simulator.map_name)

    run_simulation_loop_laps(simulator, planner, 1)


if __name__ == "__main__":
    run_test()





