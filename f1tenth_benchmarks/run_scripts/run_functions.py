from f1tenth_benchmarks.simulator import F1TenthSim_TrueLocation, F1TenthSim
from f1tenth_benchmarks.classic_racing.particle_filter import ParticleFilter
import torch
import numpy as np


NUMBER_OF_LAPS = 1

def simulate_laps(sim, planner, n_laps):
    for lap in range(n_laps):
        observation, done, init_pose = sim.reset()
        while not done:
            action = planner.plan(observation)
            observation, done = sim.step(action)

def simulate_localisation_laps(sim, planner, pf, n_laps):
    for lap in range(n_laps):
        observation, done, init_pose = sim.reset()
        observation['pose'] = pf.init_pose(init_pose)
        while not done:
            action = planner.plan(observation)
            observation, done = sim.step(action)
            observation['pose'] = pf.localise(action, observation)
        pf.lap_complete()


def simulate_training_steps(planner, train_map, test_id, extra_params={}):
    sim = F1TenthSim_TrueLocation(train_map, planner.name, test_id, False, True, extra_params=extra_params)
    observation, done, init_pose = sim.reset()
    
    for i in range(planner.planner_params.training_steps):
        action = planner.plan(observation)
        observation, done = sim.step(action)
        if done:
            planner.done_callback(observation)
            observation, done, init_pose = sim.reset()


map_list = ["aut", "esp", "gbr", "mco"]
# map_list = ["aut", "esp", "gbr"]

def test_planning_all_maps(planner, test_id, extra_params={}, number_of_laps=NUMBER_OF_LAPS):
    # map_list = ["esp", "gbr", "mco"]
    for map_name in map_list:
        test_planning_single_map(planner, map_name, test_id, extra_params=extra_params, number_of_laps=number_of_laps)

def test_planning_single_map(planner, map_name, test_id, extra_params={}, number_of_laps=NUMBER_OF_LAPS):
    print(f"Testing on {map_name}...")
    simulator = F1TenthSim_TrueLocation(map_name, planner.name, test_id, extra_params=extra_params)
    planner.set_map(map_name)
    simulate_laps(simulator, planner, number_of_laps)



def test_full_stack_all_maps(planner, test_id, extra_params={}, number_of_laps=NUMBER_OF_LAPS, extra_pf_params={}):
    for map_name in map_list:
        test_full_stack_single_map(planner, map_name, test_id, extra_params=extra_params, number_of_laps=number_of_laps, extra_pf_params=extra_pf_params)

def test_full_stack_single_map(planner, map_name, test_id, extra_params={}, number_of_laps=NUMBER_OF_LAPS, extra_pf_params={}):
    print(f"Testing on {map_name}...")
    simulator = F1TenthSim(map_name, planner.name, test_id, extra_params=extra_params)
    planner.set_map(map_name)
    extra_pf_params["dt"] = simulator.params.timestep * simulator.params.n_sim_steps
    pf = ParticleFilter(planner.name, test_id, extra_pf_params)
    # pf = ParticleFilter(planner.name, test_id, {"dt": simulator.params.timestep * simulator.params.n_sim_steps})
    pf.set_map(map_name)
    simulate_localisation_laps(simulator, planner, pf, number_of_laps)



def test_mapless_all_maps(planner, test_id, extra_params={}, number_of_laps=NUMBER_OF_LAPS):
    for map_name in map_list:
        test_mapless_single_map(planner, map_name, test_id, extra_params, number_of_laps=number_of_laps)

def test_mapless_single_map(planner, map_name, test_id, extra_params={}, number_of_laps=NUMBER_OF_LAPS):
    print(f"Testing on {map_name}...")
    simulator = F1TenthSim(map_name, planner.name, test_id, extra_params=extra_params)
    simulate_laps(simulator, planner, number_of_laps)



def seed_randomness(random_seed):
    np.random.seed(random_seed)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(random_seed)
    

