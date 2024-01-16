from f1tenth_sim.simulator import F1TenthSim_TrueLocation, F1TenthSim
from f1tenth_sim.classic_racing.particle_filter import ParticleFilter

from f1tenth_sim.data_tools.general_plotting.plot_trajectory_analysis import plot_trajectory_analysis
from f1tenth_sim.data_tools.general_plotting.plot_raceline_tracking import plot_raceline_tracking


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


def simulate_training_steps(planner, train_map, test_id):
    sim = F1TenthSim_TrueLocation(train_map, planner.name, test_id, False, True)
    observation, done, init_pose = sim.reset()
    
    for i in range(planner.planner_params.training_steps):
        action = planner.plan(observation)
        observation, done = sim.step(action)
        if done:
            planner.done_callback(observation)
            observation, done, init_pose = sim.reset()


# map_list = ["aut", "esp", "gbr", "mco"]
map_list = ["aut", "esp", "gbr"]

def test_planning_all_maps(planner, test_id):
    # map_list = ["esp", "gbr", "mco"]
    for map_name in map_list:
        test_planning_single_map(planner, map_name, test_id)

def test_planning_single_map(planner, map_name, test_id):
    print(f"Testing on {map_name}...")
    simulator = F1TenthSim_TrueLocation(map_name, planner.name, test_id)
    planner.set_map(map_name)
    simulate_laps(simulator, planner, NUMBER_OF_LAPS)



def test_full_stack_all_maps(planner, test_id):
    for map_name in map_list:
        test_full_stack_single_map(planner, map_name, test_id)

def test_full_stack_single_map(planner, map_name, test_id):
    print(f"Testing on {map_name}...")
    pf = ParticleFilter(planner.name, test_id)
    simulator = F1TenthSim(map_name, planner.name, test_id)
    planner.set_map(map_name)
    pf.set_map(map_name)
    simulate_localisation_laps(simulator, planner, pf, NUMBER_OF_LAPS)



def test_mapless_all_maps(planner, test_id):
    for map_name in map_list:
        test_mapless_single_map(planner, map_name, test_id)

def test_mapless_single_map(planner, map_name, test_id):
    print(f"Testing on {map_name}...")
    simulator = F1TenthSim(map_name, planner.name, test_id)
    simulate_laps(simulator, planner, NUMBER_OF_LAPS)

