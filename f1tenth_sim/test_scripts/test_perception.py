from f1tenth_sim.simulator import F1TenthSim_TrueLocation 
from f1tenth_sim.classic_racing.GlobalPurePursuit import GlobalPurePursuit
from f1tenth_sim.classic_racing.particle_filter import ParticleFilter
import numpy as np
from f1tenth_sim.data_tools.general_plotting.plot_trajectory import plot_analysis
from f1tenth_sim.data_tools.specific_plotting.plot_pf_errors import plot_pf_errors

def run_simulation_loop_laps(sim, planner, pf, n_laps):
    for lap in range(n_laps):
        observation, done, init_pose = sim.reset()
        observation['pose'] = pf.init_pose(init_pose)
        while not done:
            action = planner.plan(observation)
            observation, done = sim.step(action)
            observation['pose'] = pf.localise(action, observation)
        pf.lap_complete()
    sim.__del__()


def run_planning_tests(planner):
    map_list = ["aut", "esp", "gbr", "mco"]
    for map_name in map_list:
        print(f"Testing on {map_name}...")
        simulator = F1TenthSim_TrueLocation(map_name, planner.name)
        planner.set_map(map_name)
        run_simulation_loop_laps(simulator, planner, 1)


def test_pf_perception():
    tuning_map = "aut"
    perception_name = "PerceptionTesting"
    n_particles = 100
    test_id = f"{n_particles}"
    simulator = F1TenthSim_TrueLocation(tuning_map, perception_name, test_id)
    planner = GlobalPurePursuit(test_id)
    pf_localisation = ParticleFilter(perception_name, n_particles)
    
    planner.set_map_centerline(tuning_map)
    pf_localisation.set_map(tuning_map)
    run_simulation_loop_laps(simulator, planner, pf_localisation, 1)

    plot_pf_errors(perception_name, test_id)

    
def test_single_perception_config(tuning_map, perception_name, n_particles, test_id):
    simulator = F1TenthSim_TrueLocation(tuning_map, perception_name, test_id)
    planner = GlobalPurePursuit(test_id)
    pf_localisation = ParticleFilter(perception_name, n_particles)
        
    planner.set_map_centerline(tuning_map)
    pf_localisation.set_map(tuning_map)
    run_simulation_loop_laps(simulator, planner, pf_localisation, 1)

    plot_pf_errors(perception_name, test_id)


def test_pf_perception():
    tuning_map = "aut"
    perception_name = "PerceptionTesting"
    # for n_particles in [25, 50, 75, 100, 150, 250, 350, 500]:
    for n_particles in [75, 100, 150]:
    # for n_particles in [50]:
    # for n_particles in [250]:
        test_id = f"{n_particles}"
        test_single_perception_config(tuning_map, perception_name, n_particles, test_id)







if __name__ == "__main__":
    test_pf_perception()



