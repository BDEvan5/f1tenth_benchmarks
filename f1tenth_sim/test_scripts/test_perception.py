from f1tenth_sim.simulator import F1TenthSim_TrueLocation 
from f1tenth_sim.classic_racing.PurePursuit import PurePursuit
from f1tenth_sim.classic_racing.particle_filter import ParticleFilter
import numpy as np
from f1tenth_sim.data_tools.plot_trajectory import plot_analysis
from f1tenth_sim.data_tools.plot_pf_errors import plot_pf_errors

def run_simulation_loop_laps(env, planner, pf, n_laps):
    for lap in range(n_laps):
        observation, done, init_pose = env.reset()
        observation['pose'] = pf.init_pose(init_pose)
        while not done:
            action = planner.plan(observation)
            observation, done = env.step(action)
            observation['pose'] = pf.localise(action, observation)
        pf.lap_complete()


def run_planning_tests(planner):
    map_list = ["aut", "esp", "gbr", "mco"]
    for map_name in map_list:
        print(f"Testing on {map_name}...")
        simulator = F1TenthSim_TrueLocation(map_name, planner.name)
        planner.set_map(map_name)
        run_simulation_loop_laps(simulator, planner, 1)


def test_pf_perception():
    tuning_map = "aut"
    name = "PerceptionTesting"
    simulator = F1TenthSim_TrueLocation(tuning_map, name)
    planner = PurePursuit()
    pf_localisation = ParticleFilter(name, 100)
    
    planner.set_map(tuning_map)
    pf_localisation.set_map(tuning_map)
    run_simulation_loop_laps(simulator, planner, pf_localisation, 1)

    plot_pf_errors(name)




if __name__ == "__main__":
    test_pf_perception()



