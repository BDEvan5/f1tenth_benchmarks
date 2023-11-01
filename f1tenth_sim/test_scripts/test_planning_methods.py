from f1tenth_sim.simulator import F1TenthSim_TrueLocation 
from f1tenth_sim.classic_racing.PurePursuit import PurePursuit
from f1tenth_sim.classic_racing.MPCC import MPCC
import numpy as np
from f1tenth_sim.data_tools.calculate_tracking_accuracy import calculate_tracking_accuracy
from f1tenth_sim.data_tools.plot_raceline_tracking import plot_raceline_tracking
from f1tenth_sim.data_tools.plot_trajectory import plot_analysis


def run_simulation_loop_laps(env, planner, n_laps):
    for lap in range(n_laps):
        observation, done, init_pose = env.reset()
        while not done:
            action = planner.plan(observation)
            observation, done = env.step(action)


def run_planning_tests(planner):
    # map_list = ["aut", "esp", "gbr", "mco"]
    map_list = ["aut"]
    for map_name in map_list:
        print(f"Testing on {map_name}...")
        simulator = F1TenthSim_TrueLocation(map_name, planner.name)
        planner.set_map(map_name)
        run_simulation_loop_laps(simulator, planner, 1)

    calculate_tracking_accuracy(planner.name)
    plot_analysis(planner.name)
    plot_raceline_tracking(planner.name)







if __name__ == "__main__":
    # run_planning_tests(PurePursuit())
    run_planning_tests(MPCC())

    # run_tuning_tests()
    # run_tuning_tests2()



