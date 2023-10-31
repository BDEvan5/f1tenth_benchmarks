from f1tenth_sim.simulator import PlanningF1TenthSim 
from f1tenth_sim.classic_racing.PurePursuit import PurePursuit
import numpy as np
from f1tenth_sim.data_tools.TrackingAccuracy import calculate_tracking_accuracy
from f1tenth_sim.data_tools.plot_trajectory_analysis import plot_trajectory_analysis


def run_simulation_loop_laps(env, planner, n_laps):
    for lap in range(n_laps):
        observation, done, init_pose = env.reset()
        while not done:
            action = planner.plan(observation)
            observation, done = env.step(action)


def run_planning_tests(planner):
    map_list = ["aut", "esp", "gbr", "mco"]
    for map_name in map_list:
        print(f"Testing on {map_name}...")
        simulator = PlanningF1TenthSim(map_name, planner.name)
        planner.set_map(map_name)
        run_simulation_loop_laps(simulator, planner, 1)

def run_tuning_tests():
    tuning_map = "esp"
    values = np.linspace(0.4, 2, 17)

    simulator = PlanningF1TenthSim(tuning_map, "PpTuning")
    for v in values:
        print(f"Testing with constant value {v}...")
        planner = PurePursuit()
        planner.constant_lookahead = v
        
        planner.set_map(tuning_map)
        run_simulation_loop_laps(simulator, planner, 1)

    calculate_tracking_accuracy("PpTuning")


def run_tuning_tests2():
    tuning_map = "aut"
    tuning_map = "esp"
    name = "PpTuning3"
    simulator = PlanningF1TenthSim(tuning_map, name)
    v1 = 0.5
    v2 = 0.18
    print(f"Testing with constant value {v1} and variable {v2}...")
    planner = PurePursuit()
    planner.constant_lookahead = v1
    planner.variable_lookahead = v2
    
    planner.set_map(tuning_map)
    run_simulation_loop_laps(simulator, planner, 1)

    calculate_tracking_accuracy(name)
    plot_trajectory_analysis(name)




if __name__ == "__main__":
    # run_planning_tests(PpTrajectoryFollower())

    # run_tuning_tests()
    run_tuning_tests2()



