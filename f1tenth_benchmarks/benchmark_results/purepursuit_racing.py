from f1tenth_benchmarks.classic_racing.GlobalPurePursuit import GlobalPurePursuit

from f1tenth_benchmarks.data_tools.specific_plotting.plot_pf_errors import plot_pf_errors
from f1tenth_benchmarks.data_tools.general_plotting.plot_trajectory_analysis import plot_trajectory_analysis
from f1tenth_benchmarks.data_tools.general_plotting.plot_raceline_tracking import plot_raceline_tracking

from f1tenth_benchmarks.run_scripts.run_functions import *
import numpy as np



def test_pure_pursuit_planning_frequencies():
    map_name = "aut"
    friction_vals = [0.7, 0.8, 0.9, 1]
    simulator_timestep_list = [2, 4, 6, 8, 10, 12, 14]
    for simulator_timesteps in simulator_timestep_list:
        for friction in friction_vals:
            test_id = f"mu{int(friction*100)}_steps{simulator_timesteps}"
            print(f"Testing {test_id}...")
            planner = GlobalPurePursuit(test_id, False, planner_name="GlobalPlanPP", extra_params={"racetrack_set": f"mu{int(friction*100)}"})
            test_planning_single_map(planner, map_name, test_id, {"n_sim_steps": simulator_timesteps}, 10)

def test_pure_pursuit_full_stack_frequencies():
    map_name = "aut"
    friction_vals = [0.7, 0.8, 0.9, 1]
    simulator_timestep_list = [2, 4, 6, 8, 10, 12, 14]
    for simulator_timesteps in simulator_timestep_list:
        for friction in friction_vals:
            test_id = f"mu{int(friction*100)}_steps{simulator_timesteps}"
            print(f"Testing {test_id}...")
            planner = GlobalPurePursuit(test_id, False, planner_name="FullStackPP1000", extra_params={"racetrack_set": f"mu{int(friction*100)}"})
            test_full_stack_single_map(planner, map_name, test_id, {"n_sim_steps": simulator_timesteps}, 10)
            # test_full_stack_single_map(planner, map_name, test_id, {"n_sim_steps": simulator_timesteps}, 10, {"number_of_particles": 300})



def evaluate_filter_particles_pure_pursuit():
    n_particles = [50, 100, 300, 600, 1000, 1400]

    for n in n_particles:
        test_id = f"t2_{n}"
        print(f"Testing {test_id}...")
        planner = GlobalPurePursuit(test_id, True, planner_name="PerceptionTesting")
        test_full_stack_all_maps(planner, test_id, extra_pf_params={"number_of_particles": n})

        plot_pf_errors("PerceptionTesting", test_id)



# test_pure_pursuit_full_stack_frequencies()
test_pure_pursuit_planning_frequencies()
# evaluate_filter_particles_pure_pursuit()