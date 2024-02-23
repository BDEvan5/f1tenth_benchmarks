from f1tenth_sim.classic_racing.GlobalPurePursuit import GlobalPurePursuit

from f1tenth_sim.data_tools.specific_plotting.plot_pf_errors import plot_pf_errors
from f1tenth_sim.data_tools.general_plotting.plot_trajectory_analysis import plot_trajectory_analysis
from f1tenth_sim.data_tools.general_plotting.plot_raceline_tracking import plot_raceline_tracking

from f1tenth_sim.run_scripts.run_functions import *
import numpy as np

def test_pure_pursuit_planning():
    # test_id = "mu50"
    test_id = "mu70"
    map_name = "aut"
    planner = GlobalPurePursuit(test_id, False, planner_name="GlobalPlanPP")
    # test_planning_single_map(planner, map_name, test_id)
    test_planning_all_maps(planner, test_id)

    plot_trajectory_analysis(planner.name, test_id)
    plot_raceline_tracking(planner.name, test_id)

def test_pure_pursuit_planning_frequencies():
    # friction_vals = np.linspace(0.8, 1, 5)
    # friction_vals = [0.7, 0.75]
    friction_vals = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
    # friction_vals = np.linspace(0.55, 1, 10)
    # simulator_timestep_list = [1, 2, 5, 8, 10, 12]
    # simulator_timestep_list = [1, 2, 5, 10, 20]
    simulator_timestep_list = [4, 6]
    for simulator_timesteps in simulator_timestep_list:
        for friction in friction_vals:
            test_id = f"mu{int(friction*100)}_steps{simulator_timesteps}"
            print(f"Testing {test_id}...")
            planner = GlobalPurePursuit(test_id, False, planner_name="GlobalPlanPP", extra_params={"racetrack_set": f"mu{int(friction*100)}"})
            test_planning_all_maps(planner, test_id, {"n_sim_steps": simulator_timesteps}, 10)

    # plot_trajectory_analysis(planner.name, test_id)
    # plot_raceline_tracking(planner.name, test_id)


def test_full_stack_pure_pursuit_planning_frequencies():
    map_name = "aut"
    # friction_vals = [0.7]
    # simulator_timestep_list = [2]
    friction_vals = [0.7, 0.8, 0.9, 1]
    simulator_timestep_list = [2, 4, 6, 8, 10, 12, 14]
    for simulator_timesteps in simulator_timestep_list:
        for friction in friction_vals:
            test_id = f"mu{int(friction*100)}_steps{simulator_timesteps}"
            print(f"Testing {test_id}...")
            planner = GlobalPurePursuit(test_id, False, planner_name="FullStackPP300", extra_params={"racetrack_set": f"mu{int(friction*100)}"})
            test_full_stack_single_map(planner, map_name, test_id, {"n_sim_steps": simulator_timesteps}, 10, {"number_of_particles": 300})

    plot_trajectory_analysis(planner.name, test_id)
    plot_raceline_tracking(planner.name, test_id)



def test_full_stack_pure_pursuit():
    test_id = "ts_t1"
    map_name = "aut"
    # map_name = "mco"
    planner = GlobalPurePursuit(test_id, False, planner_name="FullStackPP", extra_params={"racetrack_set": "mu90"})
    # test_full_stack_all_maps(planner, test_id, number_of_laps=5)
    test_full_stack_single_map(planner, map_name, test_id, number_of_laps=5)

    plot_trajectory_analysis(planner.name, test_id)
    # plot_raceline_tracking(planner.name, test_id)


def test_particle_filter_pure_pursuit():
    test_id = "pf_t2"
    planner = GlobalPurePursuit(test_id, True, planner_name="PerceptionTesting", extra_params={"racetrack_set": test_id})
    test_full_stack_all_maps(planner, test_id, number_of_laps=5)
    # test_full_stack_single_map(planner, map_name, test_id)

    plot_trajectory_analysis(planner.name, test_id)
    # plot_raceline_tracking(planner.name, test_id)


def evaluate_particle_filter_particles_pure_pursuit():
    # n_particles = [50, 100, 200, 400]
    n_particles = [750]

    for n in n_particles:
        test_id = f"t2_{n}"
        print(f"Testing {test_id}...")
        planner = GlobalPurePursuit(test_id, True, planner_name="PerceptionTesting")
        test_full_stack_all_maps(planner, test_id, extra_pf_params={"number_of_particles": n})

        plot_pf_errors("PerceptionTesting", test_id)
        # plot_trajectory_analysis(planner.name, test_id)


# test_pure_pursuit_planning()
test_full_stack_pure_pursuit()
# test_pure_pursuit_planning_frequencies()
# test_full_stack_pure_pursuit_planning_frequencies()
# test_particle_filter_pure_pursuit()
# evaluate_particle_filter_particles_pure_pursuit()

