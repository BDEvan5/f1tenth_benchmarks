from f1tenth_sim.classic_racing.GlobalPurePursuit import GlobalPurePursuit

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
    friction_vals = np.linspace(0.55, 1, 10)
    # simulator_timestep_list = [1, 2, 5, 10, 20]
    simulator_timestep_list = [14]
    for simulator_timesteps in simulator_timestep_list:
        for friction in friction_vals:
            test_id = f"mu{int(friction*100)}_steps{simulator_timesteps}"
            print(f"Testing {test_id}...")
            planner = GlobalPurePursuit(test_id, False, planner_name="GlobalPlanPP", extra_params={"racetrack_set": f"mu{int(friction*100)}"})
            test_planning_all_maps(planner, test_id, {"n_sim_steps": simulator_timesteps})

    # plot_trajectory_analysis(planner.name, test_id)
    # plot_raceline_tracking(planner.name, test_id)


def test_full_stack_pure_pursuit():
    test_id = "mu70"
    # map_name = "aut"
    map_name = "mco"
    planner = GlobalPurePursuit(test_id, False, planner_name="FullStackPP")
    test_full_stack_all_maps(planner, test_id)
    # test_full_stack_single_map(planner, map_name, test_id)

    plot_trajectory_analysis(planner.name, test_id)
    plot_raceline_tracking(planner.name, test_id)


test_pure_pursuit_planning()
# test_full_stack_pure_pursuit()
# test_pure_pursuit_planning_frequencies()


