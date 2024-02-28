from f1tenth_benchmarks.classic_racing.GlobalPurePursuit import GlobalPurePursuit

from f1tenth_benchmarks.data_tools.specific_plotting.plot_pf_errors import plot_pf_errors
from f1tenth_benchmarks.data_tools.plot_trajectory_analysis import plot_trajectory_analysis
from f1tenth_benchmarks.data_tools.plot_raceline_tracking import plot_raceline_tracking

from f1tenth_benchmarks.run_scripts.run_functions import *
import numpy as np

def test_pure_pursuit_planning():
    test_id = "planning_pp"
    map_name = "aut"
    planner = GlobalPurePursuit(test_id, False, planner_name="GlobalPlanPP")
    test_planning_single_map(planner, map_name, test_id)
    # test_planning_all_maps(planner, test_id)

    plot_trajectory_analysis(planner.name, test_id)
    # plot_raceline_tracking(planner.name, test_id)



def test_full_stack_pure_pursuit():
    test_id = "full_stack_pp"
    map_name = "aut"
    planner = GlobalPurePursuit(test_id, False, planner_name="FullStackPP", extra_params={"racetrack_set": "mu90"})
    # test_full_stack_all_maps(planner, test_id, number_of_laps=5)
    test_full_stack_single_map(planner, map_name, test_id, number_of_laps=5)

    plot_trajectory_analysis(planner.name, test_id)
    # plot_raceline_tracking(planner.name, test_id)



test_pure_pursuit_planning()
test_full_stack_pure_pursuit()
