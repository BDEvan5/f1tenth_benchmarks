from f1tenth_sim.classic_racing.GlobalPurePursuit import GlobalPurePursuit
from f1tenth_sim.classic_racing.GlobalMPCC2 import GlobalMPCC2
from f1tenth_sim.classic_racing.ConstantMPCC import ConstantMPCC
# from f1tenth_sim.classic_racing.GlobalMPCC import GlobalMPCC

from f1tenth_sim.data_tools.general_plotting.plot_trajectory_analysis import plot_trajectory_analysis
from f1tenth_sim.data_tools.general_plotting.plot_raceline_tracking import plot_raceline_tracking

from f1tenth_sim.run_scripts.run_functions import *


def test_pure_pursuit_planning():
    # test_id = "mu50"
    test_id = "mu70"
    map_name = "aut"
    planner = GlobalPurePursuit(test_id, False, planner_name="GlobalPlanPP")
    # test_planning_single_map(planner, map_name, test_id)
    test_planning_all_maps(planner, test_id)

    plot_trajectory_analysis(planner.name, test_id)
    plot_raceline_tracking(planner.name, test_id)


def test_full_stack_pure_pursuit():
    test_id = "mu70"
    # map_name = "aut"
    map_name = "mco"
    planner = GlobalPurePursuit(test_id, False, planner_name="FullStackPP")
    test_full_stack_all_maps(planner, test_id)
    # test_full_stack_single_map(planner, map_name, test_id)

    plot_trajectory_analysis(planner.name, test_id)
    plot_raceline_tracking(planner.name, test_id)


# test_pure_pursuit_planning()
test_full_stack_pure_pursuit()


