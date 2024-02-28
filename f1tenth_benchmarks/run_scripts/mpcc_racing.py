# from f1tenth_benchmarks.classic_racing.GlobalMPCC2 import GlobalMPCC2
from f1tenth_benchmarks.classic_racing.ConstantMPCC import ConstantMPCC
from f1tenth_benchmarks.classic_racing.GlobalMPCC import GlobalMPCC

from f1tenth_benchmarks.data_tools.plot_trajectory_analysis import plot_trajectory_analysis
from f1tenth_benchmarks.data_tools.plot_raceline_tracking import plot_raceline_tracking

from f1tenth_benchmarks.run_scripts.run_functions import *



def test_constant_mpcc_planning():
    test_id = "constant_mpcc"
    map_name = "esp"
    planner = ConstantMPCC(test_id, False, planner_name="ConstantMPCC")
    test_planning_single_map(planner, map_name, test_id, {"n_sim_steps": 10})
    # test_planning_all_maps(planner, test_id, {"n_sim_steps": 10})

    plot_trajectory_analysis(planner.name, test_id)

def test_mpcc_planning():
    max_speed = 8
    test_id = "planning_mpcc"
    map_name = "mco"
    planner = GlobalMPCC(test_id, False, planner_name="GlobalPlanMPCC", extra_params={"max_speed": max_speed})
    test_planning_single_map(planner, map_name, test_id, number_of_laps=1)
    # test_planning_all_maps(planner, test_id, number_of_laps=5)


    plot_trajectory_analysis(planner.name, test_id)

def test_full_stack_mpcc():
    map_name = "gbr"

    test_id = "full_stack_mpcc"
    planner = GlobalMPCC(test_id, True, planner_name="FullStackMPCC")
    # test_planning_all_maps(planner, test_id, number_of_laps=1)
    test_planning_single_map(planner, map_name, test_id, number_of_laps=1)

    plot_trajectory_analysis(planner.name, test_id)





if __name__ == "__main__":
    test_constant_mpcc_planning()
    test_mpcc_planning()
    test_full_stack_mpcc()





