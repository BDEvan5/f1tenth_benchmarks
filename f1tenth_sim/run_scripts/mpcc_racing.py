# from f1tenth_sim.classic_racing.GlobalMPCC2 import GlobalMPCC2
from f1tenth_sim.classic_racing.ConstantMPCC import ConstantMPCC
from f1tenth_sim.classic_racing.GlobalMPCC import GlobalMPCC

from f1tenth_sim.data_tools.general_plotting.plot_trajectory_analysis import plot_trajectory_analysis
from f1tenth_sim.data_tools.general_plotting.plot_raceline_tracking import plot_raceline_tracking

from f1tenth_sim.run_scripts.run_functions import *



def test_constant_mpcc_planning():
    test_id = "mu70"
    map_name = "esp"
    # map_name = "aut"
    planner = ConstantMPCC(test_id, False, planner_name="ConstantMPCC")
    test_planning_single_map(planner, map_name, test_id, {"n_sim_steps": 10})
    # test_planning_all_maps(planner, test_id, {"n_sim_steps": 10})

    plot_trajectory_analysis(planner.name, test_id)

def test_mpcc_planning():
    # test_id = "mu70"
    max_speed = 8
    # test_id = f"max{max_speed}"
    # test_id = "params_t2"
    test_id = "t2"
    # map_name = "aut"
    map_name = "mco"
    # map_name = "gbr"
    # map_name = "esp"
    planner = GlobalMPCC(test_id, False, planner_name="GlobalPlanMPCC", extra_params={"max_speed": max_speed})
    # test_planning_single_map(planner, map_name, test_id, number_of_laps=3)
    test_planning_all_maps(planner, test_id, number_of_laps=5)


    plot_trajectory_analysis(planner.name, test_id)

def test_full_stack_mpcc():
    map_name = "aut"

    test_id = "mpcc_t1"
    planner = GlobalMPCC(test_id, True, planner_name="FullStackMPCC")
    test_planning_single_map(planner, map_name, test_id, number_of_laps=3)
    # test_full_stack_all_maps(planner, test_id, number_of_laps=5)

    plot_trajectory_analysis(planner.name, test_id)





if __name__ == "__main__":
    # test_constant_mpcc_planning()
    # test_mpcc_planning()
    test_full_stack_mpcc()




