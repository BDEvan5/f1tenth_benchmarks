from f1tenth_sim.localmap_racing.LocalMapPP import LocalMapPP
from f1tenth_sim.localmap_racing.LocalMPCC import LocalMPCC
from f1tenth_sim.run_scripts.run_functions import *


def localmap_pp_centre():
    test_id = "c1"
    # test_id = "mu70"
    map_name = "aut"
    # map_name = "esp"
    # map_name = "gbr"
    planner = LocalMapPP(test_id, True, False)
    test_planning_single_map(planner, map_name, test_id)
    # test_planning_all_maps(planner, test_id)

    # plot_trajectory_analysis(planner.name, test_id)
    # plot_raceline_tracking(planner.name, test_id)

def test_localmap_pp():
    # test_id = "test"
    # test_id = "centre"
    test_id = "mu60"
    # test_id = "mu70"
    map_name = "aut"
    planner = LocalMapPP(test_id, True, True)
    # test_planning_single_map(planner, map_name, test_id)
    test_planning_all_maps(planner, test_id)

    plot_trajectory_analysis(planner.name, test_id)
    # plot_raceline_tracking(planner.name, test_id)


def test_localmap_mpcc():
    test_id = "mu60"
    # test_id = "mu70"
    map_name = "aut"
    planner = LocalMPCC(test_id, True)
    # test_planning_single_map(planner, map_name, test_id)
    test_planning_all_maps(planner, test_id)

    plot_trajectory_analysis(planner.name, test_id)
    # plot_raceline_tracking(planner.name, test_id)





if __name__ == "__main__":
    # localmap_pp_centre()
    test_localmap_pp()
    # test_localmap_mpcc()





