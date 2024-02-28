from f1tenth_benchmarks.localmap_racing.LocalMapPP import LocalMapPP
from f1tenth_benchmarks.localmap_racing.LocalMPCC import LocalMPCC
from f1tenth_benchmarks.run_scripts.run_functions import *
from f1tenth_benchmarks.data_tools.plot_trajectory_analysis import plot_trajectory_analysis


def localmap_pp_centre():
    test_id = "c1"
    map_name = "aut"
    planner = LocalMapPP(test_id, True, False)
    test_planning_single_map(planner, map_name, test_id)
    # test_planning_all_maps(planner, test_id)

    plot_trajectory_analysis(planner.name, test_id)

def test_localmap_pp():
    test_id = "mu60"
    map_name = "aut"
    planner = LocalMapPP(test_id, True, True)
    # test_planning_single_map(planner, map_name, test_id)
    test_planning_all_maps(planner, test_id)

    plot_trajectory_analysis(planner.name, test_id)


def test_localmap_mpcc():
    test_id = "mu60"
    map_name = "gbr"
    planner = LocalMPCC(test_id, True)
    test_planning_single_map(planner, map_name, test_id)
    # test_planning_all_maps(planner, test_id)

    plot_trajectory_analysis(planner.name, test_id)





if __name__ == "__main__":
    # localmap_pp_centre()
    # test_localmap_pp()
    test_localmap_mpcc()





