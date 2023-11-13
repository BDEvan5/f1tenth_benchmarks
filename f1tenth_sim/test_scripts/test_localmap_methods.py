from f1tenth_sim.simulator import F1TenthSim_TrueLocation  
from f1tenth_sim.localmap_racing.LocalMapPlanner import LocalMapPlanner

from f1tenth_sim.data_tools.calculations.calculate_tracking_accuracy import calculate_tracking_accuracy
from f1tenth_sim.data_tools.general_plotting.plot_raceline_tracking import plot_raceline_tracking
from f1tenth_sim.data_tools.general_plotting.plot_trajectory import plot_analysis

from f1tenth_sim.test_scripts.testing_utils import *


def run_planning_tests_all_maps(planner, test_id):
    map_list = ["aut", "esp", "gbr", "mco"]
    for map_name in map_list:
        print(f"Testing on {map_name}...")
        simulator = F1TenthSim_TrueLocation(map_name, planner.name, test_id)
        planner.set_map(map_name)
        run_simulation_loop_laps(simulator, planner, 1)

    calculate_tracking_accuracy(planner.name)
    plot_analysis(planner.name, test_id)
    plot_raceline_tracking(planner.name, test_id)


def run_planning_test_single(planner, map_name, test_id):
    print(f"Testing on {map_name}...")
    simulator = F1TenthSim_TrueLocation(map_name, planner.name, test_id)
    planner.set_map(map_name)
    run_simulation_loop_laps(simulator, planner, 1)

    # calculate_tracking_accuracy(planner.name)
    plot_analysis(planner.name, test_id)
    # plot_raceline_tracking(planner.name, test_id)


def test__localmap_planner():
    test_id = "c1"
    map_name = "aut"
    planner = LocalMapPlanner(test_id, save_data=True, raceline=False)
    # planner = LocalMapPlanner(test_id, save_data=True, raceline=True)
    run_planning_test_single(planner, map_name, test_id)
    # run_planning_tests_all_maps(planner, test_id)





if __name__ == "__main__":

    test__localmap_planner()
    # test_mpcc()




