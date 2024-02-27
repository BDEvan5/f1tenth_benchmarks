# from f1tenth_benchmarks.classic_racing.GlobalMPCC2 import GlobalMPCC2
from f1tenth_benchmarks.classic_racing.ConstantMPCC import ConstantMPCC
from f1tenth_benchmarks.classic_racing.GlobalMPCC import GlobalMPCC

from f1tenth_benchmarks.data_tools.general_plotting.plot_trajectory_analysis import plot_trajectory_analysis
from f1tenth_benchmarks.data_tools.general_plotting.plot_raceline_tracking import plot_raceline_tracking

from f1tenth_benchmarks.run_scripts.run_functions import *




def run_planning_mpcc_tests():
    friction_mus = [0.5, 0.6, 0.7, 0.8, 0.9, 1]
    # friction_mus = [0.8, 0.9, 1]

    for mu in friction_mus:
        test_id = f"mu{int(mu*100)}_steps4"
        planner = GlobalMPCC(test_id, False, planner_name="GlobalPlanMPCC", extra_params={"friction_mu": mu})
        test_planning_single_map(planner, "aut", test_id, number_of_laps=10)
        # test_planning_all_maps(planner, test_id, number_of_laps=10)


def run_full_stack_mpcc_tests():
    # friction_mus = [0.7]
    friction_mus = [0.5, 0.6, 0.7, 0.8, 0.9, 1]
    # friction_mus = [0.5, 0.6, 0.7, 0.8]

    for mu in friction_mus:
        test_id = f"mu{int(mu*100)}_steps4"
        planner = GlobalMPCC(test_id, False, planner_name="FullStackMPCC", extra_params={"friction_mu": mu})
        # test_full_stack_all_maps(planner, test_id, number_of_laps=10)
        test_full_stack_single_map(planner, "aut", test_id, number_of_laps=10)






if __name__ == "__main__":
    run_planning_mpcc_tests()
    run_full_stack_mpcc_tests()


