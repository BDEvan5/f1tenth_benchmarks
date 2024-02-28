from f1tenth_benchmarks.classic_racing.GlobalPurePursuit import GlobalPurePursuit

from f1tenth_benchmarks.data_tools.specific_plotting.plot_pf_errors import plot_pf_errors
from f1tenth_benchmarks.data_tools.general_plotting.plot_trajectory_analysis import plot_trajectory_analysis
from f1tenth_benchmarks.data_tools.general_plotting.plot_raceline_tracking import plot_raceline_tracking

from f1tenth_benchmarks.run_scripts.run_functions import *
import numpy as np



def evaluate_filter_particles_pure_pursuit():
    n_particles = [50, 100, 300, 600, 1000, 1400]

    for n in n_particles:
        test_id = f"t2_{n}"
        print(f"Testing {test_id}...")
        planner = GlobalPurePursuit(test_id, True, planner_name="PerceptionTesting")
        test_full_stack_all_maps(planner, test_id, extra_pf_params={"number_of_particles": n})

        plot_pf_errors("PerceptionTesting", test_id)



evaluate_filter_particles_pure_pursuit()