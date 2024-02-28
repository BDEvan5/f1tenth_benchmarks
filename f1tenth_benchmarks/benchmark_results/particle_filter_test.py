from f1tenth_benchmarks.classic_racing.GlobalPurePursuit import GlobalPurePursuit
from f1tenth_benchmarks.data_tools.specific_plotting.plot_pf_errors import plot_pf_errors

from f1tenth_benchmarks.run_scripts.run_functions import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def evaluate_filter_particles_pure_pursuit():
    n_particles = [50, 100, 300, 600, 1000, 1400]

    for n in n_particles:
        test_id = f"t2_{n}"
        print(f"Testing {test_id}...")
        planner = GlobalPurePursuit(test_id, True, planner_name="PerceptionTesting")
        test_full_stack_all_maps(planner, test_id, extra_pf_params={"number_of_particles": n})

        plot_pf_errors("PerceptionTesting", test_id)


def make_error_particle_plot_maps_times():
    results = pd.read_csv(f"Logs/PerceptionTesting/Results_PerceptionTesting.csv")
    n_particles = [50, 100, 300, 600, 1000, 1400]
    test_ids = [f"t2_{n}" for n in n_particles]

    results = results[results["TestID"].isin(test_ids)]
    results['n_particles'] = results['TestID'].apply(lambda x: int(x.split('_')[1]))
    results = results.sort_values(by="n_particles")
    map_list = results["TestMap"].unique()

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6, 1.9))
    for map_name in map_list:
        map_df = results[results["TestMap"] == map_name]
        map_errors = map_df["MeanError"].values
        map_particles = map_df["n_particles"].values
        map_times = map_df["LocalisationTime"].values * 1000

        a1.plot(map_particles, map_errors, '-o', label=map_name.upper())
        a2.plot(map_particles, map_times, '-o', label=map_name.upper())

    a2.legend(ncol=2, fontsize=8)
    a1.set_xlabel("Number of particles")
    a2.set_xlabel("Number of particles")
    a1.set_ylabel("Mean error (cm)")
    a2.set_ylabel("Localisation \ntime (ms)")
    a1.yaxis.set_major_locator(plt.MaxNLocator(4))
    a1.xaxis.set_major_locator(plt.MaxNLocator(5))
    a2.xaxis.set_major_locator(plt.MaxNLocator(5))
    a2.set_ylim(0, 20)

    a1.grid(True)
    a2.grid(True)
    plt.tight_layout()
    plt.savefig(f"Data/BenchmarkArticle/PerceptionTesting_particle_errors_and_times.svg", pad_inches=0, bbox_inches='tight')
    plt.savefig(f"Data/BenchmarkArticle/PerceptionTesting_particle_errors_and_times.pdf", pad_inches=0, bbox_inches='tight')


evaluate_filter_particles_pure_pursuit()
make_error_particle_plot_maps_times()


