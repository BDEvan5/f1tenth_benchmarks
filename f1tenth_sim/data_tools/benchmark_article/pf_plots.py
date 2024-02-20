import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from f1tenth_sim.data_tools.plotting_utils import *

def make_error_particle_plot():
    results = pd.read_csv(f"Logs/PerceptionTesting/Results_PerceptionTesting.csv")
    results = results.sort_values(by="TestID")
    errors = results["MeanError"].values
    particles = results["TestID"].values

    plt.figure(1, figsize=(5, 2))
    plt.plot(particles, errors, 'o', color=periwinkle)
    plt.fill_between(particles, np.zeros_like(errors), errors, alpha=0.3, color=periwinkle)

    plt.xlabel("Number of particles")
    plt.ylabel("Mean error (cm)")

    plt.grid(True)
    plt.savefig(f"Data/BenchmarkArticle/PerceptionTesting_particle_errors.svg", pad_inches=0, bbox_inches='tight')
    plt.savefig(f"Data/BenchmarkArticle/PerceptionTesting_particle_errors.pdf", pad_inches=0, bbox_inches='tight')

def make_error_particle_plot_maps():
    results = pd.read_csv(f"Logs/PerceptionTesting/Results_PerceptionTesting.csv")
    n_particles = [50, 100, 200, 300, 400, 500, 600, 800, 1000, 1200, 1500]
    test_ids = [f"t2_{n}" for n in n_particles]

    results = results[results["TestID"].isin(test_ids)]
    results['n_particles'] = results['TestID'].apply(lambda x: int(x.split('_')[1]))
    results = results.sort_values(by="n_particles")
    errors = results["MeanError"].values
    particles = results["n_particles"].values
    map_list = results["TestMap"].unique()

    plt.figure(1, figsize=(5, 2))
    for map_name in map_list:
        map_errors = errors[results["TestMap"] == map_name]
        map_particles = particles[results["TestMap"] == map_name]
        plt.plot(map_particles, map_errors, '-o', label=map_name.upper())
    # plt.plot(particles, errors, 'o', color=periwinkle)
    # plt.fill_between(particles, np.zeros_like(errors), errors, alpha=0.3, color=periwinkle)

    plt.legend()
    plt.xlabel("Number of particles")
    plt.ylabel("Mean error (cm)")

    plt.grid(True)
    plt.savefig(f"Data/BenchmarkArticle/PerceptionTesting_particle_errors.svg", pad_inches=0, bbox_inches='tight')
    plt.savefig(f"Data/BenchmarkArticle/PerceptionTesting_particle_errors.pdf", pad_inches=0, bbox_inches='tight')

def record_localisation_times():
    results = pd.read_csv(f"Logs/PerceptionTesting/Results_PerceptionTesting.csv")
    entry_ids = results["EntryID"].dropna().unique()
    print(entry_ids)

    for entry_id in entry_ids:
        entry = results.loc[results["EntryID"] == entry_id]
        print(entry)
        test_id = entry["TestID"].values[0]
        map_name = entry["TestMap"].values[0]
        # test_id = results.loc[results["EntryID"] == entry_id, "TestID"].values[0]
        # map_name = results.loc[results["EntryID"] == entry_id, "TestMap"].values[0]
        computation_df = pd.read_csv(f"Logs/PerceptionTesting/RawData_{test_id}/Profile_{map_name}_{test_id}.csv")
        time = computation_df.loc[computation_df.func == 'localise', 'percall_cumtime'].values[0]
        results.at[entry.index[0], "LocalisationTime"] = time

    results.to_csv(f"Logs/PerceptionTesting/Results_PerceptionTesting.csv", index=False)


def make_error_particle_plot_maps_times():
    results = pd.read_csv(f"Logs/PerceptionTesting/Results_PerceptionTesting.csv")
    n_particles = [50, 100, 300, 600, 1000, 1400]
    # n_particles = [50, 100, 200, 400, 600, 800, 1000, 1200, 1400]
    test_ids = [f"t2_{n}" for n in n_particles]

    results = results[results["TestID"].isin(test_ids)]
    results['n_particles'] = results['TestID'].apply(lambda x: int(x.split('_')[1]))
    results = results.sort_values(by="n_particles")
    # errors = results["MeanError"].values
    # particles = results["n_particles"].values
    map_list = results["TestMap"].unique()

    # plt.figure(1, figsize=(5, 2))
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

# def make_error_particle_plot_maps():
#     results = pd.read_csv(f"Logs/PerceptionTesting/Results_PerceptionTesting.csv")
#     n_particles = [50, 100, 200, 400, 600, 800, 1000, 1200, 1400]
#     # n_particles = [20, 50, 100, 250, 500, 800, 1000, 1200, 1500]
#     # n_particles = [50, 100, 200, 300, 400, 500, 600, 800, 1000, 1200, 1500, 2000]
#     test_ids = [f"t2_{n}" for n in n_particles]

#     results = results[results["TestID"].isin(test_ids)]
#     results['n_particles'] = results['TestID'].apply(lambda x: int(x.split('_')[1]))
#     results = results.sort_values(by="n_particles")
#     errors = results["MeanError"].values
#     particles = results["n_particles"].values
#     map_list = results["TestMap"].unique()

#     plt.figure(1, figsize=(5, 2))
#     for map_name in map_list:
#         map_errors = errors[results["TestMap"] == map_name]
#         map_particles = particles[results["TestMap"] == map_name]
#         plt.plot(map_particles, map_errors, '-o', label=map_name.upper())
#     # plt.plot(particles, errors, 'o', color=periwinkle)
#     # plt.fill_between(particles, np.zeros_like(errors), errors, alpha=0.3, color=periwinkle)

#     plt.legend()
#     plt.xlabel("Number of particles")
#     plt.ylabel("Mean error (cm)")

#     plt.grid(True)
#     plt.savefig(f"Data/BenchmarkArticle/PerceptionTesting_particle_errors.svg", pad_inches=0, bbox_inches='tight')
#     plt.savefig(f"Data/BenchmarkArticle/PerceptionTesting_particle_errors.pdf", pad_inches=0, bbox_inches='tight')

def make_time_particle_plot():
    results = pd.read_csv(f"Logs/PerceptionTesting/Results_PerceptionTesting.csv")
    results = results.sort_values(by="TestID")
    particles = results["TestID"].values
    map_name =  "aut"

    times = []
    for test_id in particles:
        computation_df = pd.read_csv(f"Logs/PerceptionTesting/RawData_{test_id}/Profile_{map_name}_{test_id}.csv")
        time = computation_df.loc[computation_df.func == 'localise', 'cumtime'].values[0]
        times.append(time)

    plt.figure(1, figsize=(5, 2))
    plt.plot(particles, times, 'o', color=periwinkle)
    plt.fill_between(particles, np.zeros_like(times), times, alpha=0.3, color=periwinkle)

    plt.xlabel("Number of particles")
    plt.ylabel("Mean error (cm)")

    plt.grid(True)
    plt.savefig(f"Data/BenchmarkArticle/PerceptionTesting_particle_times.svg", pad_inches=0, bbox_inches='tight')
    plt.savefig(f"Data/BenchmarkArticle/PerceptionTesting_particle_times.pdf", pad_inches=0, bbox_inches='tight')


if __name__ == "__main__":
    # record_localisation_times()
    # make_error_particle_plot_maps()
    make_error_particle_plot_maps_times()
    # make_error_particle_plot()
    # make_time_particle_plot()



