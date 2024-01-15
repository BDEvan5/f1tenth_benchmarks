import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from f1tenth_sim.data_tools.plotting_utils import *

def make_error_particle_plot():
    results = pd.read_csv(f"logs/PerceptionTesting/Results_PerceptionTesting.csv")
    results = results.sort_values(by="TestID")
    errors = results["MeanError"].values
    particles = results["TestID"].values

    plt.figure(1, figsize=(5, 2))
    plt.plot(particles, errors, 'o', color=periwinkle)
    plt.fill_between(particles, np.zeros_like(errors), errors, alpha=0.3, color=periwinkle)

    plt.xlabel("Number of particles")
    plt.ylabel("Mean error (cm)")

    plt.grid(True)
    plt.savefig(f"logs/BenchmarkArticle/PerceptionTesting_particle_errors.svg", pad_inches=0, bbox_inches='tight')
    plt.savefig(f"logs/BenchmarkArticle/PerceptionTesting_particle_errors.pdf", pad_inches=0, bbox_inches='tight')

def make_time_particle_plot():
    results = pd.read_csv(f"logs/PerceptionTesting/Results_PerceptionTesting.csv")
    results = results.sort_values(by="TestID")
    particles = results["TestID"].values
    map_name =  "aut"

    times = []
    for test_id in particles:
        computation_df = pd.read_csv(f"logs/PerceptionTesting/RawData_{test_id}/Profile_{map_name}_{test_id}.csv")
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
    # make_error_particle_plot()
    make_time_particle_plot()



