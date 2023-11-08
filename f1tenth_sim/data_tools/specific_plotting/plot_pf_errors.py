import matplotlib.pyplot as plt
import numpy as np
from f1tenth_sim.data_tools.plotting_utils import *


def plot_pf_errors(vehicle_name="PerceptionTesting"):
    map_name = "aut"
    pf_estimates = np.load(f"Logs/{vehicle_name}/pf_estimates.npy")
    history = np.load(f"Logs/{vehicle_name}/RawData/SimLog_{map_name}_0.npy")
    true_locations = history[1:, 0:2]
    differences = pf_estimates[:-1, :2] - true_locations
    errors = np.linalg.norm(differences, axis=1) * 100
    print(f"Mean error: {np.mean(errors)} cm")

    plt.figure(1, figsize=(5, 2))
    plt.plot(errors, color=periwinkle)
    plt.fill_between(np.arange(len(errors)), errors, alpha=0.5, color=periwinkle)
    plt.xlabel("Time step")
    plt.ylabel("Error (cm)")
    plt.title("Particle Filter Error")
    plt.grid(True)

    plt.savefig(f"Logs/{vehicle_name}/pf_errors.svg")

    plt.clf()
    N_bins = 23
    bins = np.linspace(0, 22, N_bins) -0.5
    bars = np.digitize(errors, bins)
    bars = np.bincount(bars, minlength=len(bins))
    bars = bars / np.sum(bars) * 100
    plt.bar(bins, bars[:N_bins], color=periwinkle)
    plt.xlim(0, N_bins)

    plt.plot([np.mean(errors)]*2, [0, 18], color=sunset_orange, label="Mean", linestyle="--", linewidth=2)
    plt.plot([np.median(errors)]*2, [0, 18], color=sweedish_green, label="Median", linestyle="--", linewidth=2)
    plt.legend()

    plt.xlabel("Error (cm)")
    plt.ylabel("Frequency (%)")
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(f"Logs/{vehicle_name}/pf_errors_hist.svg", pad_inches=0.01, bbox_inches='tight')


if __name__ == "__main__":
    plot_pf_errors()

