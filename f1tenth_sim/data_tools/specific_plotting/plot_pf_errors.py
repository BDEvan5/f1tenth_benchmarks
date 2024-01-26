import matplotlib.pyplot as plt
import numpy as np
from f1tenth_sim.data_tools.plotting_utils import *
import os
import pandas as pd

def plot_pf_errors(vehicle_name="PerceptionTesting", test_id="100"):
    results = pd.read_csv(f"Logs/{vehicle_name}/Results_{vehicle_name}.csv")
    ind = results.loc[results["TestID"] == int(test_id)].index[0]
    map_name = results.loc[ind, "TestMap"]

    pf_estimates = np.load(f"Logs/{vehicle_name}/RawData_{test_id}/pf_estimates_{test_id}.npy")
    history = np.load(f"Logs/{vehicle_name}/RawData_{test_id}/SimLog_{map_name}_0.npy")
    save_path = f"Logs/{vehicle_name}/Images/"
    save_path_pdf = f"Logs/{vehicle_name}/Images_pdf/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(save_path_pdf):
        os.makedirs(save_path_pdf)
    true_locations = history[1:, 0:2]
    differences = pf_estimates[:-1, :2] - true_locations
    errors = np.linalg.norm(differences, axis=1) * 100
    print(f"Mean error: {np.mean(errors)} cm")

    results.at[ind, "MeanError"] = np.mean(errors)

    results.to_csv(f"Logs/{vehicle_name}/Results_{vehicle_name}.csv", index=False)

    plt.figure(1, figsize=(5, 2))
    plt.plot(errors, color=periwinkle)
    plt.fill_between(np.arange(len(errors)), errors, alpha=0.5, color=periwinkle)
    plt.xlabel("Time step")
    plt.ylabel("Error (cm)")
    plt.title("Particle Filter Error")
    plt.grid(True)

    plt.savefig(f"{save_path}pf_errors_{map_name}_{test_id}.svg")
    plt.savefig(f"{save_path_pdf}pf_errors_{map_name}_{test_id}.pdf")

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

    plt.savefig(f"{save_path}pf_errors_hist_{map_name}_{test_id}.svg", pad_inches=0.01, bbox_inches='tight')
    plt.savefig(f"{save_path_pdf}pf_errors_hist_{map_name}_{test_id}.pdf", pad_inches=0.01, bbox_inches='tight')


if __name__ == "__main__":
    # plot_pf_errors()
    plot_pf_errors("PerceptionTesting", "25")

