import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from f1tenth_sim.data_tools.plotting_utils import *


def true_moving_average(data, period):
    if len(data) < period:
        return np.zeros_like(data)
    ret = np.convolve(data, np.ones(period), 'same') / period
    for i in range(period): # start
        t = np.convolve(data, np.ones(i+2), 'valid') / (i+2)
        ret[i] = t[0]
    for i in range(period):
        length = int(round((i + period)/2))
        t = np.convolve(data, np.ones(length), 'valid') / length
        ret[-i-1] = t[-1]
    return ret

def plot_drl_training(planner_name, test_id):
    #TODO; this plotting must be changed to be per steps, not per episode as per current.

    fig = plt.figure(figsize=(10, 4))
    a1 = plt.subplot(1, 2, 1)
    a2 = plt.subplot(1, 2, 2)

    a1.set_title("Reward")
    a2.set_title("Track progress")

    a1.set_xlabel("Training Steps")
    a2.set_xlabel("Training Steps")

    results = pd.read_csv(f"Logs/{planner_name}/RawData_{test_id}/TrainingData_{test_id}.csv")
    steps = results["Steps"]
    progresses = results["Progress"]*100
    a2.plot(steps, progresses, color=periwinkle)
    a2.plot(steps, true_moving_average(progresses, 20), color=sunset_orange)

    rewards = np.load(f"Logs/{planner_name}/RawData_{test_id}/RewardHistory.npy")
    a1.plot(steps, rewards, color=periwinkle)
    a1.plot(steps, true_moving_average(rewards, 20), color=sunset_orange)

    a1.grid(True)
    a2.grid(True)

    plt.tight_layout()
    plt.savefig(f"Logs/{planner_name}/TrainingProgress.svg")



if __name__ == "__main__":
    plot_drl_training(f"TD3_endToEnd", "v1")
    # plot_drl_training(f"SAC_endToEnd_{n}")


