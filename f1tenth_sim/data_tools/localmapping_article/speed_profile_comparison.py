import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 
from matplotlib.collections import LineCollection
from f1tenth_sim.data_tools.plotting_utils import *
from f1tenth_sim.utils.MapData import MapData


def plot_speed_profiles(planner1, planner2, test_id, map_name, test_lap, name=""):
    save_path = f"Data/LocalMapRacing/"
    
    root = f"Logs/{planner1}/"
    Logs1 = np.load(root + f"RawData_{test_id}/SimLog_{map_name}_{test_lap}.npy")
    root = f"Logs/{planner2}/"
    Logs2 = np.load(root + f"RawData_{test_id}/SimLog_{map_name}_{test_lap}.npy")


    plt.figure(figsize=(6, 3))
    # plt.figure(figsize=(5, 2.2))
    plt.clf()

    ax1 = plt.subplot(2, 1, 1)
    # ax2 = plt.subplot(2, 1, 2)

    ax1.plot(Logs1[:, 9]*100, Logs1[:, 3], color=sunset_orange, linewidth=3, label="Global")
    ax1.plot(Logs2[:, 9]*100, Logs2[:, 3], color=periwinkle, linewidth=3, label="Local")

    ax1.grid(True)
    ax1.legend(ncol=2, fontsize=9)
    ax1.set_xlabel("Track progress [%]", fontsize=9)
    ax1.set_ylabel("Speed [m/s]", fontsize=9)
    # ax1.set_xticks(np.arange(0, 101, 20), fontsize=4)
    ax1.set_xticklabels(np.arange(-20, 101, 20), fontsize=9)
    ax1.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax1.set_ylim(0, 8.8)

    name = f"SpeedComparison_{test_id}_{map_name.upper()}_{test_lap}"
    plt.rcParams['pdf.use14corefonts'] = True

    plt.savefig(save_path + name + ".svg", bbox_inches="tight", pad_inches=0)
    plt.savefig(save_path + name + ".pdf", bbox_inches="tight", pad_inches=0)


def plot_speed_profiles_with_difference(planner1, planner2, test_id, map_name, test_lap, name=""):
    save_path = f"Data/LocalMapRacing/"
    
    root = f"Logs/{planner1}/"
    Logs1 = np.load(root + f"RawData_{test_id}/SimLog_{map_name}_{test_lap}.npy")
    root = f"Logs/{planner2}/"
    Logs2 = np.load(root + f"RawData_{test_id}/SimLog_{map_name}_{test_lap}.npy")


    # plt.figure(figsize=(6, 3))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 3), sharex=True)
    # plt.figure(figsize=(5, 2.2))
    # plt.clf()

    # ax1 = plt.subplot(2, 1, 1)
    # ax2 = plt.subplot(2, 1, 2)

    ax1.plot(Logs1[:, 9]*100, Logs1[:, 3], color=sunset_orange, linewidth=3, label="Global")
    ax1.plot(Logs2[:, 9]*100, Logs2[:, 3], color=periwinkle, linewidth=3, label="Local")

    ax1.grid(True)
    ax2.grid(True)
    ax1.legend(ncol=2, fontsize=9)
    ax2.set_xlabel("Track progress [%]", fontsize=9)
    ax1.set_ylabel("Speed [m/s]", fontsize=9)
    ax2.set_ylabel("Difference [m/s]", fontsize=9)
    # ax1.set_xticks(np.arange(0, 101, 20), fontsize=4)
    ax1.set_xticklabels(np.arange(-20, 101, 20), fontsize=9)
    ax1.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax1.set_ylim(0, 8.8)
    ax2.set_ylim(-3.7, 3.7)

    new_x = np.linspace(0, 1, 1000)
    new_y1 = np.interp(new_x, Logs1[:, 9], Logs1[:, 3])
    new_y2 = np.interp(new_x, Logs2[:, 9], Logs2[:, 3])
    ax2.fill_between(new_x*100, new_y1 - new_y2, color=sweedish_green, linewidth=3)

    # ax2.plot(Logs1[:, 9]*100, Logs1[:, 3] - Logs2[:, 3], color=periwinkle, linewidth=3)

    name = f"SpeedComparison_{test_id}_{map_name.upper()}_{test_lap}"
    plt.rcParams['pdf.use14corefonts'] = True

    plt.savefig(save_path + name + ".svg", bbox_inches="tight", pad_inches=0)
    plt.savefig(save_path + name + ".pdf", bbox_inches="tight", pad_inches=0)







map_name = "aut"
# map_name = "esp"
lap_n = 0

plot_speed_profiles_with_difference("FullStackPP", "LocalMapPP", "mu60", map_name, lap_n)
# plot_speed_profiles("FullStackPP", "LocalMapPP", "mu60", map_name, lap_n)



