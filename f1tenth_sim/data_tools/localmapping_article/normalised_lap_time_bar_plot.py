import pandas as pd
import glob 
import numpy as np
import os
import matplotlib.pyplot as plt
from f1tenth_sim.data_tools.plotting_utils import *




def generate_bar_plot():
    summary_df = pd.read_csv("Logs/Summary.csv")

    vehicle_id = {
                  "EndToEnd_TestSAC": "End-to-end SAC", 
                  "EndToEnd_TestTD3": "End-to-end TD3", 
                  "FollowTheGap_Std": "Follow the gap", 
                  "FullStackMPCC3_mu70": "Global MPCC", 
                  "FullStackPP_mu70": "Global two-stage", 
                  "LocalMPCC_mu70": "Local MPCC",
                  "LocalMapPP_mu60": "Local two-stage", 
                  }


    results_df = summary_df.loc[summary_df.VehicleID.isin(vehicle_id)]
    results_df = results_df.replace({"VehicleID": vehicle_id})

    # norms = results_df.groupby('MapName')['AvgTime'].apply(lambda x: x / x.mean())
    # results_df.insert(3, 'NormTime', norms.values)
    results_df.insert(3, 'NormTime', results_df['AvgTime'] / results_df.groupby('MapName')['AvgTime'].transform('mean'))
    results_df.insert(4, 'NormStd', results_df['StdTime'] / results_df.groupby('MapName')['AvgTime'].transform('mean'))


    times_df = results_df.pivot(index="VehicleID", columns="MapName", values="NormTime")
    times_df = times_df.drop(columns=["mco"])
    times_df.columns = times_df.columns.str.upper()
    times_df = times_df.T

    std_df = results_df.pivot(index="VehicleID", columns="MapName", values="NormStd")
    std_df = std_df.drop(columns=["mco"])
    std_df.columns = std_df.columns.str.upper()
    std_df = std_df.T

    print(times_df)

    color_list = ["#81ecec", "#00cec9", "#0984e3", "#ffeaa7", "#fdcb6e", "#ff7675", "#d63031"]
    # color_list = [jade_dust, fresh_t, minty_green, chrome_yellow, high_pink, vibe_yellow, red_orange]
    # color_list = [jade_dust, periwinkle, minty_green, chrome_yellow, high_pink, vibe_yellow, red_orange]
    
    times_df.plot.bar(rot=0, figsize=(6, 1.8), legend=False, color=color_list, width=0.8)
    

    plt.ylabel("Lap time (s)")
    plt.xlabel("")
    std = 0.25
    plt.ylim(1-std, 1+std)
    plt.gca().yaxis.set_major_locator(plt.MaxNLocator(5))
    plt.grid(True, axis="y")

    # legend_names = ["SAC", "TD3", "FTG", "Glob. MPCC", "Glob. Ts", "Loc. MPCC", "Loc. Ts"]
    # legend_names = ["SAC", "TD3", "FTG", "Glob. MPCC", "Loc. MPCC", "Glob. Ts", "Loc. Ts"]
    legend_names = ["SAC", "TD3", "FTG", "Global MPCC", "Global Two-stage", "Local MPCC", "Local Two-stage"]
    # legend_names = ["SAC", "TD3", "FTG", "Global MPCC", "Global Ts", "Local MPCC", "Local Ts"]
    plt.legend(legend_names, ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.35), fontsize=9)
    plt.rcParams['pdf.use14corefonts'] = True

    plt.savefig("Data/LocalMapRacing/LaptimesBarPlot.svg", bbox_inches='tight', pad_inches=0)
    plt.savefig("Data/LocalMapRacing/LaptimesBarPlot.pdf", bbox_inches='tight', pad_inches=0)


generate_bar_plot()