import pandas as pd
import glob 
import numpy as np
import os
import matplotlib.pyplot as plt
from f1tenth_sim.data_tools.plotting_utils import *

def build_main_df():
    summary_df = pd.read_csv("Logs/Summary.csv")

    vehicle_id = {"FullStackPP_mu60": "Global two-stage", 
                  "FullStackMPCC_mu60": "Global MPCC", 
                  "FollowTheGap_Std": "Follow the gap", 
                  "EndToEnd_TestSAC": "End-to-end SAC", 
                  "EndToEnd_TestTD3": "End-to-end TD3", 
                  "LocalMapPP_mu60": "Local two-stage", 
                  "LocalMPCC_mu70": "Local MPCC"}


    results_df = summary_df.loc[summary_df.VehicleID.isin(vehicle_id)]
    results_df = results_df.replace({"VehicleID": vehicle_id})
    
    # print(results_df)

    times_df = results_df.pivot(index="VehicleID", columns="MapName", values="AvgTime")
    times_df.columns = times_df.columns.str.upper()
    print(times_df)

    times_df.to_latex(f"Data/LocalMapRacing/Laptimes.tex", float_format="%.2f")


def generate_bar_plot():
    summary_df = pd.read_csv("Logs/Summary.csv")

    vehicle_id = {
                  "EndToEnd_TestSAC": "End-to-end SAC", 
                  "EndToEnd_TestTD3": "End-to-end TD3", 
                  "FollowTheGap_Std": "Follow the gap", 
                  "FullStackMPCC_mu60": "Global MPCC", 
                  "FullStackPP_mu60": "Global two-stage", 
                  "LocalMPCC_mu70": "Local MPCC",
                  "LocalMapPP_mu60": "Local two-stage", 
                  }


    results_df = summary_df.loc[summary_df.VehicleID.isin(vehicle_id)]
    results_df = results_df.replace({"VehicleID": vehicle_id})

    # norms = results_df.groupby('MapName')['AvgTime'].apply(lambda x: x / x.mean())
    # results_df.insert(3, 'NormTime', norms.values)
    results_df.insert(3, 'NormTime', results_df['AvgTime'] / results_df.groupby('MapName')['AvgTime'].transform('mean'))


    times_df = results_df.pivot(index="VehicleID", columns="MapName", values="NormTime")
    times_df = times_df.drop(columns=["mco"])
    times_df.columns = times_df.columns.str.upper()
    times_df = times_df.T

    print(times_df)

    color_list = ["#81ecec", "#00cec9", "#0984e3", "#ffeaa7", "#fdcb6e", "#ff7675", "#d63031"]
    # color_list = [jade_dust, fresh_t, minty_green, chrome_yellow, high_pink, vibe_yellow, red_orange]
    # color_list = [jade_dust, periwinkle, minty_green, chrome_yellow, high_pink, vibe_yellow, red_orange]
    
    times_df.plot.bar(rot=0, figsize=(6, 1.8), legend=False, color=color_list, width=0.8)

    plt.ylabel("Lap time (s)")
    plt.xlabel("")
    std = 0.2
    plt.ylim(1-std, 1+std)
    plt.gca().yaxis.set_major_locator(plt.MaxNLocator(4))
    plt.grid(True, axis="y")

    # legend_names = ["SAC", "TD3", "FTG", "Glob. MPCC", "Glob. Ts", "Loc. MPCC", "Loc. Ts"]
    # legend_names = ["SAC", "TD3", "FTG", "Glob. MPCC", "Loc. MPCC", "Glob. Ts", "Loc. Ts"]
    legend_names = ["SAC", "TD3", "FTG", "Global MPCC", "Global Ts", "Local MPCC", "Local Ts"]
    plt.legend(legend_names, ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.35), fontsize=9)

    plt.savefig("Data/LocalMapRacing/LaptimesBarPlot.svg", bbox_inches='tight', pad_inches=0)
    plt.savefig("Data/LocalMapRacing/LaptimesBarPlot.pdf", bbox_inches='tight', pad_inches=0)


def generate_relative_time_table():
    summary_df = pd.read_csv("Logs/Summary.csv")

    vehicle_id = {
                  "EndToEnd_TestTD3": "End-to-end TD3", 
                  "FollowTheGap_Std": "Follow the gap", 
                  "FullStackPP_mu60": "Global two-stage", 
                  "LocalMapPP_mu60": "Local two-stage", 
                  }


    results_df = summary_df.loc[summary_df.VehicleID.isin(vehicle_id)]
    results_df = results_df.replace({"VehicleID": vehicle_id})


    times_df = results_df.pivot(index="VehicleID", columns="MapName", values="AvgTime")
    times_df = times_df.drop(columns=["mco"])
    times_df.columns = times_df.columns.str.upper()

    times_df = times_df.T
    print(times_df)

    diffs = []
    for i in range(len(times_df)):
        dif = {"MapName": f"{times_df.index[i]}_dif"}
        for j in range(len(times_df.columns)):
            num_diff = times_df.iloc[i, j] - times_df.iloc[i, -1]
            percent_diff = num_diff / times_df.iloc[i, -1] * 100
            dif[times_df.columns[j]] = f"{num_diff:.2f} ({percent_diff:.1f}\%)"
            # dif[times_df.columns[j]] = times_df.iloc[i, j] - times_df.iloc[i, -1]
        diffs.append(dif)

    diffs_df = pd.DataFrame(diffs)
    diffs_df = diffs_df.set_index("MapName")
    diffs_df = diffs_df.round(2)
    print(diffs_df)

    total_df = pd.concat([times_df, diffs_df])
    total_df = total_df.sort_index()
    print(total_df)


    print(times_df)

    total_df.to_latex(f"Data/LocalMapRacing/RelativeLaptimes.tex", float_format="%.2f")
    # times_df.to_latex(f"Data/LocalMapRacing/Laptimes.tex", float_format="%.2f")


def generate_relative_time_table2():
    summary_df = pd.read_csv("Logs/Summary.csv")

    vehicle_id = {
                  "EndToEnd_TestTD3": "End-to-end TD3", 
                  "FollowTheGap_Std": "Follow the gap", 
                  "FullStackPP_mu60": "Global two-stage", 
                  "LocalMapPP_mu60": "Local two-stage", 
                  }


    results_df = summary_df.loc[summary_df.VehicleID.isin(vehicle_id)]
    results_df = results_df.replace({"VehicleID": vehicle_id})


    times_df = results_df.pivot(index="VehicleID", columns="MapName", values="AvgTime")
    times_df = times_df.drop(columns=["mco"])
    times_df.columns = times_df.columns.str.upper()

    times_df = times_df.T
    times_df = times_df.round(2)
    print(times_df)

    diffs = []
    num_diffs = []
    for i in range(len(times_df)):
        dif = {"MapName": f"{times_df.index[i]}"}
        ndif = {"MapName": f"{times_df.index[i]}"}
        for j in range(len(times_df.columns)): 
            if j == len(times_df.columns) - 1:
                dif[times_df.columns[j]] = f"{times_df.iloc[i, j]:.2f} "
                ndif[times_df.columns[j]] = 0
                continue
            num_diff = times_df.iloc[i, j] - times_df.iloc[i, -1]
            percent_diff = num_diff / times_df.iloc[i, -1] * 100
            dif[times_df.columns[j]] = f"{times_df.iloc[i, j]:.2f} ({percent_diff:.1f}\%)"
            ndif[times_df.columns[j]] = percent_diff
        diffs.append(dif)
        num_diffs.append(ndif)

    diffs_df = pd.DataFrame(diffs)
    diffs_df = diffs_df.set_index("MapName")
    diffs_df = diffs_df.round(2)

    num_diffs_df = pd.DataFrame(num_diffs)
    num_diffs_df = num_diffs_df.set_index("MapName")
    num_diffs_df = num_diffs_df.round(2)

    means = num_diffs_df.mean(axis=0)
    means = means.round(2)

    # diffs_df = diffs_df.append(means, ignore_index=False)
    diffs_df = pd.concat([diffs_df, means.to_frame().T])

    print(diffs_df)

    # total_df = pd.concat([times_df, diffs_df])
    # total_df = total_df.sort_index()
    # print(total_df)


    # print(times_df)

    diffs_df.to_latex(f"Data/LocalMapRacing/RelativeLaptimes.tex", float_format="%.2f")
    # times_df.to_latex(f"Data/LocalMapRacing/Laptimes.tex", float_format="%.2f")


# build_main_df()
# generate_bar_plot()
generate_relative_time_table2()