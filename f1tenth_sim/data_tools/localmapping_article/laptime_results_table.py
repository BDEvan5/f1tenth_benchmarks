import pandas as pd
import glob 
import numpy as np
import os
import matplotlib.pyplot as plt
from f1tenth_sim.data_tools.plotting_utils import *



def generate_relative_time_table2():
    summary_df = pd.read_csv("Logs/Summary.csv")

    vehicle_id = {
                  "EndToEnd_TestTD3": "End-to-end TD3", 
                  "FollowTheGap_Std": "Follow the gap", 
                  "FullStackPP_mu70": "Global two-stage", 
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



generate_relative_time_table2()