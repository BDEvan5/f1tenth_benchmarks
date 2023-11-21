import pandas as pd
import glob 
import numpy as np
import os



def build_main_df():
    summary_df = pd.read_csv("Logs/Summary.csv")

    vehicle_id = {"FullStackPP_mu60": "Global two-stage", 
                  "FullStackMPCC_mu60": "Global MPCC", 
                  "FollowTheGap_Std": "Follow the gap", 
                  "EndToEnd_TestSAC": "End-to-end SAC", 
                  "EndToEnd_TestTD3": "End-to-end TD3", 
                  "LocalMapPP_mu60": "Local two-stage", 
                  "LocalMPCC_mu60": "Local MPCC"}


    results_df = summary_df.loc[summary_df.VehicleID.isin(vehicle_id)]
    results_df = results_df.replace({"VehicleID": vehicle_id})
    
    # print(results_df)

    times_df = results_df.pivot(index="VehicleID", columns="MapName", values="AvgTime")
    times_df.columns = times_df.columns.str.upper()
    print(times_df)

    times_df.to_latex(f"Data/LocalMapRacing/Laptimes.tex", float_format="%.2f")

build_main_df()