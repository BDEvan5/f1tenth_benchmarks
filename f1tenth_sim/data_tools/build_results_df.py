import pandas as pd
import glob 
import numpy as np
import os



def build_main_df():
    folders = glob.glob("Logs/*")
    full_df = []
    summary_df = []
    for folder in folders:
        vehicle_name = folder.split("/")[-1]
        if not os.path.exists(folder + f"/Results_{vehicle_name}.csv"): continue
        df = pd.read_csv(folder + f"/Results_{vehicle_name}.csv")
        df["Vehicle"] = vehicle_name
        full_df.append(df)

        for map_name in df.TestMap.unique():
            map_df = df.loc[df["TestMap"] == map_name]
            completion_rate = np.count_nonzero(map_df.Progress > 0.99) / map_df.shape[0] 
            summary_df.append({"Vehicle": vehicle_name, "MapName": map_name, "AvgProgress": map_df.Progress.mean(), "AvgTime": map_df.Time.mean(), "CompletionRate": completion_rate})



    full_df = pd.concat(full_df)
    full_df = full_df.sort_values(by=["Vehicle", "TestMap"])
    full_df.to_csv("Logs/Full.csv", index=False, float_format='%.4f')
    summary_df = pd.DataFrame(summary_df)
    summary_df = summary_df.sort_values(by=["Vehicle", "MapName"])
    summary_df.to_csv("Logs/Summary.csv", index=False, float_format='%.4f')


build_main_df()