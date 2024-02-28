import pandas as pd
import glob 
import numpy as np
import os



def process_data():
    folders = glob.glob("Logs/*")
    full_df = []
    summary_df = []
    for folder in folders:
        planner_name = folder.split("/")[-1]
        if not os.path.exists(folder + f"/Results_{planner_name}.csv"): continue
        df = pd.read_csv(folder + f"/Results_{planner_name}.csv")
        df["Vehicle"] = planner_name
        full_df.append(df)

        for map_name in df.TestMap.unique():
            map_df = df.loc[df["TestMap"] == map_name]
            for test_id in map_df.TestID.unique():
                test_id_df = map_df.loc[map_df.TestID == test_id]
                completion_rate = np.count_nonzero(test_id_df.Progress > 0.99) / test_id_df.shape[0] 
                average_time = test_id_df.loc[test_id_df.Progress > 0.994].Time.mean()
                std_time = test_id_df.loc[test_id_df.Progress > 0.994].Time.std()
                summary_df.append({"Vehicle": planner_name, "TestID":test_id, "MapName": map_name, "AvgProgress": test_id_df.Progress.mean(), "AvgTime": average_time, "StdTime": std_time, "CompletionRate": completion_rate})

    full_df = pd.concat(full_df)
    full_df = full_df.sort_values(by=["Vehicle", "TestMap"])
    full_df.to_csv("Logs/Full.csv", index=False, float_format='%.4f')
    summary_df = pd.DataFrame(summary_df)
    summary_df['VehicleID'] = summary_df['Vehicle'] + "_" + summary_df['TestID']
    summary_df = summary_df.sort_values(by=["Vehicle", "MapName"])
    summary_df.to_csv("Logs/Summary.csv", index=False, float_format='%.4f')
    print("Complied Full.csv and Summary.csv.")

if __name__ == "__main__":
    process_data()