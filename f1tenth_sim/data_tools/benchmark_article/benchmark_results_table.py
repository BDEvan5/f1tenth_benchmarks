import pandas as pd
import glob 
import numpy as np
import os



def build_main_df():
    summary_df = pd.read_csv("logs/Summary.csv")

    planners = ['PurePursuit', "MPCC", "FollowTheGap"]
    test_ids = ["mu70", "mu70", "v1"]

    results_df = summary_df.loc[summary_df.Vehicle.isin(planners)]
    results_df = results_df.loc[results_df.TestID.isin(test_ids)]

    # print(results_df)

    times_df = results_df.pivot(index="Vehicle", columns="MapName", values="AvgTime")
    times_df.columns = times_df.columns.str.upper()
    print(times_df)

    times_df.to_latex(f"Data/BenchmarkArticle/Laptimes.tex", float_format="%.2f")

build_main_df()