from f1tenth_sim.simulator import StdF1TenthSim 
from f1tenth_sim.mapless_racing.follow_the_gap.FollowTheGap import FollowTheGap
import numpy as np

from testing_utils import *


def run_mapless_tests(planner):
    map_list = ["aut", "esp", "gbr", "mco"]
    inds = np.random.choice(np.arange(len(map_list)), replace=False, size=len(map_list))
    for i in inds:
        print(f"Testing on {map_list[i]}...")
        simulator = StdF1TenthSim(map_list[i], planner.name)
        run_simulation_loop_laps(simulator, planner, 5)


if __name__ == "__main__":
    run_mapless_tests(FollowTheGap())





