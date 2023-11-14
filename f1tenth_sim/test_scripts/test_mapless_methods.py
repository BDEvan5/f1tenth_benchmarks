from f1tenth_sim.simulator import F1TenthSim 
from f1tenth_sim.mapless_racing.follow_the_gap.FollowTheGap import FollowTheGap
import numpy as np

from testing_utils import *

def run_planning_test_single(planner, map_name, test_id):
    print(f"Testing on {map_name}...")
    simulator = F1TenthSim(map_name, planner.name, test_id)
    run_simulation_loop_laps(simulator, planner, 1)


def run_mapless_tests(planner):
    map_list = ["aut", "esp", "gbr", "mco"]
    inds = np.random.choice(np.arange(len(map_list)), replace=False, size=len(map_list))
    for i in inds:
        run_planning_test_single(planner, map_list[i], test_id="v1")

if __name__ == "__main__":
    run_mapless_tests(FollowTheGap())





