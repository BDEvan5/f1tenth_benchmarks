from f1tenth_sim.simulator import F1TenthSim 
from f1tenth_sim.racing_methods.planning.pp_traj_following.Planner import Planner
import numpy as np


def run_simulation_loop_laps(env, planner, n_laps):
    for lap in range(n_laps):
        observation, done = env.reset(poses=np.array([0, 0, 0]))
        while not done:
            action = planner.plan(observation)
            observation, done = env.step(action)
    

def run_test():
    map_name = "aut"
    print(f"Testing....")

    simulator = F1TenthSim(map_name, "pp_traj_following")
    planner = Planner(simulator.map_name)

    run_simulation_loop_laps(simulator, planner, 1)


if __name__ == "__main__":
    run_test()





