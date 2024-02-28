from f1tenth_benchmarks.classic_racing.GlobalMPCC import GlobalMPCC
from f1tenth_benchmarks.classic_racing.GlobalPurePursuit import GlobalPurePursuit

from f1tenth_benchmarks.run_scripts.run_functions import *



"""
Run the MPCC tests to generate the lap times graph
"""
def planning_mpcc_frictions():
    friction_mus = [0.5, 0.6, 0.7, 0.8, 0.9, 1]

    for mu in friction_mus:
        test_id = f"mu{int(mu*100)}_steps4"
        planner = GlobalMPCC(test_id, False, planner_name="GlobalPlanMPCC", extra_params={"friction_mu": mu})
        test_planning_single_map(planner, "aut", test_id, number_of_laps=10)


def full_stack_mpcc_frictions():
    friction_mus = [0.5, 0.6, 0.7, 0.8, 0.9, 1]

    for mu in friction_mus:
        test_id = f"mu{int(mu*100)}_steps4"
        planner = GlobalMPCC(test_id, False, planner_name="FullStackMPCC", extra_params={"friction_mu": mu})
        test_full_stack_single_map(planner, "aut", test_id, number_of_laps=10)


"""
Run the Pure Pursuit tests to generate the lap times graph
"""
def planning_pure_puresuit_frictions():
    map_name = "aut"
    friction_vals = [0.5, 0.6, 0.7, 0.8, 0.9, 1]
    for friction in friction_vals:
        test_id = f"mu{int(friction*100)}_steps4"
        print(f"Testing {test_id}...")
        planner = GlobalPurePursuit(test_id, False, planner_name="GlobalPlanPP", extra_params={"racetrack_set": f"mu{int(friction*100)}"})
        test_planning_single_map(planner, map_name, test_id, number_of_laps=10)



def full_stack_pure_pursuit_frictions():
    map_name = "aut"
    friction_vals = [0.5, 0.6, 0.7, 0.8, 0.9, 1]
    for friction in friction_vals:
        test_id = f"mu{int(friction*100)}_steps4"
        print(f"Testing {test_id}...")
        planner = GlobalPurePursuit(test_id, False, planner_name="FullStackPP", extra_params={"racetrack_set": f"mu{int(friction*100)}"})
        test_full_stack_single_map(planner, map_name, test_id, number_of_laps=10)


"""
Run the pure puresuit tests to generate the completion rate graph
"""

def planning_pure_puresuit_frequencies():
    map_name = "aut"
    friction_vals = [0.7, 0.8, 0.9, 1]
    simulator_timestep_list = [2, 4, 6, 8, 10, 12, 14]
    for simulator_timestep in simulator_timestep_list:
        for friction in friction_vals:
            test_id = f"mu{int(friction*100)}_steps{simulator_timestep}"
            print(f"Testing {test_id}...")
            planner = GlobalPurePursuit(test_id, False, planner_name="GlobalPlanPP", extra_params={"racetrack_set": f"mu{int(friction*100)}"})
            test_planning_single_map(planner, map_name, test_id, extra_params={"n_sim_steps": simulator_timestep}, number_of_laps=10)

def full_stack_pure_puresuit_frequencies():
    map_name = "aut"
    friction_vals = [0.7, 0.8, 0.9, 1]
    simulator_timestep_list = [2, 4, 6, 8, 10, 12, 14]
    for simulator_timestep in simulator_timestep_list:
        for friction in friction_vals:
            test_id = f"mu{int(friction*100)}_steps{simulator_timestep}"
            print(f"Testing {test_id}...")
            planner = GlobalPurePursuit(test_id, False, planner_name="FullStackPP300", extra_params={"racetrack_set": f"mu{int(friction*100)}"})
            test_full_stack_single_map(planner, map_name, test_id, extra_params={"n_sim_steps": simulator_timestep}, number_of_laps=10, extra_pf_params={"number_of_particles": 300})



if __name__ == "__main__":
    planning_mpcc_frictions()
    full_stack_mpcc_frictions()

    # planning_pure_puresuit_frictions()
    # full_stack_pure_pursuit_frictions()

    # planning_pure_puresuit_frequencies()
    # full_stack_pure_puresuit_frequencies()

