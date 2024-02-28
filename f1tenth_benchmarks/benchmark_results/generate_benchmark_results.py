from f1tenth_benchmarks.classic_racing.RaceTrackGenerator import RaceTrackGenerator, load_parameter_file_with_extras
from f1tenth_benchmarks.classic_racing.GlobalPurePursuit import GlobalPurePursuit
from f1tenth_benchmarks.classic_racing.GlobalMPCC import GlobalMPCC
from f1tenth_benchmarks.mapless_racing.FollowTheGap import FollowTheGap
from f1tenth_benchmarks.drl_racing.EndToEndAgent import EndToEndAgent, TrainEndToEndAgent

from f1tenth_benchmarks.data_tools.specific_plotting.plot_drl_training import plot_drl_training
from f1tenth_benchmarks.data_tools.general_plotting.plot_trajectory_analysis import plot_trajectory_analysis
from f1tenth_benchmarks.run_scripts.run_functions import *

NUMBER_OF_LAPS = 10


def generate_racelines():
    map_list = ["aut", "esp", "gbr", "mco"]
    params = load_parameter_file_with_extras("RaceTrackGenerator", extra_params={"mu": 0.9})
    raceline_id = f"mu{int(params.mu*100)}"
    for map_name in map_list:
        RaceTrackGenerator(map_name, raceline_id, params, plot_raceline=True)


def optimisation_and_tracking():
    test_id = "benchmark"
    planner = GlobalPurePursuit(test_id, False, planner_name="GlobalPlanPP", extra_params={"racetrack_set": "mu90"})
    test_planning_all_maps(planner, test_id, number_of_laps=NUMBER_OF_LAPS)

    plot_trajectory_analysis(planner.name, test_id)


def mpcc():
    test_id = f"benchmark"
    planner = GlobalMPCC(test_id, False, planner_name="GlobalPlanMPCC", extra_params={"friction_mu": 0.9})
    test_planning_all_maps(planner, test_id, number_of_laps=10)

    plot_trajectory_analysis(planner.name, test_id)


def follow_the_gap():
    test_id = "benchmark"
    planner = FollowTheGap(test_id)
    test_mapless_all_maps(planner, test_id, number_of_laps=NUMBER_OF_LAPS)

    plot_trajectory_analysis(planner.name, test_id)


def end_to_end_drl():
    # map_list = ["aut", "esp", "gbr", "mco"]
    # params = load_parameter_file_with_extras("RaceTrackGenerator", extra_params={"mu": 0.95})
    # raceline_id = f"mu{int(params.mu*100)}"
    # for map_name in map_list:
    #     RaceTrackGenerator(map_name, raceline_id, params, plot_raceline=True)

    seed_randomness(13)
    test_id = "benchmark"
    print(f"Training DRL agent: {test_id}")
    training_agent = TrainEndToEndAgent("mco", test_id, extra_params={'reward': "TAL", 'tal_racetrack_set': "mu95"}) # TODO: possibly include autogeneration in the TAL reward for if the files do not exist.
    simulate_training_steps(training_agent, "mco", test_id, extra_params={'n_sim_steps': 10})
    plot_drl_training(training_agent.name, test_id)

    testing_agent = EndToEndAgent(test_id)
    test_mapless_all_maps(testing_agent, test_id, number_of_laps=10)




if __name__ == "__main__":
    # generate_racelines()
    optimisation_and_tracking()
    mpcc()
    # follow_the_gap()
    # end_to_end_drl()






