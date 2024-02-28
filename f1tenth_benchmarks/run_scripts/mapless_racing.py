from f1tenth_benchmarks.mapless_racing.FollowTheGap import FollowTheGap

from f1tenth_benchmarks.run_scripts.run_functions import *




if __name__ == "__main__":
    test_mapless_single_map(FollowTheGap("Std"), "aut", "Std", number_of_laps=5)
    # test_mapless_all_maps(FollowTheGap("Std"), "Std", number_of_laps=5)





