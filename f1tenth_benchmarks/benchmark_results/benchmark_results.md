# Unifying F1TENTH Autonomous Racing: Survey, Methods and Benchmark Results

This is the codebase for the accompanying paper, available [here](arxiv.com/).

**Abstract:**
The F1TENTH autonomous racing platform, consisting of 1:10 scale RC cars, has evolved into a leading research platform.
The many publications and real-world competitions span many domains, from classical path planning to novel learning-based algorithms.
Consequently, the field is wide and disjointed, hindering direct comparison of methods and making it difficult to assess the state-of-the-art.
Therefore, we aim to unify the field by surveying current approaches, describing common methods and providing benchmark results to facilitate clear comparison and establish a baseline for future work.
We survey current work in F1TENTH racing in the classical and learning categories, explaining the different solution approaches.
We describe particle filter localisation, trajectory optimisation and tracking, model predictive contouring control (MPCC), follow-the-gap and end-to-end reinforcement learning.
We provide an open-source evaluation of benchmark methods and investigate overlooked factors of control frequency and localisation accuracy for classical methods and reward signal and training map for learning methods.
The evaluation shows that the optimisation and tracking method achieves the fastest lap times, followed by the MPCC planner.
Finally, our work identifies and outlines the relevant research aspects to help motivate future work in the F1TENTH domain.

## Data Generation

We describe the tests required to recreate the paper results.

**Friction evaluation: **
The friction evaluation considers the effect of localisation error and control frequency on vehicle performance.
- mpcc_racing.py: runs tests on all the maps with varying friction coefficients
- purepursuit_racing.py: runs tests on the AUT map with varying friction coefficients and control frequencies.
    - TODO: add in racetrack generation....
- frequency_result_plots.ipynb: plots the data to form the images in the 

**Learning evaluation:**
The learning evaluation compares reward signals and training maps.
- train_drl_agents.py: train end-to-end DRL agents with the TD3 algorithm and:
    - 3 random seeds
    - 4 training maps
    - 3 reward signals (cross-track and heading error, track progress, trajectory-aided learning)
- drl_results_plot.ipynb: plots the learning results

**Benchmark results:**
These are the times presented in the paper that can be used for future comparison.
If the previous tests have not been run, then the following script is required:
- generate_benchmark_data.py: generates data using the DRL, MPCC, optimisation and tracking and FTG methods.
Then:
- benchmark_results_plot.ipynb: plots the results and generates the table of lap times.







