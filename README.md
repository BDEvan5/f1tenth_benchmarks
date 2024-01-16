# F1Tenth Benchmark Repo

This repository contains code to run benchmark algorithms for F1Tenth autonomous racing. 
The code is divided into two categories of:
1. Classical racing: which uses a classic perception, planning and control stack with prior access to a map of the track.
2. Mapless racing: where the planner does not have any access to a map of the track and only the LiDAR scan and vehicle speed is available.

### Classical racing

The classic racing stack has several key parts:
- **Particle filter localisation:** the particle filter that localises the vehicle uses state estimation theory to sample a proposal distribution and update it using the LiDAR data. More information is available [here](http://github.com/BDEvan5/sensor_fusion)
- **Optimal Trajectory Generation:** optimal trajectories (racelines) are generated using the `generate_racelines.py` script. The [trajectory_planning_helpers](https://github.com/FTM_TUM/trajectory_planning_helpers) library is used to generate minimum cuvature trajectories, followed by minimum time speed profiles.
- **Pure Pursuit Path Tracking:** the pure pursuit path tracker uses a geometric vehicle model to follow the optimal trajectory.
- **Model predictive contouring control:** the MPCC algorithm maximises progress along the center line (not requiring an optimal trajectory) using an receeding horizon optimisation approach.


### Mapless Racing

Three different methods for mapless racing are presented:
1. **Follow the gap algorithm:** the follow the gap algorithm calculates the largest gap and then steers towards it.
2. **End-to-end deep reinforcement learning:** the SAC and TD3 algorithms are used for end-to-end reinforcement learning which uses the last two LiDAR scans and vehicle speed as input to a neural network that directly outputs speed and steering angles. The agents are trained using the [trajectory aided learning](https://ieeexplore.ieee.org/document/10182327) reward signal for 60,000 steps.


### Simulator
- The [f1tenth_gym](https://github.com/f1tenth/f1tenth_gym) base simulator is used, but repackaged to allow for the analytics to be collected. The dynamics model, and scan simulator model are kept the same to ensure that results are transferrable.

> The classical methods are tested with particle filter localisation and with the vehicle's true location. 
> This is done by providing two simulator classes; `F1TenthSim`, which only has the LiDAR scan and `F1TenthSim_TrueLocation` which includes the entire vehicle state.

## Usage

**Installation**

It is recommended that you use a virtual environment to manage dependencies. A virtual environment can be created and sourced with the following commands:
```bash
python3.9 -m venv venv
source venv/bin/activate
```

The requirements and package can then be installed using,
```bash
pip install -r requirements.txt
pip install -e .
```

The [trajectory_planning_helpers](https://github.com/TUMFTM/trajectory_planning_helpers.git) library, must be installed independantly through the following commands, 
```
git submodule update
cd trajectory_planning_helpers
pip install -e .
```

**Test scripts:**
There are several key scripts for running the tests:
- `classical_racing/GenerateOptimalTrajectory.py`: will generate the racelines in the `raceliones/` directory
- `test_planning_methods.py`: evaluate the pure pursuit and MPCC algorithms. These tests assume that the vehicle has perfect state estimation.
- `train_drl_agents.py`: this script trains the SAC or TD3 agents on a specified map. Once training is complete, the agent will be tested on all four maps.

**Analysis scripts:**
- `build_results_df.py`: this script builds a data frame of the results from all the planners.
- `plot_drl_training.py`: plots the reward and progress during training an agent.
- `plot_trajectory.py`: plots images for the trajectory for each agent.



## Getting Started with Docker

To ensure repeatability and useability, a Dockerfile is provided that can be used to run the code.

- Build the Docker image using the Dockerfile
```
sudo docker build -t f1tenth_sim -f Dockerfile .
```
- Start the docker image with
```
sudo docker compose up
```
Doing this mounts the current folder as a volume.
- Enter the docker container using,
```
sudo docker exec -it f1tenth_sim-sim-1 /bin/bash
```
- You can now run commands in the interactive shell.


# Detailed Usage

## Classic Racing

The classic racing algorithms use a model of the vehicle to calculate control comands.

**MPCC:**
The MPCC algorithm requires only the track centre line to operate.
A constant speed implementation is included to aid in understanding how the optimisation routine works.

**Pure Pursuit:**
The two-stage planner uses a trajectory optimisation algorithm to generate a raceline and the pure pursuit algorithm to track the raceline.
A raceline must be generated using the `RaceTrackGenerator.py` file before the planner can be used.

## Mapless Racing

The follow-the-gap is 





