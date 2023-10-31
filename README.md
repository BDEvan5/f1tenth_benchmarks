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
3. **Local map racing:** local map racing uses the LiDAR scan to reconstruct a map of the visible region of the track. An optimisation and control based strategy is then employed to control the vehicle.


### Simulator
- The [f1tenth_gym](https://github.com/f1tenth/f1tenth_gym) base simulator is used, but repackaged to allow for the analytics to be collected. The dynamics model, and scan simulator model are kept the same to ensure that results are transferrable.

> The classical methods are tested with particle filter localisation and with the vehicle's true location. 
> This is done by providing two simulator classes; `F1TenthSim`, which only has the LiDAR scan and `F1TenthSim_TrueLocation` which includes the entire vehicle state.


## Getting Started

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



