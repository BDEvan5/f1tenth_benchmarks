

from f1tenth_sim.simulator.dynamics_simulator import DynamicsSimulator
from f1tenth_sim.simulator.laser_models import ScanSimulator2D
from f1tenth_sim.simulator.utils import CenterLine, SimulatorHistory
import yaml
from argparse import Namespace
import numpy as np

'''
    params (dict, default={'mu': 1.0489, 'C_Sf':, 'C_Sr':, 'lf': 0.15875, 'lr': 0.17145, 'h': 0.074, 'm': 3.74, 'I': 0.04712, 's_min': -0.4189, 's_max': 0.4189, 'sv_min': -3.2, 'sv_max': 3.2, 'v_switch':7.319, 'a_max': 9.51, 'v_min':-5.0, 'v_max': 20.0, 'width': 0.31, 'length': 0.58}): dictionary of vehicle parameters.
    mu: surface friction coefficient
    C_Sf: Cornering stiffness coefficient, front
    C_Sr: Cornering stiffness coefficient, rear
    lf: Distance from center of gravity to front axle
    lr: Distance from center of gravity to rear axle
    h: Height of center of gravity
    m: Total mass of the vehicle
    I: Moment of inertial of the entire vehicle about the z axis
    s_min: Minimum steering angle constraint
    s_max: Maximum steering angle constraint
    sv_min: Minimum steering velocity constraint
    sv_max: Maximum steering velocity constraint
    v_switch: Switching velocity (velocity at which the acceleration is no longer able to create wheel spin)
    a_max: Maximum longitudinal acceleration
    v_min: Minimum longitudinal velocity
    v_max: Maximum longitudinal velocity
    width: width of the vehicle in meters
    length: length of the vehicle in meters
'''

default_run_dict = {"random_seed": 12345, "n_sim_steps": 5, "num_beams": 20}
#info: this is for parameters like seeds, noise, frequency etc
#TODO: this can be loaded based on a mode and control sim behaviour, i.e. if position is included in the action....


class F1TenthSim:
    def __init__(self, map_name, log_name=None):
        with open(f"f1tenth_sim/simulator/simulator_params.yaml", 'r') as file:
            params = yaml.load(file, Loader=yaml.FullLoader)
        self.params = Namespace(**params)
        self.map_name = map_name
        self.timestep = self.params.timestep

        self.scan_simulator = ScanSimulator2D(self.params.num_beams, self.params.fov)
        self.scan_simulator.set_map(self.map_name)
        self.dynamics_simulator = DynamicsSimulator(self.params.random_seed, self.timestep)
        self.scan_rng = np.random.default_rng(seed=self.params.random_seed)
        self.center_line = CenterLine(map_name)

        self.current_time = 0.0
        self.current_state = np.zeros((7, ))
        self.lap_number = -1

        self.history = None
        if log_name != None:
            self.history = SimulatorHistory(log_name)
            self.history.set_map_name(self.map_name)

    def step(self, action):
        if self.history is not None:
            self.history.add_memory_entry(self.current_state, action)

        mini_i = self.params.n_sim_steps
        while mini_i > 0:
            vehicle_state = self.dynamics_simulator.update_pose(action[0], action[1])
            self.current_time = self.current_time + self.timestep
            mini_i -= 1
        
        pose = np.append(vehicle_state[0:2], vehicle_state[4])
        scan = self.scan_simulator.scan(np.append(vehicle_state[0:2], vehicle_state[4]), self.scan_rng)

        self.current_state = vehicle_state
        self.collision = self.check_vehicle_collision(pose)
        self.lap_complete, progress = self.check_lap_complete(pose)

        # state is [x, y, steer_angle, vel, yaw_angle, yaw_rate, slip_angle]
        observation = {"scan": scan,
                        "vehicle_state": self.dynamics_simulator.state,
                        "collision": self.collision,
                        "lap_complete": self.lap_complete,
                        "laptime": self.current_time,
                        "progress": progress}
        
        done = self.collision or self.lap_complete
        if done:
            if self.history is not None:
                self.history.save_history()

        if self.collision:
            print(f"{self.lap_number} COLLISION: Time: {self.current_time:.2f}, Progress: {100*progress:.1f}")
        elif self.lap_complete:
            print(f"{self.lap_number} LAP COMPLETE: Time: {self.current_time:.2f}, Progress: {(100*progress):.1f}")


        return observation, done

    def check_lap_complete(self, pose):
        progress = self.center_line.calculate_pose_progress(pose)
        
        done = False
        if progress > 0.99 and self.current_time > 5: done = True
        if self.current_time > 250: 
            print("Time limit reached")
            done = True

        return done, progress
        

    def check_vehicle_collision(self, pose):
        rotation_mtx = np.array([[np.cos(pose[2]), -np.sin(pose[2])], [np.sin(pose[2]), np.cos(pose[2])]])

        pts = np.array([[self.params.vehicle_length/2, self.params.vehicle_width/2], 
                        [self.params.vehicle_length, -self.params.vehicle_width/2], 
                        [-self.params.vehicle_length, self.params.vehicle_width/2], 
                        [-self.params.vehicle_length, -self.params.vehicle_width/2]])
        pts = np.matmul(pts, rotation_mtx.T) + pose[0:2]

        for i in range(4):
            if self.scan_simulator.check_location(pts[i, :]):
                return True

        return False
    

    def reset(self, poses):
        if self.history is not None:
            self.history.set_map_name(self.map_name)
        self.dynamics_simulator.reset(poses)

        self.current_time = 0.0
        action = np.zeros(2)
        obs, done = self.step(action)

        self.lap_number += 1
        
        return obs, done


    
