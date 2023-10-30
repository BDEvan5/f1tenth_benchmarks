

from f1tenth_sim.simulator.dynamics_simulator import DynamicsSimulator
from f1tenth_sim.simulator.laser_models import ScanSimulator2D
from f1tenth_sim.simulator.utils import CenterLine, SimulatorHistory
import yaml
from argparse import Namespace
import numpy as np
import pandas as pd


class F1TenthSim:
    def __init__(self, map_name, log_name=None):
        with open(f"f1tenth_sim/simulator/simulator_params.yaml", 'r') as file:
            params = yaml.load(file, Loader=yaml.FullLoader)
        self.params = Namespace(**params)

        self.scan_simulator = ScanSimulator2D(self.params.num_beams, self.params.fov, map_name, self.params.random_seed)
        self.dynamics_simulator = DynamicsSimulator(self.params.random_seed, self.params.timestep)
        self.center_line = CenterLine(map_name)

        # state is [x, y, steer_angle, vel, yaw_angle, yaw_rate, slip_angle]
        self.current_state = np.zeros((7, ))
        self.current_time = 0.0
        self.progress = 0 
        self.lap_number = -1 # so that it goes to 0 when reset
        self.starting_progress = 0

        self.history = None
        if log_name != None:
            self.history = SimulatorHistory(log_name)
            self.history.set_map_name(map_name)
            self.lap_history = []

        self.map_name = map_name

    def step(self, action):
        if self.history is not None:
            self.history.add_memory_entry(self.current_state, action, self.scan)

        mini_i = self.params.n_sim_steps
        while mini_i > 0:
            self.current_state = self.dynamics_simulator.update_pose(action[0], action[1])
            self.current_time = self.current_time + self.params.timestep
            mini_i -= 1
        
        pose = np.append(self.current_state[0:2], self.current_state[4])
        self.collision = self.check_vehicle_collision(pose)
        self.lap_complete = self.check_lap_complete(pose)
        observation = self.build_observation(pose)
        
        done = self.collision or self.lap_complete
        if done and self.history is not None:
            self.history.save_history()
            self.lap_history.append({"Lap": self.lap_number, "TestMap": self.map_name, "TestID": self.history.run_name, "Progress": self.progress, "Time": self.current_time})

        if self.collision:
            print(f"{self.lap_number} COLLISION: Time: {self.current_time:.2f}, Progress: {100*self.progress:.1f}")
        elif self.lap_complete:
            print(f"{self.lap_number} LAP COMPLETE: Time: {self.current_time:.2f}, Progress: {(100*self.progress):.1f}")

        return observation, done

    def build_observation(self, pose):
        raise NotImplementedError("The build_observation method has not been implemented")

    def check_lap_complete(self, pose):
        self.progress = self.center_line.calculate_pose_progress(pose) - self.starting_progress
        
        done = False
        if self.progress > 0.99 and self.current_time > 5: done = True
        if self.current_time > 250: 
            print("Time limit reached")
            done = True

        return done
        
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
    

    def reset(self):
        # self.starting_progress = np.random.random()
        # start_pose = self.center_line.get_pose_from_progress(self.starting_progress)
        # print(f"Resetting to {self.starting_progress:.2f} progress with pose: {start_pose}")
        start_pose = np.zeros(3)
        self.dynamics_simulator.reset(start_pose)

        self.current_time = 0.0
        action = np.zeros(2)
        obs, done = self.step(action)

        self.lap_number += 1
        
        return obs, done

    def save_data_frame(self):
        if self.history is not None:
            agent_df = pd.DataFrame(self.lap_history)
            agent_df = agent_df.sort_values(by=["Lap"])
            agent_df.to_csv(self.history.path + f"Results_{self.map_name}.csv", index=False, float_format='%.4f')


class StdF1TenthSim(F1TenthSim):
    def __init__(self, map_name, log_name=None):
        super().__init__(map_name, log_name)
        init_pose = np.append(self.current_state[0:2], self.current_state[4])
        self.scan = self.scan_simulator.scan(init_pose)
 
    def build_observation(self, pose):
        self.scan = self.scan_simulator.scan(pose)
        observation = {"scan": self.scan,
                "vehicle_speed": self.dynamics_simulator.state[3],
                "collision": self.collision,
                "lap_complete": self.lap_complete,
                "laptime": self.current_time}
        return observation

class PlanningF1TenthSim(F1TenthSim):
    def __init__(self, map_name, log_name=None):
        super().__init__(map_name, log_name)
        init_pose = np.append(self.current_state[0:2], self.current_state[4])
        self.scan = self.scan_simulator.scan(init_pose)
    
    def build_observation(self, pose):
        self.scan = self.scan_simulator.scan(pose)
        observation = {"scan": self.scan,
                "vehicle_state": self.dynamics_simulator.state,
                "collision": self.collision,
                "lap_complete": self.lap_complete,
                "laptime": self.current_time,
                "progress": self.progress}
        return observation

