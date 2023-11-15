

from f1tenth_sim.simulator.dynamics_simulator import DynamicsSimulator
from f1tenth_sim.simulator.laser_models import ScanSimulator2D
from f1tenth_sim.simulator.utils import CenterLine, SimulatorHistory
import yaml
from argparse import Namespace
import numpy as np
import pandas as pd
import os, datetime
import cProfile, io, pstats


def ensure_path_exists(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

class F1TenthSimBase:
    def __init__(self, map_name, planner_name, test_id, save_detail_history=True, training=False):
        with open(f"f1tenth_sim/simulator/simulator_params.yaml", 'r') as file:
            params = yaml.load(file, Loader=yaml.FullLoader)
        self.params = Namespace(**params)
        self.planner_name = planner_name
        self.map_name = map_name
        self.path = f"Logs/{planner_name}/"
        self.test_id = test_id
        ensure_path_exists(self.path)
        ensure_path_exists(self.path + f"RawData_{test_id}/")
        self.training = training

        self.scan_simulator = ScanSimulator2D(self.params.num_beams, self.params.fov, map_name, self.params.random_seed)
        self.dynamics_simulator = DynamicsSimulator(self.params.random_seed, self.params.timestep)
        self.center_line = CenterLine(map_name)

        # state is [x, y, steer_angle, vel, yaw_angle, yaw_rate, slip_angle]
        self.current_state = np.zeros((7, ))
        self.current_time = 0.0
        self.progress = 0 
        self.lap_number = -1 # so that it goes to 0 when reset
        self.starting_progress = 0
        self.total_steps = 0

        self.history = None

        self.lap_history = []
        if save_detail_history:
            self.history = SimulatorHistory(self.path, test_id, True)
            self.history.set_map_name(map_name)
            
        self.pr = cProfile.Profile()
        self.pr.enable()

    def __del__(self):
        self.pr.disable()
        ps = pstats.Stats(self.pr).sort_stats('cumulative')
        stats_profile_functions = ps.get_stats_profile().func_profiles
        df_entries = []
        for k in stats_profile_functions.keys():
            v = stats_profile_functions[k]
            entry = {"func": k, "ncalls": v.ncalls, "tottime": v.tottime, "percall_tottime": v.percall_tottime, "cumtime": v.cumtime, "percall_cumtime": v.percall_cumtime, "file_name": v.file_name, "line_number": v.line_number}
            df_entries.append(entry)
        df = pd.DataFrame(df_entries)
        df = df[df.cumtime > 0]
        df = df[df.file_name != "~"] # this removes internatl file calls.
        df = df[~df['file_name'].str.startswith('<')]
        df = df.sort_values(by=['cumtime'], ascending=False)
        df.to_csv(f"Logs/{self.planner_name}/RawData_{self.test_id}/Profile_{self.map_name}_{self.test_id}.csv")

        if self.history and len(self.history.states) > 1:
            self.history.save_history()

    def step(self, action):
        if self.history is not None:
            self.history.add_memory_entry(self.current_state, action, self.scan, self.progress)

        mini_i = self.params.n_sim_steps
        while mini_i > 0:
            self.current_state = self.dynamics_simulator.update_pose(action[0], action[1])
            self.current_time = self.current_time + self.params.timestep
            mini_i -= 1
        
        pose = np.append(self.current_state[0:2], self.current_state[4])
        self.collision = self.check_vehicle_collision(pose)
        self.lap_complete = self.check_lap_complete(pose)
        observation = self.build_observation(pose)
        self.total_steps += 1
        
        done = self.collision or self.lap_complete
        if done:
            self.lap_history.append({"Lap": self.lap_number, "TestMap": self.map_name, "TestID": self.test_id, "Progress": self.progress, "Time": self.current_time, "Steps": self.total_steps, "RecordTime": datetime.datetime.now(), "Planner": self.planner_name, "EntryID": f"{self.map_name}_{self.test_id}_{self.lap_number}"})
            self.save_data_frame()
            if self.history is not None: self.history.save_history()

        if self.collision:
            print(f"{self.lap_number} :: {self.total_steps} COLLISION: Time: {self.current_time:.2f}, Progress: {100*self.progress:.1f}")
        elif self.lap_complete:
            print(f"{self.lap_number} :: {self.total_steps} LAP COMPLETE: Time: {self.current_time:.2f}, Progress: {(100*self.progress):.1f}")


        return observation, done

    def build_observation(self, pose):
        raise NotImplementedError("The build_observation method has not been implemented")

    def check_lap_complete(self, pose):
        self.progress = self.center_line.calculate_pose_progress(pose) - self.starting_progress
        
        done = False
        if self.progress > 0.99 and self.current_time > 5: done = True
        if self.current_time > 250: 
            print("Time limit reached --> Lap not complete but no collision")
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
        # start_pose = np.array([-7.3206, -3.1456,  1.6272])
        self.dynamics_simulator.reset(start_pose)
        self.current_state = np.zeros((7, ))

        self.current_time = 0.0
        action = np.zeros(2)
        obs, done = self.step(action)
        self.progress = 0

        self.lap_number += 1
        
        return obs, done, start_pose

    def save_data_frame(self):
        if os.path.exists(self.path + f"Results_{self.planner_name}.csv"):
            history_df = pd.read_csv(self.path + f"Results_{self.planner_name}.csv")
            history_dict = history_df.to_dict('records')
            for lap_df in self.lap_history:
                idx = history_df.loc[history_df["EntryID"] == lap_df["EntryID"]].index
                if len(idx) == 0:
                    history_dict.append(lap_df)
                else: # replace
                    idx = idx[0]
                    history_dict[idx] = lap_df
            save_df = pd.DataFrame(history_dict)
        else:
            save_df = pd.DataFrame(self.lap_history)
        
        save_df = save_df.sort_values(by=["EntryID"])
        if self.training:
            file_name = self.path + f"RawData_{self.test_id}/TrainingData_{self.test_id}.csv"
        else:
            file_name = self.path + f"Results_{self.planner_name}.csv"

        save_df.to_csv(file_name, index=False, float_format='%.4f')


class F1TenthSim(F1TenthSimBase):
    def __init__(self, map_name, planner_name, test_id, save_detail_history=True, training=False):
        super().__init__(map_name, planner_name, test_id, save_detail_history, training)
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

class F1TenthSim_TrueLocation(F1TenthSimBase):
    def __init__(self, map_name, planner_name, test_id, save_detail_history=True, training=False):
        super().__init__(map_name, planner_name, test_id, save_detail_history, training)
        init_pose = np.append(self.current_state[0:2], self.current_state[4])
        self.scan = self.scan_simulator.scan(init_pose)
    
    def build_observation(self, pose):
        self.scan = self.scan_simulator.scan(pose)
        observation = {"scan": self.scan,
                "vehicle_state": self.dynamics_simulator.state,
                "pose": np.append(self.current_state[0:2], self.current_state[4]),
                "vehicle_speed": self.dynamics_simulator.state[3],
                "collision": self.collision,
                "lap_complete": self.lap_complete,
                "laptime": self.current_time,
                "progress": self.progress}
        return observation

