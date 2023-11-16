import numpy as np
import yaml
from argparse import Namespace
import os, shutil

np.printoptions(precision=2, suppress=True)


def ensure_path_exists(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def load_parameter_file(planner_name):
    file_name = f"f1tenth_sim/params/{planner_name}.yaml"
    with open(file_name, 'r') as file:
        params = yaml.load(file, Loader=yaml.FullLoader)
    return Namespace(**params)


class BasePlanner:
    def __init__(self, planner_name, test_id):
        self.name = planner_name
        self.test_id = test_id
        self.data_root_path = f"Logs/{planner_name}/RawData_{test_id}/"
        ensure_path_exists(f"Logs/{planner_name}/")
        ensure_path_exists(self.data_root_path)
        self.planner_params = load_parameter_file(planner_name)
        self.vehicle_params = load_parameter_file("vehicle_params")
        self.map_name = None

        self.step_counter = 0

    def set_map(self, map_name):
        self.map_name = map_name
        pass

    def plan(self, obs):
        raise NotImplementedError
    
    def create_clean_path(self, path):
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)


