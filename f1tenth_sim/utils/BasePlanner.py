import numpy as np
import yaml
from argparse import Namespace
import os, shutil

np.printoptions(precision=2, suppress=True)


class BasePlanner:
    def __init__(self, planner_name, test_id, params_name=None, init_folder=True):
        self.name = planner_name
        self.test_id = test_id
        if params_name is None:
            self.planner_params = load_parameter_file(planner_name)
        else:
            self.planner_params = load_parameter_file(params_name)
        if init_folder:
            self.data_root_path = f"Logs/{planner_name}/RawData_{test_id}/"
            ensure_path_exists(f"Logs/{planner_name}/")
            ensure_path_exists(self.data_root_path)
            save_params(self.planner_params, self.data_root_path)
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


def ensure_path_exists(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def load_parameter_file(planner_name):
    file_name = f"params/{planner_name}.yaml"
    with open(file_name, 'r') as file:
        params = yaml.load(file, Loader=yaml.FullLoader)
    return Namespace(**params)

def save_params(params, folder, name="params"):
    file_name = f"{folder}/{name}.yaml"
    with open(file_name, 'w') as file:
        yaml.dump(params, file)
