import numpy as np
import yaml
from argparse import Namespace


def normalise_psi(psi):
    while psi > np.pi:
        psi -= 2*np.pi
    while psi < -np.pi:
        psi += 2*np.pi
    return psi

def load_mpcc_params():
    filename = "configurations/mpcc_params.yaml"
    with open(filename, 'r') as file:
        params = yaml.load(file, Loader=yaml.FullLoader)
    return Namespace(**params)

