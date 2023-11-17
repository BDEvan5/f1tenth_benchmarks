import numpy as np
from f1tenth_sim.classic_racing.ScanSimulator import ScanSimulator2D
from matplotlib import pyplot as plt
from f1tenth_sim.general_utils import load_parameter_file

L = 0.33


class ParticleFilter:
    def __init__(self, planner_name, test_id) -> None:
        self.params = load_parameter_file("particle_filter_params")
        self.planner_name = planner_name
        self.test_id = test_id
        self.data_path = f"Logs/{planner_name}/RawData_{test_id}/"
        self.estimates = None
        self.scan_simulator = None
        self.Q = np.diag([0.05**2, 0.05**2, 0.05**2])
        self.NP = self.params.number_of_particles
        self.dt = 0.04
        self.num_beams = self.params.number_of_beams
        self.lap_number = 0

        self.particles = None
        self.proposal_distribution = None
        self.weights = np.ones(self.NP) / self.NP
        self.particle_indices = np.arange(self.NP)

    def init_pose(self, init_pose):
        self.estimates = [init_pose]
        self.proposal_distribution = init_pose + np.random.multivariate_normal(np.zeros(3), self.Q*5, self.NP)
        self.particles = self.proposal_distribution

        return init_pose

    def set_map(self, map_name):
        self.scan_simulator = ScanSimulator2D(f"maps/{map_name}", self.num_beams, 4.7)

    def localise(self, action, observation):
        vehicle_speed = observation["vehicle_speed"] 
        self.particle_control_update(action, vehicle_speed)
        plt.figure(1)
        plt.clf()
        self.measurement_update(observation["scan"][::24])

        estimate = np.dot(self.particles.T, self.weights)
        self.estimates.append(estimate)

        return estimate

    def particle_control_update(self, control, vehicle_speed):
        # update the proposal distribution through resampling.

        next_states = particle_dynamics_update(self.proposal_distribution, control, vehicle_speed, self.dt)
        random_samples = np.random.multivariate_normal(np.zeros(3), self.Q, self.NP)
        self.particles = next_states + random_samples

    def measurement_update(self, measurement):
        angles = np.linspace(-4.7/2, 4.7/2, self.num_beams)
        sines = np.sin(angles) 
        cosines = np.cos(angles)
        particle_measurements = np.zeros((self.NP, self.num_beams))
        for i, state in enumerate(self.particles): 
            particle_measurements[i] = self.scan_simulator.scan(state)

        z = particle_measurements - measurement
        sigma = np.clip(np.sqrt(np.average(z**2, axis=0)), 0.01, 10)
        weights =  np.exp(-z ** 2 / (2 * sigma ** 2))
        self.weights = np.prod(weights, axis=1)

        self.weights = self.weights / np.sum(self.weights)

        proposal_indices = np.random.choice(self.particle_indices, self.NP, p=self.weights)
        self.proposal_distribution = self.particles[proposal_indices,:]

    def lap_complete(self):
        estimates = np.array(self.estimates)
        np.save(self.data_path + f"pf_estimates_{self.lap_number}.npy", estimates)
        self.lap_number += 1



def particle_dynamics_update(states, actions, speed, dt):
    states[:, 0] += speed * np.cos(states[:, 2]) * dt
    states[:, 1] += speed * np.sin(states[:, 2]) * dt
    states[:, 2] += speed * np.tan(actions[0]) / L * dt
    return states




