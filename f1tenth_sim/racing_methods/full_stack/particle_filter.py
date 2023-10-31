import numpy as np
from f1tenth_sim.racing_methods.full_stack.ScanSimulator import ScanSimulator2D

NUM_BEAMS = 45
# NUM_BEAMS = 1080 #! TODO: change this to use resampling....
L = 0.33


class ParticleFilter:
    def __init__(self, vehicle_name, NP=100) -> None:
        self.vehicle_name = vehicle_name
        self.estimates = None
        self.scan_simulator = None
        self.Q = np.diag([0.01**2, 0.01**2, 0.001**2])
        self.NP = NP
        self.dt = 0.05

        self.particles = None
        self.proposal_distribution = None
        self.weights = np.ones(self.NP) / self.NP
        self.particle_indices = np.arange(self.NP)

    def init_pose(self, init_pose):
        self.estimates = [init_pose]
        self.proposal_distribution = init_pose + np.random.multivariate_normal(np.zeros(3), self.Q*1, self.NP)
        self.particles = self.proposal_distribution

    def set_map(self, map_name):
        self.scan_simulator = ScanSimulator2D(f"maps/{map_name}", NUM_BEAMS, 4.7*np.pi)

    def localise(self, action, observation):
        self.particle_control_update(action)
        self.measurement_update(observation["scan"][::24])

        estimate = np.dot(self.particles.T, self.weights)
        self.estimates.append(estimate)

        return estimate

    def particle_control_update(self, control):
        # update the proposal distribution through resampling.
        proposal_indices = np.random.choice(self.particle_indices, self.NP, p=self.weights)
        self.proposal_distribution = self.particles[proposal_indices,:]

        next_states = particle_dynamics_update(self.proposal_distribution, control, self.dt)
        random_samples = np.random.multivariate_normal(np.zeros(3), self.Q, self.NP)
        self.particles = next_states + random_samples

    def measurement_update(self, measurement):
        particle_measurements = np.zeros((self.NP, NUM_BEAMS))
        for i, state in enumerate(self.particles): 
            particle_measurements[i] = self.scan_simulator.scan(state)

        z = particle_measurements - measurement
        sigma = np.sqrt(np.average(z**2, axis=0))
        weights = 1.0 / np.sqrt(2.0 * np.pi * sigma ** 2) * np.exp(-z ** 2 / (2 * sigma ** 2))
        # print(f"Avg: {np.average(weights, axis=1)[:10]}")
        self.weights = np.prod(weights, axis=1) 
        self.weights = np.power(self.weights, 1/5.2)

        weight_sum = np.sum(self.weights, axis=0)
        if (weight_sum == 0).any():
            print(f"Problem with weights")
            raise ValueError("Problem with weights")

        self.weights = self.weights / np.sum(self.weights)

    def lap_complete(self):
        estimates = np.array(self.estimates)
        np.save(f"Logs/{self.vehicle_name}/pf_estimates.npy", estimates)
        print(f"Estimates saved in {self.vehicle_name}/pf_estimates.npy")



def particle_dynamics_update(states, actions, dt):
    states[:, 0] += actions[1] * np.cos(states[:, 2]) * dt
    states[:, 1] += actions[1] * np.sin(states[:, 2]) * dt
    states[:, 2] += actions[1] * np.tan(actions[0]) / L * dt
    return states




