import numpy as np
from f1tenth_sim.classic_racing.ScanSimulator import ScanSimulator2D
from matplotlib import pyplot as plt

NUM_BEAMS = 45
# NUM_BEAMS = 1080 #! TODO: change this to use resampling....
L = 0.33


class ParticleFilter:
    def __init__(self, vehicle_name, NP=100) -> None:
        self.vehicle_name = vehicle_name
        self.estimates = None
        self.scan_simulator = None
        self.Q = np.diag([0.05**2, 0.05**2, 0.05**2])
        self.NP = NP
        self.dt = 0.04
        self.last_true_location = np.zeros(2)

        self.particles = None
        self.proposal_distribution = None
        self.weights = np.ones(self.NP) / self.NP
        self.particle_indices = np.arange(self.NP)

    def init_pose(self, init_pose):
        self.estimates = [init_pose]
        self.proposal_distribution = init_pose + np.random.multivariate_normal(np.zeros(3), self.Q*5, self.NP)
        self.particles = self.proposal_distribution

    def set_map(self, map_name):
        self.scan_simulator = ScanSimulator2D(f"maps/{map_name}", NUM_BEAMS, 4.7)

    def localise(self, action, observation):
        vehicle_speed = observation["vehicle_speed"] 
        self.particle_control_update(action, vehicle_speed)
        plt.figure(1)
        plt.clf()
        self.measurement_update(observation["scan"][::24])

        estimate = np.dot(self.particles.T, self.weights)
        self.estimates.append(estimate)

        self.last_true_location = observation['vehicle_state'][:2]

        return estimate

    def particle_control_update(self, control, vehicle_speed):
        # update the proposal distribution through resampling.

        next_states = particle_dynamics_update(self.proposal_distribution, control, vehicle_speed, self.dt)
        random_samples = np.random.multivariate_normal(np.zeros(3), self.Q, self.NP)
        self.particles = next_states + random_samples

    def measurement_update(self, measurement):
        angles = np.linspace(-4.7/2, 4.7/2, NUM_BEAMS)
        sines = np.sin(angles) 
        cosines = np.cos(angles)
        particle_measurements = np.zeros((self.NP, NUM_BEAMS))
        for i, state in enumerate(self.particles): 
            particle_measurements[i] = self.scan_simulator.scan(state)

        z = particle_measurements - measurement
        # ssd = np.sum(z**2, axis=1)
        # self.weights = np.exp(-ssd / (2*0.5**2))
        sigma = np.clip(np.sqrt(np.average(z**2, axis=0)), 0.01, 10)
        # weights = 1.0 / np.sqrt(2.0 * np.pi * sigma ** 2) * np.exp(-z ** 2 / (2 * sigma ** 2))
        weights =  np.exp(-z ** 2 / (2 * sigma ** 2))
        self.weights = np.prod(weights, axis=1)

        self.weights = self.weights / np.sum(self.weights)

        proposal_indices = np.random.choice(self.particle_indices, self.NP, p=self.weights)
        self.proposal_distribution = self.particles[proposal_indices,:]

    def lap_complete(self):
        estimates = np.array(self.estimates)
        np.save(f"Logs/{self.vehicle_name}/pf_estimates.npy", estimates)
        print(f"Estimates saved in {self.vehicle_name}/pf_estimates.npy")



def particle_dynamics_update(states, actions, speed, dt):
    states[:, 0] += speed * np.cos(states[:, 2]) * dt
    states[:, 1] += speed * np.sin(states[:, 2]) * dt
    states[:, 2] += speed * np.tan(actions[0]) / L * dt
    return states




