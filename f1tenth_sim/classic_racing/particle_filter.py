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
        self.Q = np.diag([0.1**2, 0.1**2, 0.01**2])
        self.NP = NP
        self.dt = 0.05
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
        self.scan_simulator = ScanSimulator2D(f"maps/{map_name}", NUM_BEAMS, 3.14)
        # self.scan_simulator = ScanSimulator2D(f"maps/{map_name}", NUM_BEAMS, 4.7)

    def localise(self, action, observation):
        o_ps = np.copy(self.particles)
        vehicle_speed = observation["vehicle_speed"]
        print(f"Vehicle speed: {vehicle_speed}")
        self.particle_control_update(action, vehicle_speed)
        plt.figure(1)
        plt.clf()
        # plt.plot(self.particles[:,0], self.particles[:,1], 'o', label="PreMeasure", alpha=0.5)
        self.measurement_update(observation["scan"][::24])

        estimate = np.dot(self.particles.T, self.weights)
        self.estimates.append(estimate)

        plt.figure(1)
        plt.plot(o_ps[:,0], o_ps[:,1], 'o', label="Old Particles", alpha=0.5)
        plt.scatter(self.particles[:,0], self.particles[:,1], s=self.weights*1200, label="Particles", alpha=0.5, color='r')
        # plt.plot(self.particles[:,0], self.particles[:,1], 'o', label="Particles", alpha=0.5)
        plt.plot(self.proposal_distribution[:,0], self.proposal_distribution[:,1], 'o', label="Resampled", alpha=0.5)

        plt.plot(estimate[0], estimate[1], '+', markersize=16, label="Estimate")
        plt.plot(self.last_true_location[0], self.last_true_location[1], 'rX', label="old T", markersize=16)
        plt.plot(observation['vehicle_state'][0], observation['vehicle_state'][1], 'bX', label="True", markersize=16)

        plt.legend()
        plt.show()
        plt.pause(0.00001)

        self.last_true_location = observation['vehicle_state'][:2]

        return estimate

    def particle_control_update(self, control, vehicle_speed):
        # update the proposal distribution through resampling.

        next_states = particle_dynamics_update(self.proposal_distribution, control, vehicle_speed, self.dt)
        random_samples = np.random.multivariate_normal(np.zeros(3), self.Q, self.NP)
        self.particles = next_states + random_samples

    def measurement_update(self, measurement):
        measurement = np.clip(measurement, 0, 10)
        angles = np.linspace(-3.14/2, 3.14/2, NUM_BEAMS)
        sines = np.sin(angles) 
        cosines = np.cos(angles)
        particle_measurements = np.zeros((self.NP, NUM_BEAMS))
        plt.figure(2)
        plt.clf()
        plt.plot(measurement * cosines, measurement * sines, 'x-', label="True", alpha=0.5)
        for i, state in enumerate(self.particles): 
            particle_measurements[i] = np.clip(self.scan_simulator.scan(state), 0, 10)

            # if i < 10:
            plt.plot(particle_measurements[i] * cosines, particle_measurements[i] * sines, '.', label=f"Particles {i}", alpha=0.5)
        # plt.legend()
        # plt.show()
        plt.gca().set_aspect('equal', adjustable='box')



        z = particle_measurements - measurement
        sigma = np.clip(np.sqrt(np.average(z**2, axis=0)), 1, 100)
        weights = 1.0 / np.sqrt(2.0 * np.pi * sigma ** 2) * np.exp(-z ** 2 / (2 * sigma ** 2))
        # print(f"Avg: {np.average(weights, axis=1)[:10]}")
        self.weights = np.prod(weights, axis=1)
        # self.weights = np.power(self.weights, 1/2.2)

        weight_sum = np.sum(self.weights, axis=0)
        if (weight_sum == 0).any():
            print(f"Problem with weights")
            raise ValueError("Problem with weights")

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




