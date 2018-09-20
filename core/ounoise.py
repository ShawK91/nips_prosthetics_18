import numpy as np


# from https://github.com/songrotek/DDPG/blob/master/ou_noise.py
class OUNoise:

    def __init__(self, action_dimension, scale=0.1, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state * self.scale



def get_list_generators(num_generators, action_dim):
    NUM_REPLICATES = 3
    noise_gens = []

    for _ in range(NUM_REPLICATES): noise_gens.append(OUNoise(action_dim, scale=0.1, mu = 0.0, theta=0.15, sigma=0.2))

    for _ in range(NUM_REPLICATES): noise_gens.append(OUNoise(action_dim, scale=0.15, mu = 0.0, theta=0.15, sigma=0.5))

    for _ in range(NUM_REPLICATES): noise_gens.append(OUNoise(action_dim, scale=0.3, mu = 0.0, theta=0.15, sigma=0.2))

    for _ in range(NUM_REPLICATES): noise_gens.append(OUNoise(action_dim, scale=0.1, mu = 0.0, theta=0.15, sigma=0.9))

    for _ in range(NUM_REPLICATES): noise_gens.append(OUNoise(action_dim, scale=0.1, mu = 0.0, theta=0.15, sigma=0.5))

    for _ in range(NUM_REPLICATES): noise_gens.append(OUNoise(action_dim, scale=0.2, mu = 0.0, theta=0.15, sigma=0.1))

    for _ in range(NUM_REPLICATES): noise_gens.append(OUNoise(action_dim, scale=0.3, mu = 0.0, theta=0.15, sigma=0.5))


    #IF anything left
    for i in range(num_generators - len(noise_gens)):
        noise_gens.append(noise_gens.append(OUNoise(action_dim, scale=0.1, mu = 0.0, theta=0.15, sigma=0.2)))

    return noise_gens