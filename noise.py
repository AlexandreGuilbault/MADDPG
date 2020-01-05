import torch
import numpy as np

# Slightly modified from https://github.com/songrotek/DDPG/blob/master/ou_noise.py
class OUNoise:
    def __init__(self, action_space, scale=0.1, mu=0., theta=0.15, sigma=0.2, device='cpu'):
        self.device = device
    
        self.action_space = action_space
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_space) * self.mu

        self.reset()

    def reset(self):
        self.state = np.ones(self.action_space) * self.mu

    def generate(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        
        return torch.tensor(self.state * self.scale).float().to(self.device)