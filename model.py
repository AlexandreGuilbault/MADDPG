import torch
import torch.nn as nn

def init_weights(m):
    classname = m.__class__.__name__
    if 'conv' in classname.lower() or 'linear' in classname.lower():       
        nn.init.xavier_uniform_(m.weight.data)

class Actor(nn.Module):
    def __init__(self, observation_space, action_space):
        super().__init__()

        self.observation_space = observation_space
        self.action_space = action_space
        
        self.network = nn.Sequential(
                nn.Linear(self.observation_space, self.observation_space*16),
                nn.ReLU(),
                nn.Linear(self.observation_space*16, self.observation_space*8),
                nn.ReLU(),
                nn.Linear(self.observation_space*8, self.action_space),
                nn.Tanh()
            ).apply(init_weights)
        
    def forward(self, x):
        x = x.reshape(-1, self.observation_space)
        return self.network(x)
    
class Critic(nn.Module):
    def __init__(self, n_agents, observation_space, action_space):
        super().__init__()

        self.n_agents = n_agents
        self.observation_space = observation_space
        self.action_space = action_space
        
        self.full_observation_space = self.n_agents*self.observation_space
        self.full_action_space = self.n_agents*self.action_space
        
        self.network = nn.Sequential(
                nn.Linear(self.full_observation_space+self.full_action_space, self.full_observation_space*16),
                nn.ReLU(),
                nn.Linear(self.full_observation_space*16, self.full_observation_space*8),
                nn.ReLU(),
                nn.Linear(self.full_observation_space*8, 1)
            ).apply(init_weights)
        
    def forward(self, obs, action):
        obs = obs.reshape(-1, self.full_observation_space)
        action = action.reshape(-1, self.full_action_space)
        
        return self.network(torch.cat([obs,action], dim=1))