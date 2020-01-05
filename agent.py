import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from model import Actor, Critic
from noise import OUNoise

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class RandomAgent():
    def __init__(self, action_space, n_agents):
        
        self.action_space = action_space
        self.n_agents = n_agents
    
    def act(self, obs):
        actions = np.random.randn(self.n_agents, self.action_space)
        actions = np.clip(actions, -1, 1)
        return actions
    
    def get_num_agents(self):
        return self.n_agents

    
    
class MADDPG():
    def __init__(self, n_agents, observation_space, action_space, action_range=[-1,1], replay_buffer_size=100000, batch_size=64, gamma=0.99, tau=0.01, actor_lr=1e-4, critic_lr=1e-4, eps_decay=0.01, min_eps=0.05, n_updates=4, learn_every=1, seed=0):

        np.random.seed(seed)
        torch.manual_seed(seed)
        
        if torch.cuda.is_available() : self.device='cuda'
        else : self.device = 'cpu'
        
        self.replay_buffer_size = replay_buffer_size
        self.batch_size = batch_size
        
        self.gamma = gamma
        
        self.learn_every = learn_every
        self.n_updates = n_updates
        self.tau = tau
        
        self.n_agents = n_agents
        self.observation_space = observation_space
        self.action_space = action_space

        self.agents = [DDPGAgent(n_agents=n_agents,
                                 observation_space=observation_space, 
                                 action_space=action_space, 
                                 action_range=action_range, 
                                 actor_lr=actor_lr,
                                 critic_lr=critic_lr,
                                 eps_decay=eps_decay,
                                 min_eps=min_eps,
                                 tau=tau, 
                                 device=self.device, 
                                 seed=seed) for _ in range(self.n_agents)]
        
        self.replay_buffer = ReplayBuffer(action_space=action_space,
                                          observation_space=observation_space,
                                          n_agents=n_agents,
                                          buffer_size=self.replay_buffer_size,
                                          device=self.device)
        
        self.steps = 0

    def reset_noise(self):
        for agent in self.agents:
            agent.reset_noise()
        
    def act(self, state, noise=False):
        state = torch.from_numpy(state).float().to(self.device)
        
        actions = torch.zeros(self.n_agents,self.action_space).float().to(self.device)
        for i in range(self.n_agents):
            actions[i] = self.agents[i].act_local(state[i], noise)
        return actions.detach().cpu().numpy()
        
    def step(self, state, actions, rewards, next_state, done):
        
        state = torch.from_numpy(state).float().to(self.device)
        actions = torch.from_numpy(actions).float().to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_state = torch.from_numpy(next_state).float().to(self.device)
        done = torch.FloatTensor(done).float().to(self.device)
        
        self.replay_buffer.add(state, actions, rewards, next_state, done)

        self.steps += 1
        
        if len(self.replay_buffer) >= self.batch_size and (self.steps%self.learn_every) == 0:
            self.learn()

    def learn(self):
 
        for i_agent in range(self.n_agents):
            samples = self.replay_buffer.sample(self.batch_size, self.n_updates)

            for states, actions, rewards, next_states, dones in samples:

                next_actions = torch.zeros(actions.shape).to(self.device)
                with torch.no_grad():
                    for i in range(self.n_agents):
                        next_actions[:,i,:] = self.agents[i].actor_local(next_states[:,i,:])

                self.agents[i_agent].learn(states=states, 
                                           actions=actions,
                                           rewards=rewards,
                                           next_states=next_states,
                                           next_actions=next_actions,
                                           dones=dones,
                                           i_agent=i_agent)
                                                                                 
    def save(self, folder, suffix):
        for i,agent in enumerate(self.agents):
            agent.save(folder, 'Agent_{}_'.format(i) + suffix)
    
    def load(self, folder, suffix):
        for i,agent in enumerate(self.agents):
            agent.load(folder, 'Agent_{}_'.format(i) + suffix)  
            
            
            

class DDPGAgent():
    def __init__(self, n_agents, observation_space, action_space, action_range=[-1.,1.], actor_lr=1e-4, critic_lr=1e-4, gamma=0.99, tau=0.01, eps_decay=0.01, min_eps=0.1, device='cpu', seed=0):
            
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        self.device = device
        
        self.n_agents = n_agents
        self.observation_space = observation_space
        self.action_space = action_space
        
        self.action_min = action_range[0]
        self.action_max = action_range[1]
        
        self.tau = tau
        self.gamma = gamma
        self.eps = 1.0
        self.eps_decay = eps_decay
        self.min_eps = min_eps
        
        self.noise_generator = OUNoise(self.action_space, scale=0.1, mu=0, theta=0.15, sigma=0.2, device=self.device)
        
        self.actor_local = Actor(self.observation_space, self.action_space).to(self.device)
        self.actor_target = Actor(self.observation_space, self.action_space).to(self.device)
    
        self.critic_local = Critic(self.n_agents, observation_space, action_space).to(self.device)
        self.critic_target = Critic(self.n_agents, observation_space, action_space).to(self.device)         
        
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=critic_lr)        
    
    def act_local(self, observation, noise=False):
        with torch.no_grad():
            action = self.actor_local(observation).squeeze()
            if noise: 
                action += self.eps*self.noise_generator.generate()
            action = torch.clamp(action, self.action_min, self.action_max)
        
        return action
    
    def learn(self, states, actions, rewards, next_states, next_actions, dones, i_agent):

        # Update Critic
        next_q_est = self.critic_target(next_states, next_actions)
        q_targets = rewards[:,i_agent,:] + self.gamma * next_q_est * (1-dones[:,i_agent,:]) 
        q_locals = self.critic_local(states, actions)
        loss = ((q_locals - q_targets)**2).mean()
        
        self.critic_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1.)
        self.critic_optimizer.step()
        
        # Update Actor
        actions[:,i_agent,:] = self.actor_local(states[:,i_agent,:])
        loss = -(self.critic_local(states, actions)).mean()
        
        self.actor_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.actor_local.parameters(), 1.)
        self.actor_optimizer.step()
        
        # Soft Update Target Parameters
        self.soft_update(self.actor_local, self.actor_target, self.tau)
        self.soft_update(self.critic_local, self.critic_target, self.tau)
        
    def reset_noise(self):
        self.eps *= self.eps_decay
        self.eps = max(self.eps, self.min_eps)        
        self.noise_generator.reset()

    def soft_update(self, local_network, target_network, tau):
        for target_param, local_param in zip(target_network.parameters(), local_network.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.-tau)*target_param.data)            

    def save(self, folder, suffix):
        torch.save(self.actor_local.state_dict(), folder + 'Local_actor_' + suffix)
        torch.save(self.critic_local.state_dict(), folder + 'Local_critic_' + suffix)
        torch.save(self.actor_target.state_dict(), folder + 'Target_actor_' + suffix)
        torch.save(self.critic_target.state_dict(), folder + 'Target_critic_' + suffix)
        
    def load(self, folder, suffix):
        self.actor_local.load_state_dict(torch.load(folder + 'Local_actor_' + suffix))
        self.critic_local.load_state_dict(torch.load(folder + 'Local_critic_' + suffix))
        self.actor_target.load_state_dict(torch.load(folder + 'Target_actor_' + suffix))
        self.critic_target.load_state_dict(torch.load(folder + 'Target_critic_' + suffix)) 

class ReplayBuffer:
    def __init__(self, action_space, observation_space, n_agents=1, buffer_size=1000000, device='cpu'):

        self.device = device
        self.buffer_size = buffer_size
                
        self.n_agents = n_agents
        self.action_space = action_space
        self.observation_space = observation_space
        
        self.states = torch.zeros((self.buffer_size, self.n_agents, self.observation_space)).to(self.device)
        self.actions = torch.zeros((self.buffer_size, self.n_agents, self.action_space)).to(self.device)
        self.rewards = torch.zeros((self.buffer_size, self.n_agents, 1)).to(self.device)
        self.next_states = torch.zeros((self.buffer_size, self.n_agents, self.observation_space)).to(self.device) 
        self.dones = torch.zeros((self.buffer_size, self.n_agents, 1)).to(self.device)

        self.reset()
        
    def reset(self):        
        self.memory_steps = 0
        
    def add(self, states, actions, rewards, next_states, dones):
        
        pos = self.memory_steps%self.buffer_size
        
        self.states[pos] = states.reshape(self.states.shape[1:]).detach().to(self.device) 
        self.actions[pos] = actions.reshape(self.actions.shape[1:]).detach().to(self.device) 
        self.rewards[pos] = rewards.reshape(self.rewards.shape[1:]).detach().to(self.device) 
        self.next_states[pos] = next_states.reshape(self.next_states.shape[1:]).detach().to(self.device) 
        self.dones[pos] = dones.reshape(self.dones.shape[1:]).detach().to(self.device) 
        
        self.memory_steps = self.memory_steps+1
            
    def sample(self, batch_size, n_batch):
    
        ms = min(self.memory_steps, self.buffer_size)
        indices = np.random.choice(ms, size=(n_batch, batch_size), replace=True)

        for i in indices:
            batch_states = self.states[:ms][[i]] 
            batch_actions = self.actions[:ms][[i]]
            batch_rewards = self.rewards[:ms][[i]]
            batch_next_states = self.next_states[:ms][[i]] 
            batch_dones = self.dones[:ms][[i]]

            yield batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones
                     
    def __len__(self):
        return min(self.memory_steps,self.buffer_size)