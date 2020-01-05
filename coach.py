import os
import time

import numpy as np

class Coach():
    def __init__(self, env, brain_name, save_directory):
        
        self.env = env
        self.save_directory = save_directory
        self.brain_name = brain_name
        
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

    def train(self, agent, n_episodes, max_steps, log_interval, save_interval):

        start_time = time.time()
        
        all_scores = []
        avg_scores = []

        for i_episode in range(1, n_episodes+1):
    
            agent.reset_noise()
            scores = np.zeros(agent.n_agents)
            env_info = self.env.reset(train_mode=True)[self.brain_name]
            state = env_info.vector_observations

            for t in range(max_steps):

                actions = agent.act(state, noise=True)
                env_info = self.env.step(actions)[self.brain_name]

                next_state = env_info.vector_observations
                rewards = env_info.rewards
                done = env_info.local_done

                agent.step(state, actions, rewards, next_state, done)
                
                state = next_state

                scores += np.array(rewards)
                if any(done):
                    break

            episode_score = scores.max()
            all_scores.append(episode_score)

            avg_score = np.array(all_scores[-100:]).mean()
            avg_scores.append(avg_score)

            completion = (i_episode)/(n_episodes)
            elapsed_time = time.time() - start_time

            em, es = divmod(elapsed_time, 60)
            eh, em = divmod(em, 60)    

            print('\rEpisode: {:4.0f}/{} | Avg.Score: {:3.3f} | Epis.Score: {:3.3f} | Elaps.Time: {:.0f}h {:02.0f}m {:02.0f}s'.format(i_episode, n_episodes, avg_score, episode_score, eh, em, es), end="")
            if i_episode % log_interval == 0: print()
            if i_episode % save_interval == 0: agent.save(self.save_directory,'MADDPG_Episode_{}.pth'.format(i_episode))
            if avg_score >= 0.5: 
                print('\nEnvironment solved after {} episodes'.format(i_episode))
                agent.save(self.save_directory,'MADDPG_Solved.pth')
                break

        return all_scores, avg_scores
        
        
    def watch(self, agent, n_episodes=1):
        
        for i_episode in range(n_episodes):
            env_info = self.env.reset(train_mode=False)[self.brain_name]
            scores = np.zeros(agent.n_agents)

            total_steps = 0
            while True:
                state = env_info.vector_observations
                actions = agent.act(state)
                env_info = self.env.step(actions)[self.brain_name]
                total_steps += 1

                rewards = env_info.rewards
                dones = env_info.local_done
                scores += env_info.rewards

                if np.any(dones): break
            print('Score (max over agents) from episode {}: {:.2f}'.format(i_episode, np.max(scores)))     
    