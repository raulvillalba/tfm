import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim

import numpy as np
import random

from copy import deepcopy, copy

class DQNAgent:
    def __init__(self, env, main_network,
                 buffer, reward_threshold,
                 epsilon=0.1, eps_decay=0.99, batch_size=32, nblock=100):
        """"
        Params
        ======
        env: environment
        main_network: neural network implementation 
        target_network: target neural network implementation
        buffer: experience buffer 
        epsilon: epsilon
        eps_decay: epsilon decay
        batch_size: batch size
        nblock: block of X last episodes which will be used to compute the mean of rewards
        reward_threshold: reward threshold
        """
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        self.env = env
        self.main_network = main_network
        self.target_network = deepcopy(main_network) # target network (copy of main network)
        self.buffer = buffer
        self.epsilon = epsilon
        self.eps_decay = eps_decay
        self.batch_size = batch_size
        self.nblock = nblock
        self.reward_threshold = reward_threshold
        self.initialize()

    def initialize(self):
        self.sync_eps = []
        self.total_reward = 0
        self.step_count = 0
        self.state0 = self.env.reset()

        self.update_loss = []
        self.training_rewards = []
        self.mean_training_rewards = []
        self.epsilon_values = []
        self.training_loss = []
        



    
    ## Apply new action 
    def take_step(self, eps, mode='train'):
        self.state0 = self.env.get_state() 
        if mode == 'explore':
            action = random.randint(0, 3) # Random action during burn-in
        else:
            action = self.main_network.get_action(self.state0, eps) # Action from Q-value 
            self.step_count += 1

        new_state, reward, done, _ = self.env.step(action)
        self.total_reward += reward
        self.buffer.append(self.state0, action, reward, done, new_state) # save the experience on the buffer 
        self.state0 = new_state

        # Reset environment if done
        if done:
            self.state0 = self.env.reset()
        return done



    ## Training
    def train(self, gamma=0.99, max_episodes=50000,
              batch_size=32,
              dnn_update_frequency=4,
              dnn_sync_frequency=2000, min_episodios=250, min_epsilon = 0.01):
        self.gamma = gamma
        # Fill the buffer with N random experiences
        print("Filling replay buffer...")
        while self.buffer.burn_in_capacity() < 1:
            self.take_step(self.epsilon, mode='explore')

        episode = 0
        training = True
        print("Training...")
        maximo = -9999999
        while training:
            self.state0 = self.env.reset()
            self.total_reward = 0
            gamedone = False
            while gamedone == False:
                # The agent takes an action
                gamedone = self.take_step(self.epsilon, mode='train')

                ## Main network update at defined frequency
                if self.step_count % dnn_update_frequency == 0:
                    self.update()
                
                # Syncronization of main and target networks at defined frequency 
                if self.step_count % dnn_sync_frequency == 0:
                    self.target_network.load_state_dict(
                        self.main_network.state_dict())
                    self.sync_eps.append(episode)

                if gamedone:
                    episode += 1
                    self.training_rewards.append(self.total_reward) 
                    self.epsilon_values.append(self.epsilon) # guardamos epsilon
              
                    self.update_loss = []
                    
                    # Compute reward mean from X last episodes 
                    mean_rewards = np.mean(self.training_rewards[-self.nblock:])
                    self.mean_training_rewards.append(mean_rewards)
                    if mean_rewards > maximo:
                        maximo = mean_rewards
                
                    print("\rEpisode {:d} Mean Rewards {:.2f} Epsilon {} , Maximo {:.2f}\t\t".format(
                        episode, mean_rewards, self.epsilon,maximo), end="")

                    if episode >= max_episodes:
                        training = False
                        print('\nEpisode limit reached.')
                        break

                    # End training if reward threshold reached
                    if mean_rewards >= self.reward_threshold and min_episodios <  episode:
                        training = False
                        print('\nEnvironment solved in {} episodes!'.format(
                            episode))
                        break

                    ## Update epsilon
                    self.epsilon =  max(self.epsilon * self.eps_decay, 0.01)


    def calculate_loss(self, batch):
        states, actions, rewards, dones, next_states = [i for i in batch]
        rewards_vals = torch.FloatTensor(rewards).to(device=self.device)
        actions_vals = torch.LongTensor(np.array(actions)).to(device=self.device).reshape(-1,1)
        dones_t = torch.ByteTensor(dones).to(device=self.device)

        # Main network Q-values 
        qvals = torch.gather(self.main_network.get_qvals(states), 1, actions_vals)
        # Target network Q-values 
        qvals_next = torch.max(self.target_network.get_qvals(next_states),
                               dim=-1)[0].detach()
        
        qvals_next[dones_t] = 0 

        
        # Bellman equation
        expected_qvals = self.gamma * qvals_next + rewards_vals


        loss = torch.nn.MSELoss()(qvals, expected_qvals.reshape(-1,1))
        return loss


    def update(self):
        self.main_network.optimizer.zero_grad()  
        batch = self.buffer.sample_batch(batch_size=self.batch_size) 
        loss = self.calculate_loss(batch)
        loss.backward() 
        self.main_network.optimizer.step() 
        
        if self.device == 'cuda':
            self.update_loss.append(loss.detach().cpu().numpy())
            self.training_loss.append(loss.detach().cpu().numpy())
        else:
            self.update_loss.append(loss.detach().numpy())
            self.training_loss.append(loss.detach().numpy())