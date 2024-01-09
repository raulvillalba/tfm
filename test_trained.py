from bufferReplay import experienceReplayBuffer
from DQN import DQN
from DQNagent import DQNAgent
from carEnv import CarEnv
import matplotlib.pyplot as plt
import torch

import time




def plot_rewards(agent,lr, gamma, batch_size, dnn_update_frequency, dnn_sync_frequency, memory_size, neurons):
        plt.figure(figsize=(12,8))
        plt.plot(agent.training_rewards, label='Rewards')
        plt.plot(agent.mean_training_rewards, label='Mean Rewards')
        plt.axhline(agent.reward_threshold, color='r', label="Reward threshold")
        plt.xlabel('Episodes')
        plt.ylabel('Rewards')
        plt.legend(loc="upper left")
        file_name = "figures/from_trained_seven_no_sleep_rewards_" + str(lr) + str(memory_size) +  "_" + str(neurons) + "_" + str(gamma) +  "_" + str(batch_size) + "_" + str(dnn_update_frequency) + "_" + str(dnn_sync_frequency) + ".png"
        plt.savefig(file_name)
        #plt.show()
 
def plot_loss(agent,lr, gamma, batch_size, dnn_update_frequency, dnn_sync_frequency, memory_size, neurons):
        plt.figure(figsize=(12,8))
        plt.plot(agent.training_loss, label='Loss')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.legend(loc="upper left")
        file_name = "figures/from_trained_seven_no_sleep_loss" + str(lr) + str(memory_size) +  "_" + str(neurons) + "_" + str(gamma) + "_" + str(batch_size) + "_" + str(dnn_update_frequency) + "_" + str(dnn_sync_frequency) + ".png"
        plt.savefig(file_name)
        #plt.show()



lr_list = [0.0005]#, 0.001, 0.01]            #Learning rate
BATCH_SIZE_list = [32]#[8,16,32,64]       #Batch size
MEMORY_SIZE_list = [50000] #[100000, 50000, 2000]  #Max buffer size 
GAMMA = 0.99          #Gamma value for Bellman's equation
EPSILON = 1           #Starting epsilon value
EPSILON_DECAY = 0.995 #Epsilon decay
BURN_IN = 100         #Number of episodes used to fill the buffer before training 
MAX_EPISODES = 500   #Max number of episodes
MIN_EPISODES = 250    #Min number of episodes
DNN_UPD = 4 #100         #Neural network update frequency
DNN_SYNC = 500        #Neural networks syncronization frequency
NEURONS_list = [512]#, 256] #[16,32,64]          #Number of neurons on neural network's hidden layers
 
reward_threshold = 1000


for lr  in lr_list:
    for BATCH_SIZE in BATCH_SIZE_list:
        for MEMORY_SIZE in MEMORY_SIZE_list:
            for NEURONS in NEURONS_list:

                tic = time.perf_counter() 
                env = CarEnv()

                buffer = experienceReplayBuffer(memory_size=MEMORY_SIZE, burn_in=BURN_IN)

                dqn = DQN(env, learning_rate=lr, neurons=NEURONS)

                file_name = "trained_agents/last_agentDQN_Trained_Model_dqn_cnn0.000550000_512_0.99_32_4_500.pth"       
                dqn.load_state_dict(torch.load(file_name))

                agent = DQNAgent(env, dqn, buffer, reward_threshold, EPSILON, EPSILON_DECAY, BATCH_SIZE)

                games = 10
                rewards = []
                n_steps = []
                t, total_reward, done = 0, 0, False
                for i in range(0, games):
                    agent.initialize()
                    t, total_reward, done = 0, 0, False

                    while not done:
                        done = agent.take_step(0)

                        if done:
                            print(agent.total_reward)
                            rewards.append(agent.total_reward)
                            n_steps.append(agent.step_count)

            print(rewards)
            print(n_steps)
                    