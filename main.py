from bufferReplay import experienceReplayBuffer
from DQN import DQN
from DQNagent import DQNAgent
from carEnv import CarEnv
import matplotlib.pyplot as plt
import torch


def plot_rewards(agent,lr, gamma, batch_size, dnn_update_frequency, dnn_sync_frequency, memory_size, neurons):
        plt.figure(figsize=(12,8))
        plt.plot(agent.training_rewards, label='Rewards')
        plt.plot(agent.mean_training_rewards, label='Mean Rewards')
        plt.axhline(agent.reward_threshold, color='r', label="Reward threshold")
        plt.xlabel('Episodes')
        plt.ylabel('Rewards')
        plt.legend(loc="upper left")
        file_name = "figures/six_no_sleep_rewards_" + str(lr) + str(memory_size) +  "_" + str(neurons) + "_" + str(gamma) +  "_" + str(batch_size) + "_" + str(dnn_update_frequency) + "_" + str(dnn_sync_frequency) + ".png"
        plt.savefig(file_name)
        #plt.show()
 
def plot_loss(agent,lr, gamma, batch_size, dnn_update_frequency, dnn_sync_frequency, memory_size, neurons):
        plt.figure(figsize=(12,8))
        plt.plot(agent.training_loss, label='Loss')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.legend(loc="upper left")
        file_name = "figures/six_no_sleep_loss" + str(lr) + str(memory_size) +  "_" + str(neurons) + "_" + str(gamma) + "_" + str(batch_size) + "_" + str(dnn_update_frequency) + "_" + str(dnn_sync_frequency) + ".png"
        plt.savefig(file_name)
        #plt.show()




lr_list = [0.0005]#, 0.001, 0.01]            #Learning rate
BATCH_SIZE_list = [32]#[8,16,32,64]       #Batch size
MEMORY_SIZE_list = [50000] #[100000, 50000, 2000]  #Max buffer size 
GAMMA = 0.99          #Gamma value for Bellman's equation
EPSILON = 1           #Starting epsilon value
EPSILON_DECAY = 0.995 #Epsilon decay
BURN_IN = 100         #Number of episodes used to fill the buffer before training 
MAX_EPISODES = 10000   #Max number of episodes
MIN_EPISODES = 250    #Min number of episodes
DNN_UPD = 4 #100         #Neural network update frequency
DNN_SYNC = 500        #Neural networks syncronization frequency
NEURONS_list = [512]#, 256] #[16,32,64]          #Number of neurons on neural network's hidden layers
 
reward_threshold = 1000


for lr  in lr_list:
    for BATCH_SIZE in BATCH_SIZE_list:
        for MEMORY_SIZE in MEMORY_SIZE_list:
            for NEURONS in NEURONS_list:

                env = CarEnv()

                buffer = experienceReplayBuffer(memory_size=MEMORY_SIZE, burn_in=BURN_IN)
                
                dqn = DQN(env, learning_rate=lr, neurons=NEURONS)
                
                agent = DQNAgent(env, dqn, buffer, reward_threshold, EPSILON, EPSILON_DECAY, BATCH_SIZE)

                agent.train(gamma=GAMMA, max_episodes=MAX_EPISODES, batch_size=BATCH_SIZE, dnn_update_frequency=DNN_UPD, dnn_sync_frequency=DNN_SYNC)

                file_name = "trained_agents/six_no_sleep_agentDQN_Trained_Model_dqn_cnn" + str(lr) + str(MEMORY_SIZE) +  "_" + str(NEURONS) + "_" + str(GAMMA) + "_" + str(BATCH_SIZE) + "_" + str(DNN_UPD) + "_" + str(DNN_SYNC) + ".pth"
                torch.save(agent.main_network.state_dict(), file_name)

                plot_rewards(agent,lr = lr, gamma=GAMMA, batch_size=BATCH_SIZE, dnn_update_frequency=DNN_UPD, dnn_sync_frequency=DNN_SYNC, memory_size=MEMORY_SIZE, neurons=NEURONS)
                plot_loss(agent,lr = lr, gamma=GAMMA, batch_size=BATCH_SIZE, dnn_update_frequency=DNN_UPD, dnn_sync_frequency=DNN_SYNC, memory_size=MEMORY_SIZE, neurons=NEURONS)
