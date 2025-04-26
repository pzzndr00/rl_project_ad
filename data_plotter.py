import numpy as np
import matplotlib.pyplot as plt

# alternatives: 
# 'DQN'
# 'Duelling_DQN'
# 'PPO'
MODEL = 'PPO'

def moving_avg_filter(array:np.array, beta = 0.01):

    filt_array = np.zeros_like(array)
    l = array.shape[0]
    for i in range(l):
        if i == 0:
            filt_array[i]=array[i]
        else:
            filt_array[i] = filt_array[i-1] + beta *(array[i] - filt_array[i-1])

    return filt_array

# DQN ############################################################
if MODEL == 'DQN':
    DQN_loss, DQN_rewards = np.genfromtxt('training_data/DQN/DQN_training_data_loss_and_rewards.csv', unpack=True, delimiter=',')

    filt_DQN_loss = moving_avg_filter(DQN_loss)
    filt_DQN_rewards = moving_avg_filter(DQN_rewards)

    # loss
    plt.figure('DQN loss')
    #plt.suptitle('DQN loss')
    plt.plot(DQN_loss, color = 'mistyrose', label = 'raw data')
    plt.plot(filt_DQN_loss, color = 'red', label = 'filtered data')
    plt.legend()
    plt.xlabel('Training steps')

    # rewards - not the most useful
    plt.figure('DQN rewards')
    #plt.suptitle('DQN rewards')
    plt.plot(DQN_rewards, color = 'mistyrose', label = 'raw data')
    plt.plot(filt_DQN_rewards, color = 'red', label = 'filtered data')
    plt.legend()
    plt.xlabel('Training steps')

    DQN_returns, DQN_episodes_steps = np.genfromtxt('training_data/DQN/DQN_training_data_returns_episode_length.csv', unpack=True, delimiter=',')

    filt_DQN_returns = moving_avg_filter(DQN_returns)
    filt_DQN_episodes_steps = moving_avg_filter(DQN_episodes_steps)

    # returns
    plt.figure('DQN returns')
    #plt.suptitle('DQN returns')
    plt.plot(DQN_returns, color = 'mistyrose', label = 'raw data')
    plt.plot(filt_DQN_returns, color = 'red', label = 'filtered data')
    plt.legend()
    plt.xlabel('Episodes')

    # episode steps
    plt.figure('DQN episodes steps')
    #plt.suptitle('DQN episode steps')
    plt.plot(DQN_episodes_steps, color = 'mistyrose', label = 'raw data')
    plt.plot(filt_DQN_episodes_steps, color = 'red', label = 'filtered data')
    plt.legend()
    plt.xlabel('Episodes')

# Duelling DQN
if MODEL == 'Duelling_DQN':
    Duelling_DQN_loss, Duelling_DQN_rewards = np.genfromtxt('training_data/DQN/Duelling_DQN_training_data_loss_and_rewards.csv', unpack=True, delimiter=',')

    filt_Duelling_DQN_loss = moving_avg_filter(Duelling_DQN_loss)
    filt_Duelling_DQN_rewards = moving_avg_filter(Duelling_DQN_rewards)

    # loss
    plt.figure('Duelling_DQN loss')
    #plt.suptitle('DQN loss')
    plt.plot(Duelling_DQN_loss, color = 'mistyrose', label = 'raw data')
    plt.plot(filt_Duelling_DQN_loss, color = 'red', label = 'filtered data')
    plt.legend()
    plt.xlabel('Training steps')

    # rewards - not the most useful
    plt.figure('Duelling_DQN rewards')
    #plt.suptitle('DQN rewards')
    plt.plot(Duelling_DQN_rewards, color = 'mistyrose', label = 'raw data')
    plt.plot(filt_Duelling_DQN_rewards, color = 'red', label = 'filtered data')
    plt.legend()
    plt.xlabel('Training steps')

    Duelling_DQN_returns, Duelling_DQN_episodes_steps = np.genfromtxt('training_data/DQN/Duelling_DQN_training_data_returns_episode_length.csv', unpack=True, delimiter=',')

    filt_Duelling_DQN_returns = moving_avg_filter(Duelling_DQN_returns)
    filt_Duelling_DQN_episodes_steps = moving_avg_filter(Duelling_DQN_episodes_steps)

    # returns
    plt.figure('Duelling_DQN returns')
    #plt.suptitle('DQN returns')
    plt.plot(Duelling_DQN_returns, color = 'mistyrose', label = 'raw data')
    plt.plot(filt_Duelling_DQN_returns, color = 'red', label = 'filtered data')
    plt.legend()
    plt.xlabel('Episodes')

    # episode steps
    plt.figure('Duelling_DQN episodes steps')
    #plt.suptitle('DQN episode steps')
    plt.plot(Duelling_DQN_episodes_steps, color = 'mistyrose', label = 'raw data')
    plt.plot(filt_Duelling_DQN_episodes_steps, color = 'red', label = 'filtered data')
    plt.legend()
    plt.xlabel('Episodes')

# PPO ############################################################
if MODEL == 'PPO':
    PPO_actor_loss, PPO_critic_loss = np.genfromtxt('training_data/PPO/PPO_training_data_losses.csv', unpack=True, delimiter=',')

    filt_PPO_actor_loss = moving_avg_filter(PPO_actor_loss)
    filt_PPO_critic_loss = moving_avg_filter(PPO_critic_loss)


    # PPO actor loss
    plt.figure('PPO actor loss')
    #plt.suptitle('PPO actor loss')
    plt.plot(PPO_actor_loss, color = 'mistyrose', label = 'raw data')
    plt.plot(filt_PPO_actor_loss, color = 'red', label = 'filtered data')
    plt.legend()
    plt.xlabel('Training steps')

    # PPO critic loss
    plt.figure('PPO critic loss')
    #plt.suptitle('PPO critic loss')
    plt.plot(PPO_critic_loss, color = 'mistyrose', label = 'raw data')
    plt.plot(filt_PPO_critic_loss, color = 'red', label = 'filtered data')
    plt.legend()
    plt.xlabel('Training steps')


    PPO_returns, PPO_episodes_steps = np.genfromtxt('training_data/PPO/PPO_training_data_returns_and_episodes_steps.csv', unpack=True, delimiter=',')

    filt_PPO_returns = moving_avg_filter(PPO_returns)
    filt_PPO_episodes_steps = moving_avg_filter(PPO_episodes_steps)

    # returns
    plt.figure('PPO returns')
    #plt.suptitle('PPO returns')
    plt.plot(PPO_returns, color = 'mistyrose', label = 'raw data')
    plt.plot(filt_PPO_returns, color = 'red', label = 'filtered data')
    plt.legend()
    plt.xlabel('Episodes')

    # episode steps
    plt.figure('PPO episodes steps')
    #plt.suptitle('PPO episode steps')
    plt.plot(PPO_episodes_steps, color = 'mistyrose', label = 'raw data')
    plt.plot(filt_PPO_episodes_steps, color = 'red', label = 'filtered data')
    plt.legend()
    plt.xlabel('Episodes')

plt.show()