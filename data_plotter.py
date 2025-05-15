# Given the csv files in the training_data folder plots the data gathered during training

# usage: python data_plotter.py --input model_name
# model_name alternatives: 
#  - 'DQN'
#  - 'Duelling_DQN'
#  - 'PPO'

# imports
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--input", required=True, type=str)

    args = parser.parse_args()

    input_arg = args.input

# alternatives: 
# 'DQN'
# 'PPO'
MODEL = input_arg

def moving_avg_filter(array:np.array, beta = 0.01):
    """
    Function used to smooth data for better visualization, computes the moving weighted average 

    Parameters
    __________
    
    array: numpy array containing the data
    beta: smoothing factor, the smaller beta the stronger the smoothing

    Returns
    _______

    filt_array: numpy array containing the weighted moving average
    """

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
    plt.plot(DQN_loss, color = 'lavender', label = 'raw data')
    plt.plot(filt_DQN_loss, color = 'blue', label = 'filtered data')
    plt.legend()
    plt.xlabel('Training steps')

    # rewards
    plt.figure('DQN rewards')
    #plt.suptitle('DQN rewards')
    plt.plot(DQN_rewards, color = 'lavender', label = 'raw data')
    plt.plot(filt_DQN_rewards, color = 'blue', label = 'filtered data')
    plt.legend()
    plt.xlabel('Training steps')

    DQN_returns, DQN_episodes_steps = np.genfromtxt('training_data/DQN/DQN_training_data_returns_episode_length.csv', unpack=True, delimiter=',')

    filt_DQN_returns = moving_avg_filter(DQN_returns)
    filt_DQN_episodes_steps = moving_avg_filter(DQN_episodes_steps)

    # returns
    plt.figure('DQN returns')
    #plt.suptitle('DQN returns')
    plt.plot(DQN_returns, color = 'lavender', label = 'raw data')
    plt.plot(filt_DQN_returns, color = 'blue', label = 'filtered data')
    plt.legend()
    plt.xlabel('Episodes')

    # episode steps
    plt.figure('DQN episodes steps')
    #plt.suptitle('DQN episode steps')
    plt.plot(DQN_episodes_steps, color = 'lavender', label = 'raw data')
    plt.plot(filt_DQN_episodes_steps, color = 'blue', label = 'filtered data')
    plt.legend()
    plt.xlabel('Episodes')

# PPO ############################################################
if MODEL == 'PPO':
    PPO_actor_loss = np.genfromtxt('training_data/PPO/PPO_training_data_actor_loss.csv', unpack=False, delimiter=',')
    PPO_critic_loss = np.genfromtxt('training_data/PPO/PPO_training_data_critic_loss.csv', unpack=False, delimiter=',')

    filt_PPO_actor_loss = moving_avg_filter(PPO_actor_loss, 0.025)
    filt_PPO_critic_loss = moving_avg_filter(PPO_critic_loss, 0.025)


    # PPO actor loss
    plt.figure('PPO actor loss')
    #plt.suptitle('PPO actor loss')
    plt.plot(PPO_actor_loss, color = 'lavender', label = 'raw data')
    plt.plot(filt_PPO_actor_loss, color = 'blue', label = 'filtered data')
    plt.legend()
    plt.xlabel('Training steps')

    # PPO critic loss
    plt.figure('PPO critic loss')
    #plt.suptitle('PPO critic loss')
    plt.plot(PPO_critic_loss, color = 'lavender', label = 'raw data')
    plt.plot(filt_PPO_critic_loss, color = 'blue', label = 'filtered data')
    plt.legend()
    plt.xlabel('Training steps')


    PPO_returns, PPO_episodes_steps = np.genfromtxt('training_data/PPO/PPO_training_data_returns_and_episodes_steps.csv', unpack=True, delimiter=',')

    filt_PPO_returns = moving_avg_filter(PPO_returns)
    filt_PPO_episodes_steps = moving_avg_filter(PPO_episodes_steps)

    # returns
    plt.figure('PPO returns')
    #plt.suptitle('PPO returns')
    plt.plot(PPO_returns, color = 'lavender', label = 'raw data')
    plt.plot(filt_PPO_returns, color = 'blue', label = 'filtered data')
    plt.legend()
    plt.xlabel('Episodes')

    # episode steps
    plt.figure('PPO episodes steps')
    #plt.suptitle('PPO episode steps')
    plt.plot(PPO_episodes_steps, color = 'lavender', label = 'raw data')
    plt.plot(filt_PPO_episodes_steps, color = 'blue', label = 'filtered data')
    plt.legend()
    plt.xlabel('Episodes')

plt.show()