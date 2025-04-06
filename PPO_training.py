import gymnasium
import highway_env

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import random
import math
import copy
import tqdm
import matplotlib.pyplot as plt

import memoryBuffer as mb

# Constants and parameters #####################################################

STATE_DIMENSIONALITY = 25 # 5 cars * 5 features

BATCH_SIZE = 128
LEARNING_START = 200 


DISCOUNT_FACTOR = 0.8
LR = 5e-4 # learning rate

loss_function =  nn.SmoothL1Loss() #nn.MSELoss() # nn.SmoothL1Loss() 

MAX_STEPS = int(5e4)  # This should be enough to obtain nice results, however feel free to change it

LANES = 3

# Set the seed and create the environment
np.random.seed(2119275)
random.seed(2119275)
torch.manual_seed(2119275)

# GPU? 
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
device = 'cpu'
print(f'>>> DEVICE =  {device}')

# Environment creation 
env_name = "highway-fast-v0"  # We use the 'fast' env just for faster training, if you want you can use "highway-v0"

env = gymnasium.make(env_name,
                     config={
                        'observation':{
                            'type': 'Kinematics',
                            "features": ["presence", "x", "y", "vx", "vy"],
                        },
                        
                        'action': {
                            'type': 'DiscreteMetaAction'
                        },
                        'lanes_count': 3,
                        'absolute': False,
                        'duration': 40, "vehicles_count": 50},
                        #render_mode = 'human'
                        )

print('>>> ENVIRONMENT INITIALIZED')

# Initialize your model


# initialization of the replay buffer
replay_buffer = mb.memory(MAX_STEPS)
print('>>> REPLAY BUFFER INITIALIZED')

state, _ = env.reset()
state = np.squeeze(state.reshape(-1))
done, truncated = False, False

episode = 1
episode_steps = 0
episode_return = 0

training_steps = 0

# training data arrays initialization
rewards_history = []

episode_length_history = []
episode_return_history = []

loss_vals_history = []

# Training loop ################################################################
for t in tqdm.tqdm(range(MAX_STEPS)):
    episode_steps += 1
    # Select the action to be performed by the agent
    

    # creating a torch state tensor to feed to the agent's neural network
    state_tensor = torch.from_numpy(state)
    state_tensor = state_tensor.to(device=device)
  
    # choosing an action following an epsilon greedy policy

     
    available_actions = env.unwrapped.get_available_actions()
    action = 0 # action selection

    # print(f'action at step {t} is: {action}')

    # Hint: take a look at the docs to see the difference between 'done' and 'truncated'
    # done: the agent entered in a terminal state (in this case crash)
    # truncated: the episode stops for external reasons (reached max duration of the episode)
    next_state, reward, done, truncated, _ = env.step(action)
    next_state = next_state.reshape(-1)
    next_state_tensor = torch.from_numpy(next_state)
    next_state_tensor = next_state_tensor.to(device=device)

    # Store transition in memory and train your model
    
    # Storing trnsition in memory
    replay_buffer.push(state_tensor, action, reward, next_state_tensor, done)

    state = next_state
    episode_return += reward

    # TRAINING #################################################################
    if t >= LEARNING_START:
        
        # batch sampling and conversion to tensors #############################
        batch = replay_buffer.sample(batch_size = BATCH_SIZE)

        batch_zip = mb.Transition(*zip(*batch))

        state_batch = torch.cat(batch_zip.state).reshape((-1, STATE_DIMENSIONALITY))       
        action_batch = np.array(batch_zip.action)
        reward_batch = np.array(batch_zip.reward)
        next_state_batch = torch.cat(batch_zip.next_state).reshape((-1, STATE_DIMENSIONALITY))
        done_batch = list(batch_zip.done)

        # moving tensors to device
        state_batch = state_batch.to(device=device)
        next_state_batch = next_state_batch.to(device=device)

        ########################################################################

        

    if done or truncated:
        #print(f"Total T: {t} Episode Num: {episode} Episode T: {episode_steps} Return: {episode_return:.3f}")

        # Save training information and model parameters
        episode_length_history.append(episode_steps)
        episode_return_history.append(episode_return)

        state, _ = env.reset()
        state = state.reshape(-1)
        episode += 1
        episode_steps = 0
        episode_return = 0