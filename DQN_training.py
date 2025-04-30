import gymnasium
import highway_env

import numpy as np

import torch
import torch.nn as nn

import random
import math
import copy
import tqdm
import matplotlib.pyplot as plt
import os

# my modules
import memoryBuffer as mb
import DQN

# Set the seed and create the environment
np.random.seed(0)
random.seed(0)
torch.manual_seed(0)

# Constants and parameters #####################################################

MAX_STEPS = int(2.5e4)

LANES = 3

STATE_DIMENSIONALITY = 25 # 5 cars * 5 features

BATCH_SIZE = 128
LEARNING_START = 256

# epsilon decay parameters
EPS_START = 0.95
EPS_END = 0.05
EPS_DECAY = MAX_STEPS/20

DISCOUNT_FACTOR = 0.85 # better results when 0.7 - 0.8 rather than > 0.8
LR = 5e-4 # learning rate
C = 10 # number of step from a copy of the weights of DQN onto Q_hat to the next

HIDDEN_LAYERS_SIZE = 128

loss_function =  nn.SmoothL1Loss()  # nn.MSELoss() # nn.SmoothL1Loss() 

################################################################################

# main #########################################################################

# GPU? # depending on the hardware it may even be better to run everything on the cpu
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

print(f'>>> DEVICE =  {device}')

# Environment creation 
env_name = "highway-fast-v0"  # "highway-v0"

env = gymnasium.make(env_name,
                     config={
                        'observation':{
                            'type': 'Kinematics',
                            "features": ["presence", "x", "y", "vx", "vy"],
                        },
                        
                        'action': {
                            'type': 'DiscreteMetaAction'
                        },
                        'lanes_count': LANES,
                        'absolute': False,
                        'duration': 40, "vehicles_count": 50},
                        render_mode = 'human'
                        )
env.unwrapped.config['high_speed_reward'] = 0.7
print('>>> ENVIRONMENT INITIALIZED')

# Initialize your model
agent = DQN.DQN_agent(input_size=STATE_DIMENSIONALITY, output_size=5, discount_factor = DISCOUNT_FACTOR ,loss_function = loss_function, lr = LR, device = device, hidden_size1=HIDDEN_LAYERS_SIZE, hidden_size2=HIDDEN_LAYERS_SIZE)
Q_hat = DQN.DQN_agent(input_size=25, output_size=5)

agent.train() # training mode
Q_hat.eval()  # evaluation mode

# moving the models to the gpu if available
agent.to(device=device)
Q_hat.to(device=device)

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

eps_values = []

print('Training progress: ')
# Training loop ################################################################
for t in tqdm.tqdm(range(MAX_STEPS)):
    episode_steps += 1
    # Select the action to be performed by the agent
    
    # eps decay
    eps = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * t / EPS_DECAY)
    eps_values.append(eps)

    # creating a torch state tensor
    state_tensor = torch.from_numpy(state)
    state_tensor = state_tensor.to(device=device)

    # compute the Q function for the given state (that is a vector containing an entry for each action)
    with torch.no_grad(): # no need to compute the gradient at this point
        Q_state_a = agent(state_tensor)
    
    # choosing an action following an epsilon greedy policy using the afore computed Q
    action = agent.act_eps_greedy(state_tensor=state_tensor, eps = eps)

    # Hint: take a look at the docs to see the difference between 'done' and 'truncated'
    # done: the agent entered in a terminal state (in this case crash)
    # truncated: the episode stops for external reasons (reached max duration of the episode)
    next_state, reward, done, truncated, _ = env.step(action)
    next_state = next_state.reshape(-1)
    next_state_tensor = torch.from_numpy(next_state)
    next_state_tensor = next_state_tensor.to(device=device)
   
    # Storing trnsition in the replay buffer
    replay_buffer.push(state_tensor, action, reward, next_state_tensor, done)

    state = next_state
    episode_return += reward

    # TRAINING #################################################################
    if t >= LEARNING_START: # checks that enough experience has been collected

        loss = agent.training_step(Q_hat = Q_hat, replay_buffer = replay_buffer, batch_size = BATCH_SIZE)

        training_steps += 1

        # storing data for plots but only starting when actual learning starts
        rewards_history.append(reward)
        loss_vals_history.append(loss.cpu().detach().numpy())

        if training_steps % C == 0: # every C steps: agent parameters -> Q_hat parameters
            Q_hat = copy.deepcopy(agent)
            Q_hat.eval() # set Q_hat to evaluation mode, gradient computation not needed

    ############################################################################

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


print('>>> TRAINING ENDED')

# saving the trained model for evaluation
if not os.path.exists('trained_models_weights'): os.makedirs('trained_models_weights')
torch.save(agent.state_dict(), 'trained_models_weights/trained_DQN_agent.pt')
print('>>> TRAINED DQN MODEL SAVED')


################################################################################

# TRAINING DATA PLOTS AND SAVING ###############################################

# saving data
if not os.path.exists('training_data/PPO'): os.makedirs('training_data/DQN')

np.savetxt('training_data/DQN/DQN_training_data_loss_and_rewards.csv', np.transpose([loss_vals_history, rewards_history]), header= 'Loss,rewards',delimiter = ',')
np.savetxt('training_data/DQN/DQN_training_data_returns_episode_length.csv', np.transpose([episode_return_history, episode_length_history]), header='returns,episode_length', delimiter=',')


# plotting training data

os.system('python data_plotter.py --input DQN')

env.close()

