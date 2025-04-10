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

import DQN

# Set the seed and create the environment
np.random.seed(2119275)
random.seed(2119275)
torch.manual_seed(2119275)


# Constants and parameters #####################################################
MAX_STEPS = int(2e4)  # This should be enough to obtain nice results, however feel free to change it

LANES = 3

STATE_DIMENSIONALITY = 25 # 5 cars * 5 features

# BUFFER_MAX_SIZE = 15000
BATCH_SIZE = 128
LEARNING_START = 200


# epsilon decay parameters
EPS_START = 0.95
EPS_END = 0.05
EPS_DECAY = 1000

DISCOUNT_FACTOR = 0.8 # better results when 0.7 - 0.8 rather than > 0.8
LR = 5e-4 # learning rate
C = 50 # number of step from a copy of the weights of DQN onto Q_hat to the next

HIDDEN_LAYERS_SIZE = 128

loss_function =  nn.MSELoss()  # nn.MSELoss() # nn.SmoothL1Loss() 


################################################################################

# main #########################################################################

# GPU? # depending on the hardware it may even be better to run everything on the cpu
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
# device = 'cpu'
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
                        'lanes_count': LANES,
                        'absolute': False,
                        'duration': 40, "vehicles_count": 50},
                        # render_mode = 'human'
                        )

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

    # Store transition in memory and train your model
    
    # Storing trnsition in the replay buffer
    replay_buffer.push(state_tensor, action, reward, next_state_tensor, done)

    # removes the oldest sample when max size has benn exceeded, this keeps of reasonable
    # size the buffer and removes older and less relevant steps
    # if len(replay_buffer) > BUFFER_MAX_SIZE:
    #     replay_buffer.popleft()

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
torch.save(agent, 'trained_DQN_agent.pt')
print('>>> TRAINED DQN MODEL SAVED')


################################################################################

# TRAINING DATA PLOTS AND SAVING ###############################################

# saving data
np.savetxt('training_data/DQN/DQN_training_data_loss_and_rewards.csv', np.transpose([loss_vals_history, rewards_history]), header= 'Loss,rewards',delimiter = ',')
np.savetxt('training_data/DQN/DQN_training_data_returns_episode_length.csv', np.transpose([episode_return_history, episode_length_history]), header='returns,episode_length', delimiter=',')


# Create a 2x2 grid of subplots
fig, ((loss_plot, returns_plot), (rewards_plot, episode_length_plot)) = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))

# Set a main title for the entire figure
fig.suptitle('Training Data Plot', fontsize=16)

# Plot LOSS
loss_plot.set_title("LOSS")
loss_plot.plot(loss_vals_history, color='red')
loss_plot.set_xlabel("Episodes")
loss_plot.set_ylabel("Loss")

# Plot RETURNS
returns_plot.set_title("RETURNS")
returns_plot.plot(episode_return_history, color='blue')
returns_plot.set_xlabel("Episodes")
returns_plot.set_ylabel("Return")

# Plot REWARDS
rewards_plot.set_title("REWARDS")
rewards_plot.plot(episode_return_history, color='green')  # Consider plotting reward per step if available
rewards_plot.set_xlabel("Episodes")
rewards_plot.set_ylabel("Rewards")

# Plot EPISODE LENGTH
episode_length_plot.set_title('EPISODE LENGTH')
episode_length_plot.plot(episode_length_history, color='purple')
episode_length_plot.set_xlabel("Episodes")
episode_length_plot.set_ylabel("Length")

# Adjust layout to prevent overlap
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

eps_fig = plt.figure('Epsilon values')
plt.plot(eps_values)

# showing data
plt.show()
plt.close('all')
################################################################################

env.close()

print('>>> PROGRAM TERMINATED')