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

STATE_DIMENSIONALITY = 25 # 5 cars * 5 features

BATCH_SIZE = 128
LEARNING_START = 200 


# epsilon decay parameters
EPS_START = 0.95
EPS_END = 0.05
EPS_DECAY = 1000

DISCOUNT_FACTOR = 0.8
LR = 5e-4 # learning rate
C = 50 # number of step from a copy of the weights of DQN onto Q_hat to the next
loss_function =  nn.SmoothL1Loss() # nn.MSELoss() # nn.SmoothL1Loss() 

MAX_STEPS = int(2e4)  # This should be enough to obtain nice results, however feel free to change it

LANES = 3

################################################################################

# main #########################################################################

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
                        'lanes_count': LANES,
                        'absolute': False,
                        'duration': 40, "vehicles_count": 50},
                        render_mode = 'human'
                        )

print('>>> ENVIRONMENT INITIALIZED')

# Initialize your model
agent = DQN.DQN_network(input_size=25, output_size=5)
Q_hat = DQN.DQN_network(input_size=25, output_size=5)

optimizer = torch.optim.AdamW(agent.parameters(), lr = LR)

Q_hat.eval()

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

print('Training progress: ')
# Training loop ################################################################
for t in tqdm.tqdm(range(MAX_STEPS)):
    episode_steps += 1
    # Select the action to be performed by the agent
    
    # eps decay
    eps = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * t / EPS_DECAY)

    # creating a torch state tensor
    state_tensor = torch.from_numpy(state)
    state_tensor = state_tensor.to(device=device)

    # compute the Q function for the given state (that is a vector containing an entry for each action)
    with torch.no_grad(): # no need to compute the gradient at this point
        Q_state_a = agent(state_tensor)
    
    # choosing an action following an epsilon greedy policy using the afore computed Q
    available_actions = env.unwrapped.get_available_actions()
    action = DQN.eps_greedy_policy(actions_values = Q_state_a.cpu().detach().numpy(), available_actions = available_actions, eps = eps)

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

    state = next_state
    episode_return += reward

    # TRAINING #################################################################
    if t >= LEARNING_START: # checks that enough experience has been collected

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

        Q_values = torch.zeros(BATCH_SIZE)
        Q_values = Q_values.to(device = device)

        ########################################################################

        y = torch.zeros(BATCH_SIZE)
        y = y.to(device=device)
 
        for i in range(BATCH_SIZE):

            if done_batch[i]:
                y[i] = reward_batch[i]
            else:
                Q_hat_next = Q_hat(next_state_batch[i])
                max_Q_hat_next = Q_hat_next.max().item()

                y[i] = reward_batch[i] + DISCOUNT_FACTOR * max_Q_hat_next

            # for each sample compute the current agent action values
            Q_values[i] = agent(state_batch[i])[action_batch[i]]
        
        # computation of the loss and training step
        loss = loss_function(Q_values, y)

        # print(f'loss at iteration t = {loss}')

        optimizer.zero_grad()
        loss.backward()

        # torch.nn.utils.clip_grad_value_(agent.parameters(), 100)
        optimizer.step()

        training_steps += 1

        # storing data for plots
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
torch.save(agent, 'trained_DQN_network.pt') 
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



# showing data
plt.show()
################################################################################

# env.close()


print('>>> PROGRAM TERMINATED')