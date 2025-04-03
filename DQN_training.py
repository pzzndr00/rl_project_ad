import gymnasium
import highway_env
import numpy as np
import torch
import torch.nn as nn
import random
import math
import copy
from collections import namedtuple, deque
import tqdm

import agent as ag

import matplotlib.pyplot as plt

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

print(f'>>> DEVICE =  {device}')


# Replay buffer ################################################################

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class memory(object):

    def __init__(self, capacity):
        """Initiates the replay buffer """

        self.memory = deque([], maxlen = capacity) # deque datatype


    def push(self, *args):
        """Saves a transition"""

        self.memory.append(Transition(*args))


    def sample(self, batch_size):
        """Samples a batch of size batch_size from the replay buffer"""

        return random.sample(self.memory, batch_size)


    def __len__(self):
        return len(self.memory)


# End Replay buffer ############################################################

# Constants and parameters #####################################################

BATCH_SIZE = 32

DISCOUNT_FACTOR = 0.8

STATE_DIMENSIONALITY = 25 # 5 cars * 5 features

# epsilon
EPS_START = 0.99
EPS_END = 0.05
EPS_DECAY = 1000

LR = 5e-4

MAX_STEPS = int(2e4)  # This should be enough to obtain nice results, however feel free to change it
LANES = 3

C = 50

loss_function = nn.MSELoss()
################################################################################


# main #########################################################################

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
agent = ag.Q_network(input_size=25, output_size=5)
Q_hat = ag.Q_network(input_size=25, output_size=5)

optimizer = torch.optim.AdamW(agent.parameters(), lr = LR)

Q_hat.eval()

# moving the models to the gpu if available
agent.to(device=device)
Q_hat.to(device=device)

# initialization of the replay buffer
replay_buffer = memory(MAX_STEPS)
print('>>> REPLAY BUFFER INITIALIZED')

state, _ = env.reset()
state = np.squeeze(state.reshape(-1))
done, truncated = False, False

episode = 1
episode_steps = 0
episode_return = 0

training_steps = 0

# training data arrays initialization
loss_values_history = []
episode_return_history = []

episode_avg_reward = 0
episode_avg_reward_history = []

# Training loop
for t in tqdm.tqdm(range(MAX_STEPS)):
    episode_steps += 1
    # Select the action to be performed by the agent
    
    # eps decay
    eps = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * t / EPS_DECAY)
    # print(eps)
    # creating a torch state tensor to feed to the agent's neural network
    state_tensor = torch.from_numpy(state)
    state_tensor = state_tensor.to(device=device)
    # print(f'>>> {state_tensor.device}')
    with torch.no_grad(): # no need to compute the gradient at this point
        Q_state_a = agent(state_tensor)
    
    # choosing an action following an epsilon greedy policy
     
    available_actions = env.unwrapped.get_available_actions()
    action = ag.eps_greedy_policy(actions_values = Q_state_a.cpu().detach().numpy(), available_actions = available_actions, eps = eps)
    # print(f'action at step {t} is: {action}')

    # Hint: take a look at the docs to see the difference between 'done' and 'truncated'
    # done: the agent entered in a terminal state (in this case crash)
    # truncated: the episode stops for external reasons (reached max duration of the episode)
    next_state, reward, done, truncated, _ = env.step(action)
    episode_avg_reward += reward
    next_state = np.squeeze(next_state.reshape(-1))
    next_state_tensor = torch.from_numpy(next_state)
    next_state_tensor = next_state_tensor.to(device=device)

    # Store transition in memory and train your model
    
    # Storing trnsition in memory
    replay_buffer.push(state_tensor, action, reward, next_state_tensor, done)

    state = next_state
    episode_return += reward

    # TRAINING #################################################################
    if t >= BATCH_SIZE:
        # if training_steps == 0: print('>>> TRAINING STARTED')

        batch = replay_buffer.sample(batch_size = BATCH_SIZE)

        batch_zip = Transition(*zip(*batch))

        y = torch.zeros(BATCH_SIZE)
        y = y.to(device=device)

        state_batch = torch.cat(batch_zip.state).reshape((-1, STATE_DIMENSIONALITY))       
        action_batch = np.array(batch_zip.action)
        reward_batch = np.array(batch_zip.reward)
        next_state_batch = torch.cat(batch_zip.next_state).reshape((-1, STATE_DIMENSIONALITY))
        done_batch = list(batch_zip.done)
        
        # moving tensors to device
        state_batch = state_batch.to(device=device)
        #action_batch.to(device=device)
        #reward_batch.to(device=device)
        next_state_batch = next_state_batch.to(device=device)
        #done_batch.to(device=device)

        Q_values = torch.zeros(BATCH_SIZE)
        Q_values = Q_values.to(device = device)

        # there are for sure better ways than a for for doing this but for now... 
        for i in range(BATCH_SIZE):

            if done_batch[i]:
                y[i] = reward_batch[i]
            else:
                Q_hat_next = Q_hat(next_state_batch[i])
                max_Q_hat_next = Q_hat_next.max().item() # should be a number and hence should not be on the gpu

                y[i] = reward_batch[i] + DISCOUNT_FACTOR * max_Q_hat_next

            # for each sample compute the current agent action values
            Q_values[i] = agent(state_batch[i])[action_batch[i]]
            

        
        # computation of the loss and training step
        loss = loss_function(Q_values, y)
        loss_values_history.append(loss.cpu().detach().numpy())
        # print(f'loss at iteration t = {loss}')

        optimizer.zero_grad()
        loss.backward()

        # torch.nn.utils.clip_grad_value_(agent.parameters(), 100)
        optimizer.step()

        training_steps += 1

        if training_steps % C == 0: # every C training steps copies Q hat and agent parameter
            Q_hat = copy.deepcopy(agent)
            Q_hat.to(device=device)
            Q_hat.eval()
    ############################################################################

    if done or truncated:
        #print(f"Total T: {t} Episode Num: {episode} Episode T: {episode_steps} Return: {episode_return:.3f}")

        # Save training information and model parameters
        episode_return_history.append(episode_return)
        episode_avg_reward_history.append(episode_avg_reward/episode_steps)
        episode_avg_reward = 0

        state, _ = env.reset()
        state = state.reshape(-1)
        episode += 1
        episode_steps = 0
        episode_return = 0



print('>>> TRAINING ENDED')
# saving the trained model for evaluation
torch.save(agent, 'trained_DQN_agent.pt') 
print('>>> TRAINED DQN MODEL SAVED')

# training data plot
fig, (loss_plot, returns_plot, rewards_plot) = plt.subplots(1, 3)

fig.suptitle('Training data plot')


loss_plot.set_title("LOSS")
loss_plot.plot(loss_values_history)

returns_plot.set_title("RETURNS")
returns_plot.plot(episode_return_history)

rewards_plot.set_title("AVG REWARDS")
rewards_plot.plot(episode_avg_reward_history)

plt.show()


env.close()

print('>>> PROGRAM TERMINATED')