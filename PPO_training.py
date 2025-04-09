import gymnasium
import highway_env
import numpy as np
import torch
import random

import PPO

# Constants and parameters #####################################################
MAX_STEPS = int(2e4)  # This should be enough to obtain nice results, however feel free to change it

LANES = 3

STATE_DIMENSIONALITY = 25 # 5 cars * 5 features

# epsilon decay parameters
EPS_START = 0.95
EPS_END = 0.05
EPS_DECAY = MAX_STEPS / 15

DISCOUNT_FACTOR = 0.8 # better results when 0.7 - 0.8 rather than > 0.8
LR = 5e-4 # learning rate

SAMPLES_TRAJECTORY = 500 # number of step from a copy of the weights of DQN onto Q_hat to the next

HIDDEN_LAYERS_SIZE = 128

loss_function =  nn.SmoothL1Loss()  # nn.MSELoss() # nn.SmoothL1Loss() 

# Set the seed and create the environment
np.random.seed(2119275)
random.seed(2119275)
torch.manual_seed(2119275)

MAX_STEPS = int(2e4)  # This should be enough to obtain nice results, however feel free to change it
env_name = "highway-fast-v0"  # We use the 'fast' env just for faster training, if you want you can use "highway-v0"

env = gymnasium.make(env_name,
                     config={'action': {'type': 'DiscreteMetaAction'}, 'duration': 40, "vehicles_count": 50})

# Initialize your model
agent = PPO.PPO_agent(state_size=STATE_DIMENSIONALITY, actions_set_cardinality=5, env = env, device = device) # da rivedere solo provissoria
raise NotImplementedError


state, _ = env.reset()
state = state.reshape(-1)
done, truncated = False, False

episode = 1
episode_steps = 0
episode_return = 0

# Training loop
trj_set = PPO.trj_set()

for t in range(MAX_STEPS):
    episode_steps += 1

    # conversion of the state numpy array into a pyTorch tensor
    state_tensor = torch.from_numpy(state).to(device=device)

    # Select the action to be performed by the agent
    action = agent.act(state_tensor=state_tensor)

    next_state, reward, done, truncated, _ = env.step(action)
    next_state = next_state.reshape(-1)

    next_state_tensor = torch.from_numpy(next_state).to(device=device)

    state = next_state
    episode_return += reward


    # Store transition in memory and train your model
    probs = agent.actor(state_tensor).cpu().detach().numpy() # temporary not clear what to save exactly yet
    trj.append_transition(state = state_tensor, action = action, reward = reward, next_state = next_state_tensor, probs = probs, done = done)

    # every tot iteration train
    if t%500 == 0: 
        raise NotImplementedError

    if done or truncated:
        print(f"Total T: {t} Episode Num: {episode} Episode T: {episode_steps} Return: {episode_return:.3f}")

        # Save training information and model parameters
        raise NotImplementedError

        state, _ = env.reset()
        state = state.reshape(-1)
        episode += 1
        episode_steps = 0
        episode_return = 0

env.close()