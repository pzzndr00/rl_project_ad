import gymnasium
import highway_env
import numpy as np
import torch
import torch.nn as nn
import random
import tqdm
import PPO
import memoryBuffer as mb

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


BATCH_SIZE = 500

# GPU? # depending on the hardware it may even be better to run everything on the cpu
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
# device = 'cpu'
print(f'>>> DEVICE =  {device}')


# Set the seed and create the environment
np.random.seed(2119275)
random.seed(2119275)
torch.manual_seed(2119275)

MAX_STEPS = int(2e4)  # This should be enough to obtain nice results, however feel free to change it
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

# initialization of the replay buffer
replay_buffer = mb.memory(MAX_STEPS)
print('>>> REPLAY BUFFER INITIALIZED')



# Initialize your model
agent = PPO.PPO_agent(state_size=STATE_DIMENSIONALITY, actions_set_cardinality=5, env = env, device = device)


state, _ = env.reset()
state = state.reshape(-1)
done, truncated = False, False

episode = 1
episode_steps = 0
episode_return = 0

# Training loop

for t in tqdm.trange(MAX_STEPS):
    episode_steps += 1

    # conversion of the state numpy array into a pyTorch tensor
    state_tensor = torch.from_numpy(state).to(device=device)

    # Select the action to be performed by the agent
    action,_ = agent.act(state_tensor=state_tensor)

    next_state, reward, done, truncated, _ = env.step(action)
    next_state = next_state.reshape(-1)

    # next state tensor
    next_state_tensor = torch.from_numpy(next_state).to(device=device)

    state = next_state
    episode_return += reward


    # Store transition in memory and train your model
    replay_buffer.push(state_tensor, action, reward, next_state_tensor, done)

    agent.training_step(batch_size = BATCH_SIZE, replay_buffer = replay_buffer)


    if done or truncated:
        # print(f"Total T: {t} Episode Num: {episode} Episode T: {episode_steps} Return: {episode_return:.3f}")

        # Save training information and model parameters
        

        state, _ = env.reset()
        state = state.reshape(-1)
        episode += 1
        episode_steps = 0
        episode_return = 0

env.close()