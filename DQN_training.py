import gymnasium
import highway_env
import numpy as np
import torch
import random

import agent

# Set the seed and create the environment
np.random.seed(2119275)
random.seed(2119275)
torch.manual_seed(2119275)

# GPU?
use_cuda = torch.cuda.is_available()

if use_cuda:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

MAX_STEPS = int(2e4)  # This should be enough to obtain nice results, however feel free to change it
env_name = "highway-fast-v0"  # We use the 'fast' env just for faster training, if you want you can use "highway-v0"


LANES = 3

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

# Initialize your model
agent = agent.agent_model(input_size=25, output_size=5)
Q_hat = agent.agent_model(input_size=25,output_size= 5)

agent.to(device=device)
Q_hat.to(device=device)


state, _ = env.reset()
state = state.reshape(-1)
done, truncated = False, False

episode = 1
episode_steps = 0
episode_return = 0

BATCH_SIZE = 100

# Training loop
for t in range(MAX_STEPS):
    episode_steps += 1
    # Select the action to be performed by the agent
    state_tensor = torch.from_numpy(state(reshape(-1)))

    state_tensor.to(device)
    print(state_tensor)
    action = None


    # Hint: take a look at the docs to see the difference between 'done' and 'truncated'
    next_state, reward, done, truncated, _ = env.step(action)
    next_state = next_state.reshape(-1)

    # Store transition in memory and train your model
    raise NotImplementedError

    state = next_state
    episode_return += reward

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
