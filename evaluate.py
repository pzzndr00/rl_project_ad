import gymnasium
import highway_env
import numpy as np
import torch
import random

device = 'cpu'

# Set the seed and create the environment
np.random.seed(2119275)
random.seed(2119275)
torch.manual_seed(2119275)

env_name = "highway-v0"

env = gymnasium.make(env_name,
                     config={'action': {'type': 'DiscreteMetaAction'}, "lanes_count": 3, "ego_spacing": 1.5},
                     render_mode='human')

# Initialize your model and load parameters
# agent = ag.AgentModel(input_size=25, output_size=5)
DQN_agent = torch.load('trained_DQN_network.pt', weights_only = False) # weights_only true?
DQN_agent.to(device=device)

# Evaluation loop
state, _ = env.reset()
state = state.reshape(-1)
done, truncated = False, False

episode = 1
episode_steps = 0
episode_return = 0

while episode <= 10:
    episode_steps += 1
    state_tensor = torch.from_numpy(state)
    state_tensor = state_tensor.to(device=device)
    # Select the action to be performed by the agent
    with torch.no_grad():
        Q_state_a = DQN_agent(state_tensor)
    action = torch.argmax(Q_state_a)
    # print(f'Q_state_a = {Q_state_a.cpu().detach().numpy()}')
    # print(f'action: {action}')

    # Hint: take a look at the docs to see the difference between 'done' and 'truncated'
    state, reward, done, truncated, _ = env.step(action)
    state = state.reshape(-1)
    env.render()

    episode_return += reward

    if done or truncated:
        print(f"Episode Num: {episode} Episode T: {episode_steps} Return: {episode_return:.3f}, Crash: {done}")

        state, _ = env.reset()
        state = state.reshape(-1)
        episode += 1
        episode_steps = 0
        episode_return = 0

# env.close()
