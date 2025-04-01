import gymnasium
import highway_env
import numpy as np
import torch
import random
import Agent

ACTIIONS_ALL = highway_env.envs.common.action.DiscreteMetaAction.ACTIONS_ALL
# GPU?
use_cuda = torch.cuda.is_available()

if use_cuda:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print(f'selected device = {device}')

# Set the seed and create the environment
np.random.seed(2119275)
random.seed(2119275)
torch.manual_seed(2119275)

MAX_STEPS = int(2e4)  # This should be enough to obtain nice results, however feel free to change it
EPISODES = 180 # two hours
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
agent = Agent.Agent(env, batch_size=256, actor_rep=10, critic_rep=10, device = device)

state, _ = env.reset()
state = np.squeeze(state.reshape(-1))

done, truncated = False, False

episode = 1
episode_steps = 0
episode_return = 0

# Training loop

#rewards_history

while episode < EPISODES:

    for t in range(MAX_STEPS):
        episode_steps += 1
        # Select the action to be performed by the agent
        state_tensor = torch.from_numpy(state)

        state_tensor = state_tensor.to(device=device)
        # print(f'state tensor device = {state_tensor.get_device()}')

        actor_out = agent.actor.forward(x = state_tensor)
        # print(actor_out)
        action = Agent.policy(actor_out, env)
        print(f'Selected action at step {t}: {action}')
        # print(f'action tensor device = {action.get_device()}')
        # print(f'Chosen action = {action}')

        # Hint: take a look at the docs to see the difference between 'done' and 'truncated'
        # 'truncated' is true when the episodes terminates because of external factor (like time limit has
        # been reached when time is not part of the state space)
        # 'done' is true when the episode has terminated because the agent has entered
        # a terminal state (like car crash)
        # https://farama.org/Gymnasium-Terminated-Truncated-Step-API 

        next_state, reward, done, truncated, _ = env.step(action)
        next_state = next_state.reshape(-1)

        # Store transition in memory and train your model
        state = next_state
        episode_return += reward

        if done or truncated:
            print(f"Total T: {t} Episode Num: {episode} Episode T: {episode_steps} Return: {episode_return:.3f}")

            # Save training information and model parameters
            

            state, _ = env.reset()
            state = state.reshape(-1)
            episode += 1
            episode_steps = 0
            episode_return = 0

env.close()
