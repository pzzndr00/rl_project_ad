import gymnasium
import highway_env
import numpy as np
import torch
import torch.nn as nn
import random
import Agent
import matplotlib.pyplot as plt 

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
agent = Agent.Agent(env, batch_size=256, actor_rep=10, critic_rep=10)
optimizer = torch.optim.AdamW(agent.Q.parameters(), lr = 1e-4)
# move the networks to the gpu if available
# agent.Q.to(device=device)
# agent.Q_hat.to(device=device)
# agent.actor.to(device=device)
# agent.critic.to(device=device)

state, _ = env.reset()
state = np.squeeze(state.reshape(-1))

done, truncated = False, False

episode = 1
episode_steps = 0
episode_return = 0

# Training loop

class sample():

    def __init__(self, state, action, reward, next_state, done):
        super(sample, self).__init__()

        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state

loss_function = nn.MSELoss()

rewards_history = []
rewards = []
experience_buffer = []
eps = 0.9
FORGETTING_FACTOR = 0.99

for t in range(MAX_STEPS):
    
    episode_steps += 1
    # Select the action to be performed by the agent
    state_tensor = torch.from_numpy(state)

    # state_tensor = state_tensor.to(device=device)
    # print(f'state tensor device = {state_tensor.get_device()}')

    actor_out = agent.actor(x = state_tensor)
    # print(actor_out)
    action = Agent.policy(actor_out, env)
    # print(f'Selected action at step {t}: {action}')
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

    experience_buffer.append(sample(state, action, reward, next_state, done))
    
    # training step ###########################################################
    BATCH_SIZE = 10
    C = 25

    if len(experience_buffer) > BATCH_SIZE:

        batch = random.sample(experience_buffer, BATCH_SIZE)
        y = np.zeros(BATCH_SIZE)
        Q = np.zeros(BATCH_SIZE)

        
        actions = []
        for i, sample in enumerate(batch):
            actions.append([sample.action])
            if done:
                y[i] = sample.reward
            else:
                y[i] = sample.reward + FORGETTING_FACTOR*(torch.argmax(agent.Q_hat(torch.from_numpy(next_state))))
            
            Q[i] = agent.Q(torch.from_numpy(sample.state)).detach().numpy()[action]
        
        print(y.shape, Q.shape)
        loss = loss_function(torch.from_numpy(y),torch.from_numpy(Q))
        agent.Q.zero_grad()
        loss.backward()

        optimizer.step()

        agent.Q_hat = agent.Q.copy()

    ###########################################################################

    rewards.append(reward)
    # Store transition in memory and train your model
    state = next_state
    episode_return += reward

    if done or truncated:
        print(f"Total T: {t} Episode Num: {episode} Episode T: {episode_steps} Return: {episode_return:.3f}")

        # Save training information and model parameters
        
        state, _ = env.reset()
        state = state.reshape(-1)
        episode += 1

        
        rewards_history.append(np.mean(np.array(rewards)))
        rewards = []
        
        #print(episode)
        episode_steps = 0
        episode_return = 0


    
rewards_history = np.array(rewards_history)
plt.plot(rewards_history)
plt.show()


env.close()

